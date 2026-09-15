import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor


def compute_segmentation_loss(
    bce_fn, dice_fn, outputs, targets,
    boundary_weight: float = 0.0,
    lovasz_weight: float = 0.0,
):
    """Compute BCE + Dice segmentation loss with NaN sentinel.

    When targets contains "valid_mask" (a [B] float tensor), applies
    per-sample segmentation validity masking with sum / (valid_count + eps).
    Samples with valid_mask=0 contribute zero gradient to the seg head.
    boundary_weight/lovasz_weight > 0 时额外外挂两项(阶段2 seg-only):
    返回值变为 (loss_seg, extra_dict),extra_dict 含 "seg_boundary"/
    "seg_lovasz" 分项(权重=0 时仍返回 dict,键值为 0.0 张量——消费端
    无条件 update,不查权重,防"接线被删 JSONL 静默丢分项")。

    Falls back to original scalar BCE+Dice path when "valid_mask" is absent.
    """
    pred = outputs["mask"]
    gt = targets["mask"]

    # ── Masked path (per-sample valid_mask) ──────────────────────────────────
    if "valid_mask" in targets:
        v = targets["valid_mask"]  # [B] float tensor
        _eps = 1e-8

        # Per-sample BCE: reduce over H×W, keep batch dim
        bce_per = F.binary_cross_entropy_with_logits(
            pred, gt, reduction="none"
        ).mean(dim=[1, 2, 3])  # [B]

        # Per-sample Dice
        pred_sig = torch.sigmoid(pred)
        inter = (pred_sig * gt).sum(dim=[1, 2, 3])      # [B]
        union = pred_sig.sum(dim=[1, 2, 3]) + gt.sum(dim=[1, 2, 3])  # [B]
        dice_per = 1.0 - (2.0 * inter + _eps) / (union + _eps)       # [B]

        loss_per = bce_per + dice_per  # [B]
        if boundary_weight > 0.0:
            _b_per = boundary_loss_per_sample(pred_sig, gt)           # [B]
            loss_per = loss_per + boundary_weight * _b_per
        else:
            _b_per = torch.zeros_like(loss_per)
        if lovasz_weight > 0.0:
            _l_per = lovasz_hinge_per_sample(pred, gt)                # [B]
            loss_per = loss_per + lovasz_weight * _l_per
        else:
            _l_per = torch.zeros_like(loss_per)

        loss_seg = (loss_per * v).sum() / (v.sum() + _eps)
        extra = {
            "seg_boundary": (_b_per * v).sum() / (v.sum() + _eps),
            "seg_lovasz": (_l_per * v).sum() / (v.sum() + _eps),
        }
    else:
        # ── Legacy path — behaviour unchanged ──────────────────────────────
        loss_bce_seg = bce_fn(pred, gt)
        loss_dice_seg = dice_fn(pred, gt)
        loss_seg = loss_bce_seg + loss_dice_seg
        # legacy 无 valid_mask 路径不支持新损失项(官方训练恒有 valid_mask):
        # 静默丢变量 = 空微调被误读为损失无效(r7 事故同型),显式 raise。
        if boundary_weight > 0.0 or lovasz_weight > 0.0:
            raise ValueError(
                "boundary/lovasz 损失需要 valid_mask(官方训练路径);"
                "legacy 无屏蔽路径不支持,勿在旧数据集上开这两项")
        extra = {
            "seg_boundary": torch.zeros((), device=loss_seg.device,
                                        dtype=loss_seg.dtype),
            "seg_lovasz": torch.zeros((), device=loss_seg.device,
                                      dtype=loss_seg.dtype),
        }

    if torch.isnan(loss_seg):
        print(
            f"  [NaN Guard] loss_seg is NaN! "
            f"mask range=[{pred.min().item():.4f}, {pred.max().item():.4f}]",
            flush=True,
        )
    return loss_seg, extra


def boundary_loss_per_sample(pred_sig, gt):
    """Boundary Loss(Kervadec et al. 2019)逐样本版,带符号距离场。

    L_b = mean(φ(x) · p(x)),φ = **带符号**归一化距离场:GT 内为负(到边界
    距离)、GT 外为正(到前景距离)。最小化 → 概率质量被推向 φ<0 的 GT 内部、
    离边界越远越负——边界像素的梯度信号不再被海量背景稀释。

    🔴 **第一版教训(2026-09-13 实测)**:用纯内部距离变换(内部为正)时方向
    恰好反了——对齐预测的 loss(0.144)反而大于平移预测(0.027),因为该形式
    奖励"概率质量集中在 GT 中心"而非"对齐边界"。带符号版方向正确
    (test_boundary_aligned_lower_than_shifted 钉死)。

    硬约束(plan 阶段2):
    - GT 距离变换**在损失内计算**,勿进 data pipeline(mixup/cutmix 后的
      mask 拿不到预计算,缓存链也不该为损失买单);
    - 🔴 内部强制 float32:autocast(bfloat16) 区内 pred_sig 是 bf16,
      φ 计算与乘积/mean 在 float32 累加,避免 262144 项求和的精度损失
      (注意:上转救的是归约精度,已被截断的输入精度回不来);
    - 归一化 [−1,1](除以 |φ|max):小病灶 dist 仅几个像素,不归一化则本项
      对大病灶天然权重更大,与"小结节边界"目标相反;
    - GT 全零样本(阴性)→ 0(本项只管前景边界,阴性监督归 BCE);
    - φ 是仅依赖 GT 的常数,对 p 可导,梯度经 pred_sig 自然回流;
    - EDT 用 ThreadPoolExecutor 并行(2026-09-14 Gate2 F3):scipy EDT 释放
      GIL,逐样本 CPU 循环 304ms→75ms/批(B=16@512²,3.47×);云端基线
      0.156s/step,串行版会把 step 拉到 ~1.9×。
    - 🔴 **mask 先二值化再算距离场**(2026-09-14 修复轮专审 F1 + 三批专审
      F-2 修正表述):mixup 软 mask 会产生分数值,`distance_transform_edt
      (非零)` 把任意非零当前景 → 覆盖率>50% 时 mean(φ)<0,**梯度反向**
      (奖励膨胀预测)且 |L| 塌向 0。⚠️ 两条实测限定:
      ① 二值化后的行为是"**整项归零**"而非"均值口径"——软值全域 ≤0.5
      (如 0.42×70% 面积)时 (g>0.5) 恒 False → 该样本贡献 0(阴性监督
      归 BCE);
      ② **覆盖>50% 梯度反向是带符号距离场的固有几何**(φ 外正内负,前景
      过半则均值必负,二值化救不了)——真数据病灶占 0.1-3% 不越界;真实
      mixup 掩膜取值只落在 {0,1-λ,λ,1},λ≠0.5 时二值化结果逐位等于多数派
      掩膜,故本数据集安全;**勿把此损失用于前景占比可能过半的大目标**。
      - lovasz 与本项对软 GT 的口径**有意不同**:lovasz 不二值化(软 IoU
      语义,BCE/Dice 也是纯软)——三项各吃各的口径,非遗漏。
    """
    import numpy as np
    from scipy import ndimage

    B = pred_sig.shape[0]
    device = pred_sig.device

    def _one(b):
        g = gt[b, 0].detach().float().cpu().numpy()
        if g.max() == 0.0:
            return None  # GT 全零占位,下面统一填 0
        # 🔴 二值化:软 mask(mixup 分数值)先 >0.5 归一到 0/1,防覆盖>50%
        # 时距离场符号翻转(见 docstring F1 条目)。
        g = (g > 0.5).astype(g.dtype)
        if g.max() == 0.0:
            return None
        # 带符号距离场:GT 外正(背景点到前景的最近距离),GT 内负(到边界)。
        dist_out = ndimage.distance_transform_edt(g == 0)
        dist_in = ndimage.distance_transform_edt(g)
        phi = dist_out - dist_in
        dmax = np.abs(phi).max()
        if dmax > 0:
            phi = phi / dmax
        return torch.from_numpy(np.ascontiguousarray(phi, dtype="float32"))

    if B == 0:
        return torch.zeros(0, device=device, dtype=torch.float32)
    with ThreadPoolExecutor(max_workers=max(1, min(B, 8))) as ex:
        phis = list(ex.map(_one, range(B)))
    losses = []
    for b, phi in enumerate(phis):
        if phi is None:
            losses.append(torch.zeros((), device=device, dtype=torch.float32))
            continue
        phi_t = phi.to(device)
        # 🔴 pred_sig 切片 .float() 不 detach——梯度必须回流(φ 为常数)。
        losses.append((phi_t * pred_sig[b, 0].float()).mean())
    return torch.stack(losses)


def lovasz_hinge_per_sample(logits, gt):
    """Lovász-Hinge 逐样本版(Berman et al. 2018 二值形式)。

    IoU 的 convex surrogate:像素按 error 降序排列,以 Lovász 梯度加权
    ReLU(errors) 求和——直接优化 IoU 家族,与官方 Dice 指标同族。
    场景:region loss 已收敛但 hard-Dice/IoU 差一截的小病灶。

    🔴 排序/hinge 强制 float32:bf16 尾数 3 位会打乱 errors 排序,
    排序错即不再是 IoU 的凸上界(梯度错向)。
    标准口径(与参考实现一致):errors = 1 - logits·signs,signs = 2y-1。
    GT 全零样本 → 0。
    """
    B = logits.shape[0]
    device = logits.device
    if B == 0:
        return torch.zeros(0, device=device, dtype=torch.float32)
    losses = []
    for b in range(B):
        lg = logits[b].float().reshape(-1)          # [N] float32
        g = gt[b].float().reshape(-1)               # [N] 0/1
        gts = float(g.sum())
        if gts == 0.0:
            losses.append(torch.zeros((), device=device, dtype=torch.float32))
            continue
        signs = 2.0 * g - 1.0
        errors = 1.0 - lg * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        g_sorted = g[perm]
        # 官方口径(Berman lovasz_grad):inclusive jaccard 曲线再**差分**
        # (官方源码 jaccard[1:p] -= jaccard[0:-1])。🔴 第一版教训(2026-09-14
        # Gate2 三路独立实锤):只做 exclusive cumsum 平移、漏掉差分步——拿到的是
        # jaccard 曲线本身而非其梯度,①权重顺序反(误差最大像素权重=0);
        # ②量级 O(像素数)(512² 单样本 2.46e5,官方恒 O(1),half=1.0/worst=11.0);
        # ③梯度方向都不一致(cos≈0.40)。差分步等价实现:末尾补 1.0 再做一阶差分,
        # 已与官方参考实现逐 case 逐位对拍(0/1.0/11.0/randn 全等)。
        gtm_cum = g_sorted.cumsum(0) - g_sorted
        gcm_cum = (1.0 - g_sorted).cumsum(0) - (1.0 - g_sorted)
        intersection = gts - gtm_cum
        union = gts + gcm_cum
        gm = 1.0 - intersection / union
        padded = torch.cat([gm, torch.ones(1, dtype=gm.dtype, device=gm.device)])
        gradient = padded[1:] - padded[:-1]
        losses.append(torch.dot(torch.relu(errors_sorted), gradient))
    return torch.stack(losses)
