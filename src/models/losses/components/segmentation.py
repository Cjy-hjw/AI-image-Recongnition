import math

import numpy as np
from scipy import ndimage

import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor


def compute_segmentation_loss(
    bce_fn, dice_fn, outputs, targets,
    boundary_weight: float = 0.0,
    lovasz_weight: float = 0.0,
    ccdice_weight: float = 0.0,
):
    """Compute BCE + Dice segmentation loss with NaN sentinel.

    When targets contains "valid_mask" (a [B] float tensor), applies
    per-sample segmentation validity masking with sum / (valid_count + eps).
    Samples with valid_mask=0 contribute zero gradient to the seg head.
    boundary_weight/lovasz_weight/ccdice_weight > 0 时额外外挂三项
    (阶段2 seg-only / plan 任务C):返回值变为 (loss_seg, extra_dict),
    extra_dict 含 "seg_boundary"/"seg_lovasz"/"seg_ccdice" 分项(权重=0 时
    仍返回 dict,键值为 0.0 张量——消费端无条件 update,不查权重,防
    "接线被删 JSONL 静默丢分项";n_cc_* 计数键仅权重>0 时出现)。

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
        if ccdice_weight > 0.0:
            _c_per, _cc_diag = cc_dice_ce_per_sample(pred, gt)        # [B], diag
            loss_per = loss_per + ccdice_weight * _c_per
        else:
            _c_per = torch.zeros_like(loss_per)
            _cc_diag = None

        loss_seg = (loss_per * v).sum() / (v.sum() + _eps)
        extra = {
            "seg_boundary": (_b_per * v).sum() / (v.sum() + _eps),
            "seg_lovasz": (_l_per * v).sum() / (v.sum() + _eps),
            "seg_ccdice": ((_c_per * v).sum() / (v.sum() + _eps)
                           if _c_per is not None
                           else torch.zeros((), device=loss_seg.device,
                                            dtype=loss_seg.dtype)),
        }
        # CC 覆盖诊断(权重>0 才记;JSONL extra 消费方=_step_diagnostics,
        # 对标量逐键 round;缺键=权重 0,操作者不会误读)。
        if _cc_diag is not None:
            n_k_pos, n_soft, n_k0 = _cc_diag
            extra["n_cc_supervised"] = int(n_k_pos)      # K≥1 且非软 GT 的样本数
            extra["n_cc_soft_gt"] = int(n_soft)          # 软 GT 归 0 的样本数(mixup)
            extra["n_cc_k0_fallback"] = int(n_k0)        # K=0(GT全零)样本数
    else:
        # ── Legacy path — behaviour unchanged ──────────────────────────────
        loss_bce_seg = bce_fn(pred, gt)
        loss_dice_seg = dice_fn(pred, gt)
        loss_seg = loss_bce_seg + loss_dice_seg
        # legacy 无 valid_mask 路径不支持新损失项(官方训练恒有 valid_mask):
        # 静默丢变量 = 空微调被误读为损失无效(r7 事故同型),显式 raise。
        if boundary_weight > 0.0 or lovasz_weight > 0.0 or ccdice_weight > 0.0:
            raise ValueError(
                "boundary/lovasz/ccdice 损失需要 valid_mask(官方训练路径);"
                "legacy 无屏蔽路径不支持,勿在旧数据集上开这三项")
        extra = {
            "seg_boundary": torch.zeros((), device=loss_seg.device,
                                        dtype=loss_seg.dtype),
            "seg_lovasz": torch.zeros((), device=loss_seg.device,
                                      dtype=loss_seg.dtype),
            "seg_ccdice": torch.zeros((), device=loss_seg.device,
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


# ─────────────────────────────────────────────────────────────────────────────
# CC-DiceCE(plan 任务 C:实例级局部 DiceCE,默认关,权重=0 零回归)
# ─────────────────────────────────────────────────────────────────────────────
def _cc_instances_cpu(g: "np.ndarray") -> list:
    """GT 单样本的连通域实例列表(CPU,detached 数据)。

    **4 连通**(十字结构,ndimage.label 默认)。🔴 P1 订正(2026-09-18
    Gate2 审查实锤):docstring 曾写"8 连通"并归因于"ndimage.label 默认
    结构"——后半句对、前半句错,默认结构是交叉(4 连通)非 8 连通;
    对角接触的两点默认下是 2 个实例(实测)。**有意保持 4 连通**:医学
    GT 的两个病灶仅对角相接不算同一病灶(粘连合并问题归预测侧机制2),
    4 连通把"边缘相接"才并为一实例,与临床直觉一致。structure 显式
    传入钉死语义,防未来 scipy 改默认值时静默漂移。
    返回每个实例的 bool mask(按 label 升序)。空 GT 返回 []。scipy 缺失
    会在 import 期 fail-loud(静默跳过=空微调被误读"CC 无效",r7 同型)。
    """
    # 十字结构 = 4 连通(显式,不依赖 scipy 默认)
    _structure4 = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=bool)
    labeled, n_inst = ndimage.label(g > 0.5, structure=_structure4)
    if n_inst == 0:
        return []
    return [(labeled == k) for k in range(1, n_inst + 1)]


def _instance_roi_slices(inst: "np.ndarray",
                         all_fg: "np.ndarray",
                         pad: int) -> tuple:
    """实例 bbox 四边外扩 pad(裁到图内)的切片窗口 + 其他实例前景并集。

    🔴 P2 修复(2026-09-18 审查,O(K²)→O(K)):原签名收 other_instances
    列表、逐实例 OR 累加——K=100 时每实例 O(K) 扫描 = O(K²)(实测
    3.81s/step)。改为收**全图前景并集** all_fg,other_fg =
    all_fg[窗口] & ~inst[窗口] ——一次窗口切片,复杂度 O(K)。
    语义逐位等价(审查验证:三实例场景新旧口径 (old==new).all())。

    Args:
        inst: 本实例 bool mask (H,W)
        all_fg: 同图全部实例前景并集(bool (H,W);调用方 = g_bin)
        pad: 外扩像素 = max(2, ceil(0.1*max(h,w)))(调用方算好传入)

    Returns:
        (ys, xs, y0, y1, x0, x1, other_fg) — 前两者是 np.nonzero 索引,
        四个边界是窗口坐标,other_fg 是窗口内其他实例前景 bool mask。
    """
    h, w = inst.shape
    ys, xs = np.nonzero(inst)
    y0 = max(0, ys.min() - pad)
    y1 = min(h, ys.max() + 1 + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(w, xs.max() + 1 + pad)
    other_fg = all_fg[y0:y1, x0:x1] & ~inst[y0:y1, x0:x1]
    return ys, xs, y0, y1, x0, x1, other_fg


def _cc_dice_ce_one(inst_info: tuple, logit_win_b) -> "torch.Tensor":
    """单实例局部 DiceCE(在 GPU 张量上按 ROI 窗口切片)。

    ROI 内逐像素 BCE 平均 + 局部 Soft-Dice:
      BCE 只算 (本实例前景 ∪ ROI 背景非其他实例) 像素——本实例前景为正,
      背景不含邻近其他实例(plan 裁决的局部背景);
      Dice 在同一 ROI 窗口上算(soft,标准 2|p∩g|/( |p|+|g| ) 形式)。

    logit_win_b: (1,H,W) 该样本切片的 **logits**(.float() 后,可导);
    inst_info 由 CPU 侧预计算(见 cc_dice_ce_per_sample)。全部 float32 归约。
    """
    (ys, xs, y0, y1, x0, x1, inst_win, other_fg) = inst_info
    logit_win = logit_win_b[0, y0:y1, x0:x1].float()    # (h,w) 可导,logit 域
    # 🔴 前景必须是**本实例**(inst_win),不能拿整 GT 切窗当前景——
    # 窗口会覆盖邻近实例的像素,整 GT 口径把它们算进本实例前景
    # (npos 虚增、Dice 分子掺假;测试 test_mutation_neighbor_bg_
    # exclusion_matters 钉死)。
    gt_win = torch.from_numpy(inst_win).to(logit_win.device).float()
    # 局部背景 mask:窗口内非本实例、非其他实例的像素
    bg_valid = ~gt_win.bool() & ~torch.from_numpy(other_fg).to(logit_win.device)
    # BCE:前景(正) + 有效背景(负),logit 域直算。
    # 🔴 Gate2 P0 修复(2026-09-18 审查实锤):BCE 必须直接消费 logits
    # (BCEWithLogits 原生 logit 域),不得走 sigmoid→clamp→log 反解——
    # 反解的 clamp 上界 1-1e-8 在 float32 下舍入为 1.0,sigmoid 饱和
    # (logit≥17 即触发;r5 日志实测 seg_prob_max=1.0 存在)时 1-p==0
    # → log(+inf) → NaN/Inf;NaN×valid_mask=NaN 屏蔽无效,GradScaler
    # 静默跳步(r7 空微调同型)。BCEWithLogits 在饱和域行为正确
    # (logit=50 → loss≈50,有限)。
    eps = 1e-8
    n_pos = float((gt_win > 0.5).sum())
    n_neg = float(bg_valid.sum())
    _pos_in = logit_win[gt_win > 0.5]
    _neg_in = logit_win[bg_valid]
    pos_bce = F.binary_cross_entropy_with_logits(
        _pos_in, torch.ones_like(_pos_in))
    neg_bce = F.binary_cross_entropy_with_logits(
        _neg_in, torch.zeros_like(_neg_in))
    bce = ((n_pos * pos_bce if n_pos else torch.zeros((), device=logit_win.device))
           + (n_neg * neg_bce if n_neg else torch.zeros((), device=logit_win.device))
           ) / (n_pos + n_neg + eps)
    # 局部 Soft-Dice(ROI 窗口,含全部窗口像素——Dice 天然对背景像素
    # 通过分母起作用,无需再排除;与全局 per-sample Dice 同构,只是窗口局部)
    pred_sig_win = torch.sigmoid(logit_win)
    inter = (pred_sig_win * gt_win).sum()
    union = pred_sig_win.sum() + gt_win.sum()
    dice = 1.0 - (2.0 * inter + eps) / (union + eps)
    return bce + dice


def cc_dice_ce_per_sample(logits, gt):
    """CC-DiceCE 逐样本版(plan 任务 C,实例等权)。

    每样本:①GT CPU 8 连通标记(detached;预测保持可导);②每实例构建
    局部背景 ROI(bbox 外扩 max(2, ceil(0.1*max(h,w))),邻近实例前景从
    背景惩罚排除);③ROI 内 BCE 平均 + 局部 Soft-Dice,实例等权平均;
    ④返回 (per_sample_loss [B], (n_k_pos, n_soft, n_k0)) 诊断计数。

    边界契约(plan C2 预注册;P2 修复后口径):
    - K=0(GT 全零)/ 软 GT → 该样本 CC 值恒 0.0(与 boundary/lovasz
      归一口径):阴性监督由主项 BCE+Dice 全额承担,CC 外挂不重复计权
      (原"回落全局"版:同一监督计两次+全局/局部混纲,2026-09-18 审查
      P2 实锤后改)。
    - 软 GT(mixup 分数值,非 {0,1}):>0.5 二值化后若仍有实例,按二值
      实例算(boundary 同型)且计入 n_soft;二值化后全零 → 同 K=0 置 0。
      CutMix 的整块替换是二值可算的,归属灰区样本由 mask_weight 闸门在
      分类侧处理,此处不重复发明。
    - 🔴 **autocast 语义**:torch 的 F.binary_cross_entropy 在 autocast
      区内直接 raise("unsafe to autocast")——本函数的 sigmoid+BCE 必须
      在 float32 域。故入口第一步 `logits.float()` **脱离 autocast 域**
      (float() 输入显式标记 dtype,autocast 不再下罩),训练循环无需
      感知;若未来有人在 autocast 区内直接调 F.binary_cross_entropy,
      torch 自己会拦(比静默 bf16 精度损失好)。
    - B=0 → 空 [0] 张量,计数 (0,0,0)。

    诊断计数:
      n_k_pos  = K≥1 且非软 GT 的样本数(CC 项真正生效的人口)
      n_soft   = 软 GT 样本数(二值化后有实例或塌零都计;不进 n_k_pos)
      n_k0     = 原生 GT 全零的样本数
    🔴 P2 修复(2026-09-18 审查):K=0/软GT 样本的 per-sample 值恒 0.0
    (不再回落全局 BCE+Dice)——原实现的"回落全局"与主项 loss_seg 完全
    同构,同一监督被主项+CC 项计两次权,且全局值(~1.8)与窗口局部值
    (~1.4)不同纲,混进 seg_ccdice 单标量后绝对值随 K=0 占比漂移、不可
    跨轮比较。阴性样本的监督 100% 由主项承担,CC 项对无实例样本本来
    就无实例语义。n_cc_k0_fallback/n_cc_soft_gt 计数保留(覆盖度可观测)。
    """
    B = logits.shape[0]
    device = logits.device
    if B == 0:
        return (torch.zeros(0, device=device, dtype=torch.float32),
                (0, 0, 0))

    # 🔴 .float() 先行:autocast 对显式 dtype 的输入不再罩 bf16,
    # sigmoid/BCE 全程 float32(与 boundary/lovasz 的 float32 归约同口径;
    # bf16 域里 BCE 的 -log(1-p) 精度不可接受)。
    # P0 修复(2026-09-18 审查):BCE 两条分支都直接消费 logits_f
    # (sigmoid→clamp→log 反解在饱和 p==1.0 时产 NaN),pred_sig 仅供 Dice。
    logits_f = logits.float()                           # [B,1,H,W] logit 域
    pred_sig = torch.sigmoid(logits_f)                  # [B,1,H,W] Dice 用
    gt_f = gt.float()

    n_k_pos = n_soft = n_k0 = 0
    losses = []
    for b in range(B):
        g_np = gt_f[b, 0].detach().cpu().numpy()
        is_soft = bool(g_np.max() > 0 and
                       ((g_np != 0.0) & (g_np != 1.0)).any())
        g_bin = g_np > 0.5
        insts = _cc_instances_cpu(g_bin) if g_bin.any() else []

        if not insts:
            # K=0 / 软GT塌零 → 计数 + 0.0(P2 修复:监督归主项,见 docstring)
            if is_soft:
                n_soft += 1
            else:
                n_k0 += 1
            losses.append(torch.zeros((), device=device, dtype=torch.float32))
            continue

        # P2 修复(2026-09-18 审查):软 GT 且二值化后仍有实例 → n_soft
        # (docstring 语义:软 GT 不进 n_k_pos,即使 CC 项在该样本上生效;
        # 否则 n_cc_soft_gt 恒 0,软 GT 覆盖度不可观测)。
        if is_soft:
            n_soft += 1
        else:
            n_k_pos += 1
        h, w = g_bin.shape
        pad = max(2, int(math.ceil(0.1 * max(h, w))))
        # CPU 侧预计算每实例的窗口信息(含 other_fg),GPU 侧只做切片算子
        infos = []
        for k, inst in enumerate(insts):
            ys, xs, y0, y1, x0, x1, other_fg = _instance_roi_slices(
                inst, g_bin, pad)          # g_bin = 全实例前景并集(O(K) 口径)
            inst_win = inst[y0:y1, x0:x1]
            infos.append((ys, xs, y0, y1, x0, x1, inst_win, other_fg))
        per_inst = [_cc_dice_ce_one(info, logits_f[b])
                    for info in infos]
        losses.append(torch.stack(per_inst).mean())

    return torch.stack(losses), (n_k_pos, n_soft, n_k0)
