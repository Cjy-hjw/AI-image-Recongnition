"""ROI(per-lesion)位置损失 —— 两阶段 ROI 范式损失组(方案一 P3,2026-09-25)。

与 `components/location.py`(整图版)的**同式数学**:侧别/纵向双分支 masked CE。
唯一差别是输入形态——整图版吃 [B,3] side_logits/[B,3] level_logits(模型面
`LocationHead` 的输出);ROI 版吃 `location_probs` [K,7](ROI 模型 forward_roi
契约输出的**7 类联合分布**),侧别/纵向边缘分布由 7 类联合概率**边缘化**导出:

    P(side=s) = Σ_{k: side(k)=s} location_probs[:, k]
    P(level=l) = Σ_{k: level(k)=l} location_probs[:, k]

数学上与双分支各出 [K,3] softmax 后取边缘**完全等价**(因子分解模型
`compose_location_probs` 的逆),但对联合分布直接算 NLL 与双分支 CE 不再同式
(`components/location.py` docstring 里"两个 CE 而非 7 维 NLL"的数值稳定性
论证,针对的是**乘积合成概率**的 log 下溢;此处输入是模型面已 softmax 好的
联合概率,不存在乘积下溢问题,但 softmax 输出可能含 0 → log(0)=-inf)。
故实现走 `log(clamp(p, min=eps))` 的对数域,梯度经联合概率回流 ROI 头。

## 与整图版的口径对齐(勿漂移成两把尺子)

- **标签语义不变**:0-based 7 类位置标签,峡部下标 6;侧别/纵向映射表
  `LOCATION_TO_SIDE`/`LOCATION_TO_LEVEL` 单源复用 `heads/location.py`
  (顺序锁断言在那边导入期钉死,此处不重复抄)。
- **峡部 level 屏蔽**:`build_level_validity` 同式(位置有效且非峡部)。
  峡部 level 标签是 -1,本实现不喂 CE(边缘分布只有 3 档,无 -1 可言),
  屏蔽语义 = 峡部样本对 level 分支贡献 0 —— 与整图版"clamp 到 0 再乘 0"等价。
- **is_main 门**:官方口径三新维度仅主病灶评估 → 损失监督只吃 is_main==1 的
  lesion(判据与 `train_step.py:90-95` 整图四头同口径:`v * main_w`,
  is_main 缺席时全当主病灶)。**与 ETE/T 头的恶性限定不同、与位置头整图版
  相同——位置指标不带恶性限定**。
- **valid 过滤**:per-lesion `valid_lesion_loc` 标志,键名与
  `roi_dataset._ROI_SCALAR_KEYS` 产出的 collate 键一致(损失消费的是
  per-lesion list,不是 batch tensor,键名后缀 `_lesion_loc` 同源)。
- **归一范式**:`(loss * v).sum() / (v.sum() + eps)` 分母只数有效样本
  (与全项目 masked loss 同范式,详见 `components/concepts.py`)。
"""

import torch

from src.models.heads.location import (
    ISTHMUS_LOCATION_INDEX,
    LOCATION_TO_LEVEL,
    LOCATION_TO_SIDE,
)

__all__ = ["compute_location_roi_loss"]

_EPS = 1e-8
# k_ref 分母平滑(债6 同批,2026-09-25,DR 机制 1):masked 归一分母
# (v.sum()+eps) → clamp(v.sum()+eps, min=4.0)。三段语义:
#   n_valid ≥ 4:clamp 不激活,分母 = v.sum()+eps 与旧公式**逐位一致**(零回归);
#   n_valid ∈ (0,4):退化小批分母抬到 4(实际 ROI 批 16 图合计 K≈16-48,
#     1-3 条是"末批只剩 1 病灶"兜底非主修正,防该头单样本独大);
#   v 全 0:分子每项乘 v=0 恒 0 → 0/4 = 精确 0(eps 的 0 值路径语义保留)。
_K_REF = 4.0


def _edge_log_probs(location_probs: torch.Tensor) -> tuple:
    """[K,7] 联合概率 → (侧别 log 边缘 [K,3], 纵向 log 边缘 [K,3])。

    边缘化在**概率域**做(加法,数值稳定),log 在最后取并 clamp 防 0 概率
    (softmax 输出可为精确 0,如极端 logit gap 下)。
    🔴 先加概率再取 log,不是累加 log(Σlog p = log Πp ≠ log Σp——
    边缘分布是求和不是求积,累 log 会把 0.75 的边缘算成 0.75³ 量级)。

    🔴 level 边缘**必须归一**(Gate2 B 路 P1-3 实测定案,2026-09-25):因子
    分解联合分布 P(k)=P(side)·P(level) 下,6 个非峡部槽的 level 边缘行和 =
    1−P(峡部) 而非 1(实测行和 0.51~0.82)。不归一则:① level CE 系统性
    偏高 −log(1−P_峡部)(训练初期 +0.405/样本,与 level CE 本身同量级);
    ② level 梯度经联合分布耦合到 side 头的峡部通道(压 P_峡部 降损失,
    整图双 softmax 版无此耦合);③ P_峡部→1 时梯度死。归一 = 条件分布
    P(level|非峡部),与整图版独立 level softmax 的口径对齐。side 边缘天然
    行和 1 无需归一。
    """
    probs = location_probs.float()
    n_side = len(set(LOCATION_TO_SIDE))
    n_level = max(LOCATION_TO_LEVEL) + 1          # -1 哨兵不占档
    side_p = torch.zeros(probs.shape[0], n_side,
                         device=probs.device, dtype=probs.dtype)
    level_p = torch.zeros(probs.shape[0], n_level,
                          device=probs.device, dtype=probs.dtype)
    for k in range(probs.shape[1]):
        s = LOCATION_TO_SIDE[k]
        side_p[:, s] = side_p[:, s].clone() + probs[:, k]
        l = LOCATION_TO_LEVEL[k]
        if l >= 0:                                # 峡部无纵向,不入 level 边缘
            level_p[:, l] = level_p[:, l].clone() + probs[:, k]
    # level 归一:除以非峡部总概率(=1−P_峡部)。clamp 防 P_峡部→1 除零
    # (此时非峡部槽无监督意义,梯度由 clamp 截断——与整图版极端饱和同语义)。
    level_total = level_p.sum(dim=1, keepdim=True)
    level_p = level_p / level_total.clamp(min=_EPS)
    return side_p.clamp(min=_EPS).log(), level_p.clamp(min=_EPS).log()


def compute_location_roi_loss(
    location_probs: torch.Tensor,
    lesion_loc: torch.Tensor,
    location_validity: torch.Tensor,
    is_main: torch.Tensor = None,
):
    """ROI 位置双分支 masked CE(边缘分布版)。

    Args:
        location_probs: [K,7] ROI 模型 forward_roi 输出的联合概率(softmax 后,
            契约键 `location_probs`)。梯度经联合概率回流。
        lesion_loc: [K] long,0-based 7 类位置标签。无效样本处的值任意。
        location_validity: [K] float,1 = 该病灶有位置标注。
        is_main: [K] float/int,1 = 主病灶。None → 全当主病灶(与整图链路
            `train_step` 的 `is_main` 缺席处理同口径)。
    Returns:
        (total, detail):total 标量;detail 含 `roi_location`/`roi_location_side`/
        `roi_location_level` 三个标量(side/level 便于分别观察哪一支在学)。
        整批无有效样本 → 精确 0、梯度 0、无 NaN(masked 范式分母 eps)。
    """
    assert location_validity.dim() == 1, (
        f"location_validity 必须是 [K] 一维,got {tuple(location_validity.shape)}"
        " —— [K,1] 会隐式广播成 [K,K] 静默放大 loss(QA_VP §2.2)"
    )
    assert lesion_loc.shape == location_validity.shape, (
        f"标签与 validity 形状不一致: {tuple(lesion_loc.shape)} vs "
        f"{tuple(location_validity.shape)}"
    )
    assert location_probs.dim() == 2 and location_probs.shape[1] == 7, (
        f"location_probs 应为 [K,7],got {tuple(location_probs.shape)}"
        " —— ROI 头契约输出 7 类联合分布(位置 7 类实测定稿)"
    )
    assert int(lesion_loc.min()) >= 0 and int(lesion_loc.max()) < 7, (
        f"位置标签越界: min={int(lesion_loc.min())}, max={int(lesion_loc.max())},"
        " 合法区间 [0,6](无标注样本应填占位 0 + validity=0)"
    )

    K = location_probs.shape[0]
    device = location_probs.device

    # is_main 门:官方口径三新维度仅主病灶 → 监督只吃主病灶(与整图四头
    # train_step.py:90-95 同判据:v * main_w;is_main 缺席 = 全主病灶)。
    if is_main is None:
        main_w = torch.ones(K, dtype=torch.float32, device=device)
    else:
        assert is_main.shape == location_validity.shape, (
            f"is_main 与 validity 形状不一致: {tuple(is_main.shape)} vs "
            f"{tuple(location_validity.shape)}"
        )
        main_w = is_main.to(device=device, dtype=torch.float32)

    v_side = (location_validity.to(device) * main_w).float()
    side_log, level_log = _edge_log_probs(location_probs)

    # 侧别目标:LOCATION_TO_SIDE[lesion_loc]
    side_target = torch.tensor(
        LOCATION_TO_SIDE, device=device, dtype=torch.long)[lesion_loc]
    loss_side_ps = -side_log.gather(1, side_target.unsqueeze(1)).squeeze(1)

    # 纵向:峡部无 level 标签 → build_level_validity 同式屏蔽
    # (位置有效 × 非峡部 × is_main)。
    not_isthmus = (lesion_loc != ISTHMUS_LOCATION_INDEX).to(v_side.dtype)
    v_level = v_side * not_isthmus
    level_target = torch.tensor(
        LOCATION_TO_LEVEL, device=device, dtype=torch.long)[lesion_loc]
    # LOCATION_TO_LEVEL[峡部] = -1:level 边缘分布只有 3 档,直接 gather 会
    # IndexError → clamp 到 0 保证不崩,随后 v_level=0 屏蔽(两道保险,与整图
    # 版"clamp 再屏蔽"同构;只 clamp 会编造监督,只屏蔽会先崩在 gather 里)。
    level_target_safe = level_target.clamp(min=0)
    loss_level_ps = -level_log.gather(1, level_target_safe.unsqueeze(1)).squeeze(1)

    loss_side = (loss_side_ps * v_side).sum() / torch.clamp(
        v_side.sum() + _EPS, min=_K_REF)
    loss_level = (loss_level_ps * v_level).sum() / torch.clamp(
        v_level.sum() + _EPS, min=_K_REF)

    total = loss_side + loss_level
    return total, {
        "roi_location": total.detach(),
        "roi_location_side": loss_side.detach(),
        "roi_location_level": loss_level.detach(),
    }
