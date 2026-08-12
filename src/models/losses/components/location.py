"""位置任务损失 —— 侧别/纵向双分支各自的 masked CE（Phase2 阶段 2.1）。

与项目既有 masked loss 同范式：`(loss * v).sum() / (v.sum() + eps)`，
分母只数有效样本，不被屏蔽样本稀释（见 `components/concepts.py`）。

**为什么是两个 CE 而不是对 7 维概率取 NLL**：合成概率是 `P(side)·P(level)` 的乘积，
两个 1e-4 相乘就是 1e-8，`log` 之后接近 -18 甚至下溢 -inf；分支各自算 CE 则走
`log_softmax` 的数值稳定实现，不经过乘积。
"""

import torch
import torch.nn.functional as F

from src.models.heads.location import (
    ISTHMUS_LOCATION_INDEX,
    LOCATION_TO_LEVEL,
    LOCATION_TO_SIDE,
    build_level_validity,
)

_EPS = 1e-8


def _lut(table, device, dtype=torch.long):
    """把 python 元组映射表搬到与标签同一 device（每次调用重建，避免跨设备缓存踩坑）。"""
    return torch.tensor(table, device=device, dtype=dtype)


def compute_location_loss(
    side_logits: torch.Tensor,
    level_logits: torch.Tensor,
    location_idx: torch.Tensor,
    location_validity: torch.Tensor,
):
    """位置双分支 masked CE。

    Args:
        side_logits: [B,3] / level_logits: [B,3] —— 来自 `LocationHead`。
        location_idx: [B] long，0-based 7 类位置标签。无效样本处的值任意。
        location_validity: [B] float，1=有位置标注。
    Returns:
        (total, detail_dict)：total 是标量；detail 含 `location_side`/`location_level`
        两个标量，便于分别观察哪一支在学。

    三个边界都已处理：
      - **峡部的 level 标签是 -1**（不存在纵向），直接喂 `cross_entropy` 会
        `IndexError: Target -1 is out of bounds`。故先 clamp 到 0 保证不崩，
        再由 `build_level_validity` 把这些样本的 loss 乘 0 屏蔽掉 —— 两道保险，
        缺任一条都会出问题（只 clamp 会编造监督，只屏蔽会先崩在 CE 里）。
      - **整批无有效样本**：分子逐项已乘 v，恒为 0；分母 `0 + eps` → loss 精确 0、
        梯度 0，不产生 NaN。
      - **整批全是峡部**：level 分支 validity 全 0 → 该支 loss 0、无梯度，正确。
    """
    device = location_idx.device
    # 🔴 validity 必须是 1-D。传 [B,1] 时 `loss_ps[B] * v[B,1]` 会**隐式广播成 [B,B]**
    # 且不报错,loss 被静默放大(实测同一批 3 样本:[B] 得 1.98、[B,1] 得 4.575)。
    # 这正是 QA_VP §1.5/§2.2 点名的坑 —— 数值错但训练照跑,最难查。
    assert location_validity.dim() == 1, (
        f"location_validity 必须是 [B] 一维,got {tuple(location_validity.shape)}"
        " —— [B,1] 会隐式广播成 [B,B] 静默放大 loss(QA_VP §2.2)"
    )
    assert location_idx.shape == location_validity.shape, (
        f"标签与 validity 形状不一致: {tuple(location_idx.shape)} vs "
        f"{tuple(location_validity.shape)}"
    )
    # 标签越界必须 fail-loud。注意**正负不对称**:`idx=7` 会让 `_lut(...)[7]` 直接
    # IndexError,而 `idx=-1` 走 python 负索引语义、静默取到末元素(恰好是峡部的 -1),
    # 再被 clamp+validity 吞掉 —— 表现为"这个样本没有监督",而不是报错。若上游漏设
    # validity,这类样本会静默丢监督且无任何迹象。故显式拦下界。
    assert int(location_idx.min()) >= 0 and int(location_idx.max()) < len(
        LOCATION_TO_SIDE
    ), (
        f"位置标签越界: min={int(location_idx.min())}, max={int(location_idx.max())}，"
        f"合法区间 [0,{len(LOCATION_TO_SIDE) - 1}]（无标注样本应填占位 0 + validity=0）"
    )
    v_side = location_validity.to(side_logits.dtype)
    side_target = _lut(LOCATION_TO_SIDE, device)[location_idx]
    level_target = _lut(LOCATION_TO_LEVEL, device)[location_idx]

    v_level = build_level_validity(location_idx, location_validity).to(
        level_logits.dtype
    )
    # 峡部处 level_target = -1 → clamp 到 0（该位置随后被 v_level=0 屏蔽）
    level_target_safe = level_target.clamp(min=0)

    loss_side_ps = F.cross_entropy(side_logits, side_target, reduction="none")
    loss_level_ps = F.cross_entropy(level_logits, level_target_safe, reduction="none")

    loss_side = (loss_side_ps * v_side).sum() / (v_side.sum() + _EPS)
    loss_level = (loss_level_ps * v_level).sum() / (v_level.sum() + _EPS)

    total = loss_side + loss_level
    return total, {"location_side": loss_side, "location_level": loss_level}


def location_target_factors(location_idx: torch.Tensor):
    """[B] 位置标签 → ([B] 侧别, [B] 纵向)。峡部的纵向为 -1。供测试与调试用。"""
    device = location_idx.device
    return (
        _lut(LOCATION_TO_SIDE, device)[location_idx],
        _lut(LOCATION_TO_LEVEL, device)[location_idx],
    )


__all__ = [
    "compute_location_loss",
    "location_target_factors",
    "ISTHMUS_LOCATION_INDEX",
]
