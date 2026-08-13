"""位置头 —— 主病灶 7 类位置的解剖因子分解（Phase2 阶段 2.1）。蓝图 = `docs/训练与架构/训练架构方案.md` §2。

## 为什么分解而不是直接 7 分类

7 类联合分类里峡部只占 ~1.8%，直接学难以收敛。拆成「侧别 3 类 × 纵向 3 类」后：
- **结构约束**：`P(k) = P(side) × P(level)` 天然排除非法组合，模型不会输出
  「峡部 + 上极」这种解剖上不存在的搭配。
- **样本复用**：每个左叶样本都为「左」这一侧别类贡献监督，不再被切碎到 3 个子类。

⚠️ **收益要说准**（`docs/训练与架构/训练架构方案.md` 曾写「峡部 F1 +12~15%」，实测数据不支持）：
分解带来的是**结构合法性**与**参数效率**，不是「让峡部从不可学变可学」——峡部在侧别
头里的占比仍只从 1.8% 变到 2.0%。类别不平衡仍须靠采样/加权解决，别指望分解顺手治好。

## 归一化是数学恒等式，不是靠 softmax 兜

    ΣP = P(左)·(P上+P中+P下) + P(右)·(P上+P中+P下) + P(峡)
       = P(左)·1 + P(右)·1 + P(峡)
       = P(左) + P(右) + P(峡) = 1

只要 side/level 各自是合法分布，合成的 7 维**严格和为 1**，无需再归一化。
`tests/test_location_head.py` 对此有 atol=1e-6 的断言。

## 两个必须记住的坑

1. **峡部没有纵向属性**。训练时 level 分支对峡部样本必须屏蔽，否则等于逼模型给
   峡部编造一个「上/中/下」标签。见 `build_level_validity`。
2. **水平翻转必须同时翻左右标签**。数据增强里任何 `HorizontalFlip` 都要联动改
   `lesion_loc`，否则监督信号与图像左右相反、位置头永远学不对。见 `flip_location_index`。
"""

import torch
import torch.nn as nn

from src.models.label_schema import LOCATION_LABELS, NUM_LOCATIONS
from src.models.blocks import CoordConv2d

__all__ = [
    "LocationHead",
    "LocationOutputs",
    "compose_location_probs",
    "build_level_validity",
    "flip_location_index",
    "LOCATION_TO_SIDE",
    "LOCATION_TO_LEVEL",
    "NUM_SIDES",
    "NUM_LEVELS",
    "SIDE_LEFT",
    "SIDE_RIGHT",
    "SIDE_ISTHMUS",
    "ISTHMUS_LOCATION_INDEX",
]

SIDE_LEFT, SIDE_RIGHT, SIDE_ISTHMUS = 0, 1, 2
LEVEL_UPPER, LEVEL_MID, LEVEL_LOWER = 0, 1, 2
NUM_SIDES = 3
NUM_LEVELS = 3

# `LOCATION_LABELS` 里峡部的 0-based 下标。
ISTHMUS_LOCATION_INDEX = 6

# 🔴 顺序锁:下面的因子映射**完全依赖** LOCATION_LABELS 的具体次序。若哪天官方
# 类别顺序变了而这里没跟着改,合成概率会**静默错位**(把左叶预测填进右叶的槽),
# JSON 依然合法、和依然为 1、测试依然过 —— 是最难查的一类错。故在导入期就断言死。
# 这份"重复"是有意的:它把隐式假设变成显式契约。
_EXPECTED_LABEL_ORDER = (
    "左上叶", "左中叶", "左下叶",
    "右上叶", "右中叶", "右下叶",
    "峡部",
)
assert tuple(LOCATION_LABELS) == _EXPECTED_LABEL_ORDER, (
    f"LOCATION_LABELS 顺序已变更: {tuple(LOCATION_LABELS)} != {_EXPECTED_LABEL_ORDER}。"
    " 位置头的 LOCATION_TO_SIDE/LOCATION_TO_LEVEL 因子映射与 compose_location_probs "
    "的拼接顺序均按旧序写死,必须同步修正后再解除本断言。"
)
assert LOCATION_LABELS[ISTHMUS_LOCATION_INDEX] == "峡部"
assert NUM_LOCATIONS == 7

# 7 类 → (侧别, 纵向)。峡部无纵向,记 -1。
LOCATION_TO_SIDE = (
    SIDE_LEFT, SIDE_LEFT, SIDE_LEFT,
    SIDE_RIGHT, SIDE_RIGHT, SIDE_RIGHT,
    SIDE_ISTHMUS,
)
LOCATION_TO_LEVEL = (
    LEVEL_UPPER, LEVEL_MID, LEVEL_LOWER,
    LEVEL_UPPER, LEVEL_MID, LEVEL_LOWER,
    -1,
)

# 水平翻转的类别置换:左↔右同层,峡部不变。
_FLIP_PERMUTATION = (3, 4, 5, 0, 1, 2, 6)


def flip_location_index(idx: torch.Tensor) -> torch.Tensor:
    """水平翻转后的位置标签(左↔右同层互换,峡部不变)。

    数据增强里每加一次 `HorizontalFlip`,标签就必须过一次本函数 —— 漏了会让监督
    信号与图像左右相反,位置头学到的左右恰好是反的,而 loss 看上去仍在下降。
    """
    perm = torch.tensor(_FLIP_PERMUTATION, device=idx.device, dtype=idx.dtype)
    return perm[idx]


def build_level_validity(
    location_idx: torch.Tensor, location_validity: torch.Tensor
) -> torch.Tensor:
    """纵向分支的逐样本有效性 = 位置标签有效 **且** 不是峡部。

    Args:
        location_idx: [B] 0-based 位置标签(无效样本处的值任意,由 validity 屏蔽)。
        location_validity: [B] 1=该样本有位置标注。
    Returns:
        [B] float,可直接进 masked loss 的 `(loss*v).sum()/(v.sum()+eps)`。

    峡部样本的 level 标签**不存在**(不是"未知"),给它任何一个 0/1/2 都是编造监督。
    """
    not_isthmus = (location_idx != ISTHMUS_LOCATION_INDEX).to(location_validity.dtype)
    return location_validity * not_isthmus


def compose_location_probs(
    side_probs: torch.Tensor, level_probs: torch.Tensor
) -> torch.Tensor:
    """(侧别分布 [B,3], 纵向分布 [B,3]) → 7 类位置分布 [B,7]，严格和为 1。

    拼接顺序必须与 `LOCATION_LABELS` 一致：`(左,右) × (上,中,下)` 行优先展开恰好是
    左上/左中/左下/右上/右中/右下，再接峡部。
    """
    assert side_probs.shape[-1] == NUM_SIDES, side_probs.shape
    assert level_probs.shape[-1] == NUM_LEVELS, level_probs.shape
    # [B,2,1] * [B,1,3] -> [B,2,3] -> reshape [B,6]（行优先：先走完左叶三层，再右叶）
    lr = side_probs[..., :2].unsqueeze(-1)
    lv = level_probs.unsqueeze(-2)
    six = (lr * lv).reshape(*side_probs.shape[:-1], 6)
    return torch.cat([six, side_probs[..., SIDE_ISTHMUS:SIDE_ISTHMUS + 1]], dim=-1)


class LocationOutputs(dict):
    """位置头输出。用 dict 子类而非 dataclass，与 `ModelOutputs` 的字典风格一致。

    键：
      - `side_logits` [B,3] / `level_logits` [B,3] —— 训练走这两个分支各自的 masked CE。
        **不要**对合成的 7 维概率取 log 再算 NLL：乘积概率下溢到 0 时 log 会产生 -inf。
      - `location_probs` [B,7] —— 推理/评估用，严格和为 1，直接喂 `LocationProbs` 字段。
    """


class LocationHead(nn.Module):
    """Stage-4 空间特征 → 侧别/纵向双分支 → 7 类位置分布。

    Args:
        in_channels: 必须是 `MultiTaskThyroidModel.CLS_POOLED_DIM`(**352**)，
            不是 `CLS_STAGE4_CHANNELS`(208) —— 后者是 adaptor 的输入通道，
            `blocks[6]` 之后已升到 352。写错 208 要到真训练才崩。
        hidden_channels: CoordConv 输出通道。
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.spatial = nn.Sequential(
            CoordConv2d(in_channels, hidden_channels, kernel_size=3),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.side_fc = nn.Linear(hidden_channels, NUM_SIDES)
        self.level_fc = nn.Linear(hidden_channels, NUM_LEVELS)

    def forward(self, stage4_spatial: torch.Tensor) -> LocationOutputs:
        assert stage4_spatial.dim() == 4, (
            f"位置头需要空间特征图 [B,C,H,W]，got {tuple(stage4_spatial.shape)}"
            " —— CoordConv 的坐标通道依赖 H/W，池化后的 [B,C] 用不了"
        )
        assert stage4_spatial.shape[1] == self.in_channels, (
            f"通道失配: 头按 in_channels={self.in_channels} 构建，"
            f"实际输入 {stage4_spatial.shape[1]}"
        )
        feat = self.pool(self.spatial(stage4_spatial)).flatten(1)
        side_logits = self.side_fc(feat)
        level_logits = self.level_fc(feat)
        # softmax 在 fp32 下算：fp16 autocast 时指数运算易下溢，而合成概率是乘积，
        # 下溢会被平方放大（两个 1e-4 相乘即 1e-8）。
        side_probs = torch.softmax(side_logits.float(), dim=-1)
        level_probs = torch.softmax(level_logits.float(), dim=-1)
        return LocationOutputs(
            side_logits=side_logits,
            level_logits=level_logits,
            location_probs=compose_location_probs(side_probs, level_probs),
        )
