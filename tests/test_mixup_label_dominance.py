"""四头监督闸门的主导权阈值 —— 钉住 2026-08-14 修的真 bug。

## 背景（勿删，这是本文件存在的唯一理由）

首次真数据训练出现 loss ~5 / ~0.8 双峰，低值行里**四个官方头（位置/ETE/临床T/
TI-RADS）的分项精确为 0.0**，只剩 seg/supcon/spin_l1 在贡献。

根因在 `build_official_head_targets` 的闸门用了**精确相等**：
`donor = lam == 0.0`、`hard = donor | (lam == 1.0)`。
- CutMix 分支逐样本把 `lams` 判成 0.0/1.0（DR-CutMix 按 donor 占比 0.65/0.35），
  精确相等对它成立；
- **但 MixUp 分支的 `lams` 是全批同一个 `lam ~ Beta(0.2,0.2)`，严格落在 (0,1)**
  → `hard` 全 False → 四头 valid 整批乘 0。
- `prob=0.3` 且 CutMix/MixUp 各半 → 约 **15% 的 batch 四头零监督**，不报错、只是没学。

修复：闸门改用与 DR-CutMix 同一套阈值（`lam<=0.35` 判 donor 主导、`lam>=0.65` 判
recipient 主导、中间屏蔽）。

🔴 **刻意只改 `train_step.py` 的闸门，不动 `mixup.py`**：`lam` 还被 CBM 概念损失
（`losses/components/concepts.py` 软插值）和 SLR（`loop/slr.py`）当**软加权系数**消费，
在 mixup 侧把它硬化成 0/1 会连带改掉旧任务的数学。`mask_weight` 同理不动
（它还门控概念损失）。本文件的 `test_cutmix_behavior_unchanged` 就是钉这条边界。
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.dataset.augmentation.mixup as M  # noqa: E402
from src.training.loop.train_step import build_official_head_targets  # noqa: E402

B = 4
TASKS = ("lesion_loc", "extrathy_ext", "clin_t_stage", "tirads")
DEV = torch.device("cpu")


def _labels():
    """样本0 的四头标签全 0，样本1 全 3 —— 用于区分标签继承自谁。"""
    lb = {}
    for t in TASKS:
        v = torch.zeros(B, dtype=torch.long)
        v[1] = 3
        lb[t] = v
        lb[f"valid_{t}"] = torch.ones(B)
    lb["is_main"] = torch.ones(B)
    return lb


def _mix(lam_value):
    """构造 index=[1,0,3,2] 的 mix_info：样本0 的 donor 是样本1。"""
    return {
        "index": torch.tensor([1, 0, 3, 2]),
        "lam": torch.full((B,), float(lam_value)),
        "mask_weight": torch.ones(B),
    }


@pytest.mark.parametrize("lam_value,want_label", [
    (0.00, 3),   # CutMix donor 档（原就支持）
    (0.05, 3),   # donor 主导 —— 修复前 hard=False，四头零监督
    (0.35, 3),   # 边界：恰好 0.35 仍判 donor 主导
    (0.65, 0),   # 边界：恰好 0.65 仍判 recipient 主导
    (0.95, 0),   # recipient 主导 —— 修复前同样零监督
    (1.00, 0),   # CutMix recipient 档（原就支持）
])
def test_dominant_lam_keeps_supervision_with_correct_label(lam_value, want_label):
    """主导方明确时必须保留监督，且标签继承自**图像里占主导的那一方**。

    🔴 方向判反比零监督更糟：零监督只是没学，错标签是学反。
       `mixed = lam*recipient + (1-lam)*donor` → lam 小 = donor 占主导。
    """
    tg = build_official_head_targets(_labels(), _mix(lam_value), B, DEV)
    for task in TASKS:
        assert tg[f"valid_{task}"][0].item() == 1.0, (
            f"lam={lam_value} 主导方明确，{task} 不该被屏蔽（这正是修复前的 bug）")
        assert tg[task][0].item() == want_label, (
            f"lam={lam_value} 时 {task} 标签应继承 {want_label}，"
            f"实得 {tg[task][0].item()} —— 方向judged反了")


@pytest.mark.parametrize("lam_value", [0.36, 0.50, 0.64])
def test_ambiguous_lam_still_masked(lam_value):
    """归属不明确(0.35 < lam < 0.65)仍屏蔽 —— 这条**不是** bug，要保留。

    四头是硬标签 masked CE，真·五五开的混合图配单一硬标签就是喂错标签。
    """
    tg = build_official_head_targets(_labels(), _mix(lam_value), B, DEV)
    for task in TASKS:
        assert tg[f"valid_{task}"].tolist() == [0.0] * B, (
            f"lam={lam_value} 归属不明，{task} 必须屏蔽")


def test_cutmix_behavior_unchanged():
    """CutMix 的 lams 已是 0.0/1.0，新阈值对它**完全等价**，行为一律不变。

    这是修复的边界条件：只放宽判据，不改变任何既有路径。
    """
    for lam_value, want in [(0.0, 3), (1.0, 0)]:
        tg = build_official_head_targets(_labels(), _mix(lam_value), B, DEV)
        for task in TASKS:
            assert tg[f"valid_{task}"][0].item() == 1.0
            assert tg[task][0].item() == want


def test_mixup_module_untouched_still_emits_soft_lam():
    """`mixup.py` 必须保持原样产出软 lam —— 概念损失/SLR 依赖它做软插值。

    🔴 反向保护:若有人"顺手"把 mixup 侧也硬化成 0/1,CBM 概念损失
    (`concepts.py` 的 `lam*loss_a + (1-lam)*loss_b`)与 SLR(`slr.py`)的数学
    会被静默改掉。本测试确保修复范围**只在 train_step 的闸门**。
    """
    seq = iter([0.0, 0.1])          # 第1次判触发，第2次 <0.5 → 走 MixUp 分支
    orig_rand, orig_beta = M.random.random, M.np.random.beta
    try:
        M.random.random = lambda: next(seq, 0.1)
        M.np.random.beta = lambda a, b: 0.42     # 中间档软值
        images = torch.rand(B, 1, 8, 8)
        masks = torch.zeros(B, 1, 8, 8)
        masks[:, :, 2:6, 2:6] = 1.0
        _i, _m, _l, mix_info = M.apply_mixup_or_cutmix(
            images, masks, {"malignancy": torch.zeros(B, dtype=torch.long)},
            alpha=0.2, prob=1.0)
    finally:
        M.random.random, M.np.random.beta = orig_rand, orig_beta

    assert mix_info is not None
    assert mix_info["lam"].tolist() == pytest.approx([0.42] * B, abs=1e-6), (
        "mixup.py 的 MixUp 分支必须保留软 lam（概念损失/SLR 靠它软插值）")
    assert mix_info["mask_weight"].tolist() == [1.0] * B, (
        "mixup.py 的 mask_weight 不该被改动（它还门控 CBM 概念损失）")


def test_supervision_retention_rate_under_beta02():
    """量化修复收益：Beta(0.2,0.2) 下四头监督保留率应远高于 0。

    修复前 MixUp 分支保留率恒为 **0%**（软 lam 全被 `hard` 判 False 屏蔽）。
    Beta(0.2,0.2) 是 U 形分布（alpha<1，抽样集中两端），故修复后应保留绝大多数。
    """
    draws = np.random.default_rng(0).beta(0.2, 0.2, size=4000)
    kept = ((draws <= 0.35) | (draws >= 0.65)).mean()
    assert kept > 0.75, f"保留率 {kept:.1%} 与 U 形分布预期不符"
