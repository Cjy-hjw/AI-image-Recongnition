"""`train_step.build_official_head_targets` 的口径测试。

三条都是"错了也不会报错、只会让指标系统性偏低"的静默陷阱:
  ① **三头只训主病灶**(评审规则:三新维度仅对主病灶标注/评估)。次病灶带标注
     (约 13%)但不是评分对象,进损失即口径错位;
  ② **mixup 后标签归属**:DR-CutMix 的 lam∈{0,1} 决定标签归 donor 还是 recipient,
     gather 方向反了会让整批学到别人的位置/ETE/T;
  ③ **标准 MixUp 的软 lam** 归属不明确,必须整批屏蔽三头监督。

🔴 **`import sklearn` 必须在 torch / src.training 之前,别"整理"掉**(2026-08-12 实测):
本机 Windows + Demo_D 环境下,先 `import torch` 再走 `src.training.__init__`
(→ `trainer.py` → sklearn)会**段错误崩进程**(EXIT=-1073741819,pytest 报 exit 5),
最小复现只需两行 import。sklearn 先导入即正常。成因是 sklearn / torch 各自捆绑的
OpenMP 运行时在 Windows 上冲突;**Linux 云端不复现**,故只是本地测试的约束,
不是产品缺陷。本文件单独跑时必然触发(全套跑时靠字母序更早的测试恰好先导入了
sklearn,是巧合而非保证)。
"""
# ↓ 顺序敏感,见上方 docstring
import sklearn.metrics  # noqa: F401  (import 副作用:抢先加载 OpenMP 运行时)

import torch

from src.training.loop.train_step import build_official_head_targets

B = 4
TASKS = ("lesion_loc", "extrathy_ext", "clin_t_stage")


def _labels(is_main=None, **override):
    lab = {
        "lesion_loc": torch.tensor([0, 1, 2, 3]),
        "valid_lesion_loc": torch.ones(B),
        "extrathy_ext": torch.tensor([0, 1, 2, 3]),
        "valid_extrathy_ext": torch.ones(B),
        "clin_t_stage": torch.tensor([0, 1, 0, 1]),
        "valid_clin_t_stage": torch.ones(B),
    }
    if is_main is not None:
        lab["is_main"] = torch.tensor(is_main)
    lab.update(override)
    return lab


def _build(labels, mix_info=None):
    return build_official_head_targets(labels, mix_info, B, "cpu")


def test_secondary_lesions_get_zero_head_validity():
    """🔴 `is_main==0` 的次病灶必须拿到 validity=0,不进三头损失。

    评审规则:三新维度仅对主病灶标注/评估。次病灶带标注但不是评分对象,
    放进损失 = 拿非评分对象训评分头,指标系统性错位且不报错。
    """
    tg = _build(_labels(is_main=[1, 0, 1, 0]))
    for task in TASKS:
        assert tg[f"valid_{task}"].tolist() == [1.0, 0.0, 1.0, 0.0], \
            f"{task} 未按 is_main 屏蔽次病灶"


def test_missing_is_main_key_treats_all_as_main():
    """legacy 数据集无 `is_main` 键 → 全当主病灶,与接线前行为一致(零回归)。"""
    tg = _build(_labels(is_main=None))
    assert tg["valid_lesion_loc"].tolist() == [1.0] * B


def test_is_main_combines_with_existing_validity():
    """is_main 与原 validity 是**相乘**关系,不是覆盖 —— 两个条件都要满足。"""
    labels = _labels(is_main=[1, 1, 0, 0],
                     valid_lesion_loc=torch.tensor([1.0, 0.0, 1.0, 0.0]))
    tg = _build(labels)
    assert tg["valid_lesion_loc"].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_cutmix_donor_labels_are_gathered():
    """DR-CutMix lam==0 → 该位继承 donor(index[i])的标签。

    不 gather 或方向反了,会让图像是 donor 的、标签是 recipient 的。
    原 lesion_loc=[0,1,2,3];0 号取 index[0]=3 → 3;2 号取 index[2]=1 → 1。
    """
    mix_info = {
        "index": torch.tensor([3, 2, 1, 0]),
        "lam": torch.tensor([0.0, 1.0, 0.0, 1.0]),   # 0号/2号继承 donor
        "mask_weight": torch.ones(B),
    }
    tg = _build(_labels(is_main=[1, 1, 1, 1]), mix_info)
    assert tg["lesion_loc"].tolist() == [3, 1, 1, 3]
    assert tg["valid_lesion_loc"].tolist() == [1.0] * B


def test_cutmix_donor_carries_its_own_is_main():
    """继承 donor 标签时 is_main 也必须跟着换 —— 否则把次病灶标签当主病灶训。

    1 号是次病灶;0 号继承 index[0]=1 的标签 → 0 号也应被屏蔽。
    """
    mix_info = {
        "index": torch.tensor([1, 0, 3, 2]),
        "lam": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "mask_weight": torch.ones(B),
    }
    tg = _build(_labels(is_main=[1, 0, 1, 1]), mix_info)
    assert tg["valid_lesion_loc"].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_cutmix_donor_carries_its_own_validity():
    """donor 的 validity 同理要跟着换,不能拿 recipient 的 validity 配 donor 的标签。"""
    mix_info = {
        "index": torch.tensor([1, 0, 3, 2]),
        "lam": torch.tensor([0.0, 1.0, 1.0, 1.0]),
        "mask_weight": torch.ones(B),
    }
    labels = _labels(is_main=[1, 1, 1, 1],
                     valid_lesion_loc=torch.tensor([1.0, 0.0, 1.0, 1.0]))
    tg = _build(labels, mix_info)
    # 0 号继承 index[0]=1 的 validity(0.0)
    assert tg["valid_lesion_loc"].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_soft_mixup_masks_all_head_supervision():
    """标准 MixUp 的软 lam(既非 0 也非 1)→ 归属不明,三头整批屏蔽。

    三头是硬标签 masked CE,没有软标签路径;不屏蔽就是拿混合图配单一硬标签。
    """
    mix_info = {
        "index": torch.tensor([1, 0, 3, 2]),
        "lam": torch.full((B,), 0.37),
        "mask_weight": torch.ones(B),
    }
    tg = _build(_labels(is_main=[1, 1, 1, 1]), mix_info)
    for task in TASKS:
        assert tg[f"valid_{task}"].tolist() == [0.0] * B, f"{task} 软混合未屏蔽"


def test_cutmix_mask_weight_zero_masks_head():
    """DR-CutMix 判定归属不明(mask_weight=0)时三头同样屏蔽,复用既有判据。"""
    mix_info = {
        "index": torch.tensor([1, 0, 3, 2]),
        "lam": torch.ones(B),
        "mask_weight": torch.tensor([1.0, 0.0, 1.0, 0.0]),
    }
    tg = _build(_labels(is_main=[1, 1, 1, 1]), mix_info)
    assert tg["valid_clin_t_stage"].tolist() == [1.0, 0.0, 1.0, 0.0]


def test_legacy_labels_without_head_keys_return_empty():
    """无三头键的 legacy labels → 返回空 dict(损失侧整体跳过,零回归)。"""
    assert _build({"malignancy": torch.zeros(B)}) == {}


def test_partial_head_keys_only_emit_present_tasks():
    """只有位置标注时,只出位置键 —— 缺席任务不得被造成默认 0 混进损失。"""
    labels = {"lesion_loc": torch.tensor([0, 1, 2, 3]),
              "valid_lesion_loc": torch.ones(B)}
    tg = _build(labels)
    assert set(tg) == {"lesion_loc", "valid_lesion_loc"}


def test_missing_valid_key_defaults_to_all_valid():
    """标签在但 valid_* 键缺席 → 默认全有效(与既有 label_validity 分支同语义)。"""
    labels = {"lesion_loc": torch.tensor([0, 1, 2, 3])}
    tg = _build(labels)
    assert tg["valid_lesion_loc"].tolist() == [1.0] * B


def test_validity_is_float_and_1d():
    """validity 必须是 [B] 一维 float —— [B,1] 会在损失里隐式广播成 [B,B]。"""
    tg = _build(_labels(is_main=[1, 1, 0, 0]))
    for task in TASKS:
        v = tg[f"valid_{task}"]
        assert v.dim() == 1 and v.shape[0] == B and v.dtype == torch.float32
        assert tg[task].dtype == torch.int64
