"""官方三头 A/B/C 调度(`schedule_official.py`)+ JSONL 日志(`jsonl_logger.py`)的
关键数值/契约锁定测试。

**为什么需要**:官方数据三头训练是"只能提交一次、踩上去无法补救"的关键路径,而它
刻意**不复用** `stages.py`(后者 Stage1→2 边界 lr 100 倍跳变实测崩分割)。本文件把
新调度的三条硬约束钉进 CI:
  ① 相邻 epoch lr 比值 ≤ 5(把 100× 事故钉成不可能);
  ② 各 stage 的冻结/解冻集合(A/C 只训三头 + vclub;B 额外解冻 cls Top-2 block);
  ③ 所有可训参数都进 optimizer(除 vclub 独立优化器)。

纯 CPU / 无网络 / pretrained=False,可直接进 CI。sklearn 须在 torch 前导入以避开
Windows OpenMP DLL 冲突(段错误 EXIT=-1073741819)—— 本文件不用 sklearn,但保持
与其它测试同序无害。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from src.models.loss import MultiTaskLoss
from src.models.multi_task_model import MultiTaskThyroidModel
from src.training.loop.bn_freeze import set_frozen_modules_eval
from src.training.schedule_official import (
    MAX_LR_JUMP_RATIO,
    STAGE_A_END,
    STAGE_B_END,
    STAGE_C_END,
    STAGE_B_UNFREEZE,
    configure_official_stage,
    get_official_stage,
    official_head_lr,
)

LR_BASE = 0.0024
WEIGHT_DECAY = 5e-4
_HEAD_PFX = MultiTaskThyroidModel.OFFICIAL_HEAD_PREFIXES


@pytest.fixture(scope="module")
def model_and_criterion():
    model = MultiTaskThyroidModel(pretrained=False)
    criterion = MultiTaskLoss()
    return model, criterion


# ═════════════════════════════════════════════════════════════════════════════
# 阶段划分
# ═════════════════════════════════════════════════════════════════════════════


def test_stage_boundaries():
    assert get_official_stage(1) == "A"
    assert get_official_stage(STAGE_A_END) == "A"          # 10
    assert get_official_stage(STAGE_A_END + 1) == "B"      # 11
    assert get_official_stage(STAGE_B_END) == "B"          # 30
    assert get_official_stage(STAGE_B_END + 1) == "C"      # 31
    assert get_official_stage(STAGE_C_END) == "C"          # 35


# ═════════════════════════════════════════════════════════════════════════════
# lr 曲线 —— 本文件核心:相邻比值 ≤ 5
# ═════════════════════════════════════════════════════════════════════════════


def test_no_adjacent_lr_jump_exceeds_cap():
    """🔴 任意相邻 epoch 的 lr 比值 ≤ MAX_LR_JUMP_RATIO(=5)。

    这是把 stages.py 的 100× 崩塌根因钉成不可能的核心断言。全程 1..35 逐点核。
    """
    lrs = [official_head_lr(e, LR_BASE) for e in range(1, STAGE_C_END + 1)]
    assert all(x > 0 for x in lrs), "lr 必须恒正"
    for i in range(1, len(lrs)):
        ratio = max(lrs[i], lrs[i - 1]) / min(lrs[i], lrs[i - 1])
        assert ratio <= MAX_LR_JUMP_RATIO, (
            f"epoch {i}→{i+1} lr 跳变 {ratio:.2f}× 超上限 {MAX_LR_JUMP_RATIO}×"
            f"(lr {lrs[i-1]:.3e}→{lrs[i]:.3e})——重演 stages.py 100× 事故的风险"
        )


def test_lr_key_values():
    """A warmup 端点 / B 余弦端点 / C 固定值。"""
    # A: e1 = lr_base/10、e10 = lr_base
    assert official_head_lr(1, LR_BASE) == pytest.approx(LR_BASE * 0.1)
    assert official_head_lr(STAGE_A_END, LR_BASE) == pytest.approx(LR_BASE)
    # B: e11 起点回到 lr_base(与 e10 衔接比值 1.0)、e30 退到 lr_base×0.1
    assert official_head_lr(STAGE_A_END + 1, LR_BASE) == pytest.approx(LR_BASE)
    assert official_head_lr(STAGE_B_END, LR_BASE) == pytest.approx(LR_BASE * 0.1)
    # C: 固定 lr_base×0.05
    for e in range(STAGE_B_END + 1, STAGE_C_END + 1):
        assert official_head_lr(e, LR_BASE) == pytest.approx(LR_BASE * 0.05)


def test_stage_b_cosine_monotonic_decreasing():
    """Stage B(11-30)余弦退火严格单调下降。"""
    lrs = [official_head_lr(e, LR_BASE) for e in range(STAGE_A_END + 1, STAGE_B_END + 1)]
    assert all(a > b for a, b in zip(lrs, lrs[1:])), f"Stage B 应严格下降,got {lrs}"


def test_stage_a_warmup_ramps_up():
    """Stage A(1-10)线性 warmup 严格上升,末端达 lr_base。"""
    lrs = [official_head_lr(e, LR_BASE) for e in range(1, STAGE_A_END + 1)]
    assert all(a < b for a, b in zip(lrs, lrs[1:])), "warmup 段应严格上升"
    assert lrs[-1] == pytest.approx(LR_BASE)


# ═════════════════════════════════════════════════════════════════════════════
# 冻结集合 / 优化器覆盖
# ═════════════════════════════════════════════════════════════════════════════


def _trainable(model):
    return {n for n, p in model.named_parameters() if p.requires_grad}


def test_stage_a_only_heads_and_vclub_trainable(model_and_criterion):
    model, criterion = model_and_criterion
    configure_official_stage(model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY)
    trainable = _trainable(model)
    assert trainable, "Stage A 至少三头可训"
    for n in trainable:
        assert n.startswith(_HEAD_PFX) or "vclub" in n, (
            f"Stage A 只应训三头 + vclub,却有 {n} 可训"
        )
    # 三头确实全可训
    assert any(n.startswith("location_head") for n in trainable)
    assert any(n.startswith("extension_head") for n in trainable)
    assert any(n.startswith("clinical_t_head") for n in trainable)
    # backbone/分割/CBM 全冻
    assert not any(n.startswith("seg_backbone") for n in trainable), "Stage A 应冻分割主干"
    assert not any(n.startswith("cls_backbone") for n in trainable), "Stage A 应冻分类主干"


def test_stage_b_unfreezes_exactly_top2_cls_blocks(model_and_criterion):
    model, criterion = model_and_criterion
    configure_official_stage(model, criterion, epoch=STAGE_A_END + 1, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY)
    trainable = _trainable(model)
    cls_trainable = {n for n in trainable if n.startswith("cls_backbone")}
    assert cls_trainable, "Stage B 应解冻部分 cls_backbone"
    # 所有解冻的 cls 参数必属 top-2 block 前缀
    for n in cls_trainable:
        assert any(n.startswith(pfx) for pfx in STAGE_B_UNFREEZE), (
            f"Stage B 解冻了非 top-2 的 cls 参数 {n}"
        )
    # 两个 top-2 block 都确有参数被解冻(前缀没写错)
    for pfx in STAGE_B_UNFREEZE:
        assert any(n.startswith(pfx) for n in cls_trainable), f"{pfx} 未被解冻(前缀拼写?)"
    # seg 主干仍冻
    assert not any(n.startswith("seg_backbone") for n in trainable), "Stage B 不应解冻分割主干"


def test_stage_c_back_to_heads_only(model_and_criterion):
    model, criterion = model_and_criterion
    configure_official_stage(model, criterion, epoch=STAGE_B_END + 1, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY)
    trainable = _trainable(model)
    for n in trainable:
        assert n.startswith(_HEAD_PFX) or "vclub" in n, f"Stage C 只应训三头 + vclub,却有 {n}"
    assert not any(n.startswith("cls_backbone") for n in trainable), "Stage C 应重新冻结 cls Top-2"


@pytest.mark.parametrize("epoch", [1, 11, 31])
def test_optimizer_covers_all_trainable_except_vclub(model_and_criterion, epoch):
    """所有可训模型参数都进某个 param_group —— vclub 除外(独立优化器)。"""
    model, criterion = model_and_criterion
    param_groups, _lr, _name = configure_official_stage(
        model, criterion, epoch=epoch, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY
    )
    covered = {id(p) for g in param_groups for p in g["params"]}
    missing = [
        n for n, p in model.named_parameters()
        if p.requires_grad and "vclub" not in n and id(p) not in covered
    ]
    assert not missing, f"epoch {epoch}: 可训但未进 optimizer 的参数 {missing[:5]}"
    leaked = [n for n, p in model.named_parameters() if "vclub" in n and id(p) in covered]
    assert not leaked, f"epoch {epoch}: vclub 漏进主 optimizer(会被双重更新){leaked[:5]}"


def test_aucm_duals_bound_when_legacy_on_frozen_when_off(model_and_criterion):
    """freeze_legacy 决定 AUCM 对偶变量是否可训 + 是否进 optimizer。"""
    model, criterion = model_and_criterion
    duals = list(criterion.mal_loss_fn.parameters())
    assert duals, "AUCM 应有对偶参数"

    # 默认(legacy on):对偶可训且进组
    pg_on, _, _ = configure_official_stage(
        model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY, freeze_legacy=False
    )
    covered_on = {id(p) for g in pg_on for p in g["params"]}
    assert all(p.requires_grad for p in duals), "legacy on 时对偶应可训"
    assert all(id(p) in covered_on for p in duals), "legacy on 时对偶应进 optimizer"

    # freeze_legacy:对偶冻结且不进组
    pg_off, _, _ = configure_official_stage(
        model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY, freeze_legacy=True
    )
    covered_off = {id(p) for g in pg_off for p in g["params"]}
    assert not any(p.requires_grad for p in duals), "freeze_legacy 时对偶应冻结"
    assert not any(id(p) in covered_off for p in duals), "freeze_legacy 时对偶不应进 optimizer"


def test_stage_b_backbone_lr_smaller_than_head(model_and_criterion):
    """Stage B backbone(LLRD)组 lr 应远小于头 lr(= 头 lr × scale × η^i)。"""
    model, criterion = model_and_criterion
    param_groups, head_lr, _ = configure_official_stage(
        model, criterion, epoch=STAGE_A_END + 1, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY,
        backbone_lr_scale=0.1, llrd_eta=0.75,
    )
    # 收集各组 lr(去重):应有头 lr 与两档 backbone lr。
    lrs = sorted({g["lr"] for g in param_groups if g["params"]})
    assert head_lr == pytest.approx(max(lrs)), "头 lr 应是最大组"
    backbone_lrs = [x for x in lrs if x < head_lr]
    assert backbone_lrs, "Stage B 应有更小的 backbone lr 组"
    assert max(backbone_lrs) == pytest.approx(head_lr * 0.1), "blocks[6] 应 = 头 lr×0.1"
    # 逐层 LLRD:blocks[5] = 头 lr×0.1×0.75
    assert min(backbone_lrs) == pytest.approx(head_lr * 0.1 * 0.75, rel=1e-6)


# ═════════════════════════════════════════════════════════════════════════════
# BN 冻结(冻结子模块置 eval,三头保持 train)
# ═════════════════════════════════════════════════════════════════════════════


def test_frozen_bn_eval_heads_train(model_and_criterion):
    """set_frozen_modules_eval 后:冻结主干的 BN 置 eval,三头(全可训)保持 train。"""
    model, criterion = model_and_criterion
    configure_official_stage(model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY)
    model.train()
    set_frozen_modules_eval(model)
    # 三头是全可训子模块 → 应保持 training(其内含 BatchNorm2d 需正常训练)
    assert model.location_head.training, "位置头应保持 train"
    assert model.extension_head.training, "侵犯头应保持 train"
    assert model.clinical_t_head.training, "临床T头应保持 train"
    # 分割主干全冻 → 其任一 BN 子模块应 eval
    seg_bns = [m for m in model.seg_backbone.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if seg_bns:
        assert all(not m.training for m in seg_bns), "冻结分割主干的 BN 应置 eval"


# ═════════════════════════════════════════════════════════════════════════════
# JSONL 日志(V2.1 硬约束)
# ═════════════════════════════════════════════════════════════════════════════


def test_jsonl_logger_writes_required_fields(tmp_path):
    import json

    from src.training.jsonl_logger import JsonlLogger

    with JsonlLogger(log_dir=str(tmp_path), data_source="official/train_v1") as lg:
        rec = lg.log(epoch=1, step=0, phase="train", loss=2.345, lr=1e-3)
    # 必填字段齐全
    for k in ("timestamp", "epoch", "step", "phase", "mode", "data_source"):
        assert k in rec, f"缺必填字段 {k}"
    assert rec["phase"] == "train" and rec["mode"] == "training"
    # timestamp ISO 8601 带毫秒 + Z
    assert rec["timestamp"].endswith("Z") and "T" in rec["timestamp"]
    # 落盘可被逐行 JSON 解析
    path = os.path.join(str(tmp_path), "train.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(ln) for ln in f if ln.strip()]
    assert len(lines) == 1 and lines[0]["loss"] == pytest.approx(2.345)


def test_jsonl_logger_rejects_bad_phase_mode(tmp_path):
    from src.training.jsonl_logger import JsonlLogger

    with JsonlLogger(log_dir=str(tmp_path)) as lg:
        with pytest.raises(AssertionError):
            lg.log(epoch=1, step=0, phase="bogus")
        with pytest.raises(AssertionError):
            lg.log(epoch=1, step=0, phase="train", mode="bogus")
