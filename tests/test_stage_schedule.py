"""三阶段训练调度的关键数值锁定测试(2026-08-11 Stage2 崩塌事故后补)。

**为什么需要这个文件**:2026-08-11 从混训 e40 权重续训 Stage2,3 个 epoch 内分割
从 97.71% 塌到 80.6%。根因是 Stage2 起点 **lr 100 倍跳变 + 全模型解冻 + mal loss
首启 + 物理先验通路满强度**四件事同 epoch 齐发。而当时:

- `tests/` 下**没有任何**测试断言 `configure_stage_parameters` 的 lr 返回值或边界行为;
- `scripts/verify/gates/gate7_optimizer_lr.py` 只在 epoch=1/41/101 三个单点验
  "lr 算得出且 optimizer 对齐",**不验跨阶段跳变幅度** —— 与 #119 同一类教训
  (只验"有变化"不验"方向/幅度",缺陷从门下穿过)。

本文件把这些人工推算的数字**钉进 CI**。注意基调:

⚠️ **本文件断言的是「当前实测值」,不是「理想值」**。100 倍跳变是**已知缺陷**,
测试断言它存在,是为了让将来任何改 `stages.py` 调度的人**必须先看到这个数字和它的
出处**,而不是为了认可它。三头训练按 `docs/训练与架构/训练架构方案.md` 另建 A/B/C 调度
(见 `stages.py` 的 Stage2 docstring 警告),**禁止复用 `configure_stage_parameters`**。

纯 CPU / 无网络 / 无权重依赖 / 不用 CUDA,可直接进 CI。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from src.models.loss import MultiTaskLoss
from src.models.multi_task_model import MultiTaskThyroidModel
from src.training.stages import (
    compute_gamma_prior,
    configure_stage_parameters,
    get_stage_number,
)

# configs/thyroid_config.yaml 的实际值(train.lr / train.weight_decay,已逐行核对)。
# 注:config 里 `lr: 0.0024` 的注释自称 "Tuned for batch size 128",而 data.batch_size
# 实际是 64 —— 这个不一致本身也是崩塌的背景因素之一,一并记录在此。
LR_BASE = 0.0024
WEIGHT_DECAY = 5e-4


@pytest.fixture(scope="module")
def model_and_criterion():
    """构造一次复用。

    `configure_stage_parameters` 每次调用都会 `set_training_stage(stage)` 全量重设
    `requires_grad`(遍历 named_parameters 逐个赋值),是幂等重置而非增量修改,
    因此 module 作用域下测试之间**不会**残留冻结状态污染。
    """
    model = MultiTaskThyroidModel(pretrained=False)
    criterion = MultiTaskLoss()
    return model, criterion


def _lr_at(model, criterion, epoch, lr_base=LR_BASE):
    _param_groups, lr, _stage_name = configure_stage_parameters(
        model, criterion, epoch=epoch, lr_base=lr_base, weight_decay=WEIGHT_DECAY
    )
    return lr


def _reference_lr(epoch, lr_base=LR_BASE):
    """独立复写一份 `stages.py` 的 lr 调度作为对照。

    刻意**不 import 实现**:实现与本参考式任何一方漂移,测试立刻红,改动者被迫
    回到本文件、读到上面的缺陷说明。
    """
    if epoch <= 5:  # Stage 1 warmup
        return lr_base * (epoch / 5.0)
    if epoch <= 40:  # Stage 1 cosine
        progress = (epoch - 5) / (40 - 5)
        return lr_base * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))
    if epoch <= 100:  # Stage 2 cosine
        progress = (epoch - 40) / (100 - 40)
        return lr_base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
    return 5e-6  # Stage 3 固定


# ═════════════════════════════════════════════════════════════════════════════
# 阶段划分
# ═════════════════════════════════════════════════════════════════════════════


def test_stage_boundaries():
    """epoch → stage 的四个边界点。改 epoch 划分必须同步改 train.py 的 --epochs 语义。"""
    assert get_stage_number(1) == 1
    assert get_stage_number(40) == 1
    assert get_stage_number(41) == 2, "epoch 41 是 Stage2 首个 epoch(崩塌起点)"
    assert get_stage_number(100) == 2
    assert get_stage_number(101) == 3
    assert get_stage_number(120) == 3


# ═════════════════════════════════════════════════════════════════════════════
# lr 调度 —— 本文件的核心
# ═════════════════════════════════════════════════════════════════════════════


def test_lr_matches_reference_curve(model_and_criterion):
    """全 120 个 epoch 的 lr 与独立参考式逐点一致(防公式被静默改写)。"""
    model, criterion = model_and_criterion
    for epoch in range(1, 121):
        got = _lr_at(model, criterion, epoch)
        want = _reference_lr(epoch)
        assert got == pytest.approx(want, rel=1e-12), (
            f"epoch {epoch}: stages.py 给 {got:.6e},参考式给 {want:.6e}。"
            "若你是有意改调度,请先读本文件头部与 stages.py 的 Stage2 警告。"
        )


def test_stage2_entry_lr_jumps_100x(model_and_criterion):
    """🔴 已知缺陷锁定:Stage1→Stage2 边界 lr 暴涨约 100 倍。

    e40 = lr_base × 0.01              = 2.400e-5   (Stage1 余弦退到底)
    e41 = lr_base × 0.99938           = 2.399e-3   (Stage2 余弦从顶开始)
    ratio ≈ 99.94

    这是 2026-08-11 分割崩塌(97.71% → 80.6%)的首要根因。**断言它存在 ≠ 认可它**
    —— 见本文件头部。此外 train.py 在 stage 边界重建 AdamW,优化器动量不承接,
    等于"从零动量 + 百倍 lr"冷启动。
    """
    model, criterion = model_and_criterion
    lr_40 = _lr_at(model, criterion, 40)
    lr_41 = _lr_at(model, criterion, 41)

    # 绝对数值(改 LR_BASE 之外的任何东西都会打破)
    assert lr_40 == pytest.approx(2.400e-5, rel=1e-4)
    assert lr_41 == pytest.approx(2.39852e-3, rel=1e-4)

    ratio = lr_41 / lr_40
    assert ratio == pytest.approx(99.94, rel=1e-3), (
        f"Stage2 入口 lr 跳变倍数 = {ratio:.2f}(历史值 99.94)。"
        "若这个数字变小了,说明有人修了调度 —— 请同步更新 stages.py 的 Stage2 "
        "docstring、docs/项目现状总纲.md §6,并确认三头训练的 A/B/C 调度不受影响。"
    )
    assert ratio > 10, "跳变倍数应远大于 1;若已降到个位数说明缺陷已被修复,请改写本测试"


def test_stage3_entry_lr_drops(model_and_criterion):
    """Stage2→Stage3 边界方向相反:lr 骤降(2.4e-4 → 5e-6),约 1/48。

    降 lr 不会炸训练,但同样是无 warmup 的硬切换,一并钉住方向。
    """
    model, criterion = model_and_criterion
    lr_100 = _lr_at(model, criterion, 100)
    lr_101 = _lr_at(model, criterion, 101)

    assert lr_100 == pytest.approx(2.400e-4, rel=1e-4)
    assert lr_101 == pytest.approx(5e-6, rel=1e-12), "Stage3 是固定 lr,不随 epoch 变"
    assert lr_101 < lr_100, "Stage3 入口应是降 lr(与 Stage2 入口的暴涨方向相反)"


def test_stage3_lr_is_constant(model_and_criterion):
    """Stage3 全程固定 5e-6,不做退火。"""
    model, criterion = model_and_criterion
    lrs = {_lr_at(model, criterion, e) for e in (101, 110, 120)}
    assert lrs == {5e-6}, f"Stage3 lr 应恒为 5e-6,实得 {lrs}"


def test_stage1_warmup_ramps_up(model_and_criterion):
    """Stage1 前 5 个 epoch 是线性 warmup(上升),不是退火。

    单独成测:下面的单调下降测试必须跳过这一段,否则会误判。
    """
    model, criterion = model_and_criterion
    warmup = [_lr_at(model, criterion, e) for e in range(1, 6)]
    assert warmup == pytest.approx(
        [LR_BASE * e / 5.0 for e in range(1, 6)], rel=1e-12
    )
    assert all(a < b for a, b in zip(warmup, warmup[1:])), "warmup 段应严格上升"
    assert warmup[-1] == pytest.approx(LR_BASE, rel=1e-12), "warmup 末端应达到 lr_base"


def test_lr_monotonic_within_stage(model_and_criterion):
    """各 stage **内部**余弦退火严格单调下降(防公式被改成反向/震荡)。

    覆盖 Stage1 退火段(6-40)与 Stage2(41-100);warmup(1-5)见上一测试。
    """
    model, criterion = model_and_criterion
    for lo, hi, label in ((6, 40, "Stage1 cosine"), (41, 100, "Stage2 cosine")):
        lrs = [_lr_at(model, criterion, e) for e in range(lo, hi + 1)]
        bad = [
            (lo + i, lrs[i], lrs[i + 1])
            for i in range(len(lrs) - 1)
            if lrs[i] <= lrs[i + 1]
        ]
        assert not bad, f"{label} 应严格单调下降,违例(epoch, lr_n, lr_n+1): {bad[:5]}"


# ═════════════════════════════════════════════════════════════════════════════
# gamma_prior 退火(物理先验通路强度)
# ═════════════════════════════════════════════════════════════════════════════


def test_gamma_prior_annealing():
    """γ_prior = clamp((epoch-20)/20, 0, 1) 的关键取值。

    该系数控制 `loss_mal → seg_backbone` 唯一非 detach 路径的强度(见函数 docstring)。
    **它恰在 epoch 40 升满,与 Stage2 起点 epoch 41 紧邻** —— 崩塌事故里"四件事同
    epoch 齐发"的第四件。
    """
    assert compute_gamma_prior(1) == 0.0
    assert compute_gamma_prior(10) == 0.0
    assert compute_gamma_prior(20) == 0.0, "epoch 20 仍是完全关闭(边界含在 0 侧)"
    assert compute_gamma_prior(21) == pytest.approx(0.05)
    assert compute_gamma_prior(30) == pytest.approx(0.5)
    assert compute_gamma_prior(39) == pytest.approx(0.95)
    assert compute_gamma_prior(40) == 1.0, "epoch 40 升满 —— 紧邻 Stage2 起点 41"
    assert compute_gamma_prior(41) == 1.0
    assert compute_gamma_prior(120) == 1.0, "升满后不回落"


def test_gamma_prior_clamped_and_monotonic():
    """全程落在 [0,1] 且单调不减(防公式写反或越界)。"""
    vals = [compute_gamma_prior(e) for e in range(0, 130)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert all(a <= b for a, b in zip(vals, vals[1:]))


def test_train_step_uses_shared_gamma_prior():
    """`train_step.py` 必须调用本函数,不能再内联一份公式(防两处漂移)。

    崩塌复盘时 γ 的公式内联在 `train_batch` 里、无法单测,才被提取成模块级函数。
    """
    import inspect

    from src.training.loop import train_step

    src = inspect.getsource(train_step.train_batch)
    assert "compute_gamma_prior(epoch)" in src, "train_batch 应调用 compute_gamma_prior"
    assert "(epoch - 20) / 20" not in src, "内联公式已提取,不应在 train_batch 里复现"


# ═════════════════════════════════════════════════════════════════════════════
# 冻结集合 / 优化器覆盖
# ═════════════════════════════════════════════════════════════════════════════


def test_stage_frozen_sets(model_and_criterion):
    """各 stage 的冻结/解冻集合与 `is_param_trainable` 的注册表一致。

    Stage1:只冻 CBM 直连头(`classifier_direct`/`latent_mlp`),**主干可训**
            —— 勿把"Stage1 = 头 warmup"误读为冻主干。
    Stage2:全解冻(零冻结)—— 这正是崩塌根因之二。
    Stage3:冻 `STAGE3_FROZEN_PREFIXES` 全部前缀,只留 CBM 头 + vclub。
    """
    model, criterion = model_and_criterion

    # ── Stage 1 ──
    configure_stage_parameters(
        model, criterion, epoch=1, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY
    )
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert frozen, "Stage1 至少应冻结 CBM 直连头"
    assert all(
        any(sub in n for sub in MultiTaskThyroidModel.STAGE1_FROZEN_CBM_SUBMODULES)
        for n in frozen
    ), f"Stage1 只应冻 {MultiTaskThyroidModel.STAGE1_FROZEN_CBM_SUBMODULES},实冻 {sorted(frozen)[:5]}"
    assert any(n.startswith("seg_backbone") for n in trainable), "Stage1 主干必须可训"
    assert any("vclub" in n for n in trainable), "vclub 在所有阶段恒可训(独立优化器)"

    # ── Stage 2:全解冻 ──
    configure_stage_parameters(
        model, criterion, epoch=41, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY
    )
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert not frozen, (
        f"Stage2 设计上全模型解冻(崩塌根因之二),实有冻结 {sorted(frozen)[:5]}。"
        "若你已把它改成 LLRD/部分解冻,请同步 stages.py docstring 与本测试。"
    )

    # ── Stage 3 ──
    configure_stage_parameters(
        model, criterion, epoch=101, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY
    )
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    prefixes = MultiTaskThyroidModel.STAGE3_FROZEN_PREFIXES
    assert all(any(n.startswith(p) for p in prefixes) for n in frozen)
    # 反向:被列进前缀表的参数(vclub 除外)必须真的冻上,防前缀拼写错导致静默不生效
    should_freeze = {
        n
        for n, _ in model.named_parameters()
        if any(n.startswith(p) for p in prefixes) and "vclub" not in n
    }
    assert should_freeze <= frozen, f"这些应冻未冻:{sorted(should_freeze - frozen)[:5]}"
    assert any(n.startswith("cbm_head") for n in trainable), "Stage3 只训 CBM 头"
    assert any("vclub" in n for n in trainable)


@pytest.mark.parametrize("epoch", [1, 41, 101])
def test_optimizer_covers_all_trainable_except_vclub(model_and_criterion, epoch):
    """所有 `requires_grad=True` 的参数都进了某个 param_group —— vclub 除外。

    两个方向都要守(`QA_VP.md §1.2` 优化器绑定):
    - **漏进**:解冻了却不在 optimizer → 该参数永不更新,静默失效;
    - **误进**:vclub 由独立 `vclub_optimizer` 管(`stages.py` 用 `"vclub" in name`
      显式排除),若混进主 param_group 会被两个优化器双重更新。
    """
    model, criterion = model_and_criterion
    param_groups, _lr, _name = configure_stage_parameters(
        model, criterion, epoch=epoch, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY
    )
    covered = {id(p) for g in param_groups for p in g["params"]}

    missing = [
        n
        for n, p in model.named_parameters()
        if p.requires_grad and "vclub" not in n and id(p) not in covered
    ]
    assert not missing, f"epoch {epoch}: 可训练但未进 optimizer 的参数 {missing[:5]}"

    leaked = [
        n for n, p in model.named_parameters() if "vclub" in n and id(p) in covered
    ]
    assert not leaked, f"epoch {epoch}: vclub 参数漏进主 optimizer(会双重更新){leaked[:5]}"


@pytest.mark.parametrize(
    "epoch,expect_bound",
    [(1, False), (41, True), (101, True)],
)
def test_aucm_dual_params_optimizer_binding(model_and_criterion, epoch, expect_bound):
    """AUCM 对偶变量(`a`/`b`/`alpha`)的优化器绑定 —— `QA_VP.md §1.2` 领域红线。

    上一个测试只遍历 `model.named_parameters()`,**看不见 `criterion` 上的参数**。
    而 AUCM 的拉格朗日对偶变量是 `criterion.mal_loss_fn` 的 `nn.Parameter`,由
    `configure_stage_parameters` 在 `stage >= 2` 时并入 `active_no_decay`
    (刻意不加 weight_decay,避免衰减拉扯乘子)。

    QA_VP §1.2 原文:损失/正则模块内的**所有**可学习变量必须显式注册并验证在活跃
    optimizer 内。漏绑的后果不是崩溃而是**静默退化**——对偶变量永不更新,minimax
    的内层最大化失效,AUCM 退化成一个普通的 margin 损失,而训练日志一切正常。

    Stage 1:mal loss 关闭(`enable_mal_loss=(stage>1)`),对偶变量被冻结且**不应**入组;
    Stage 2/3:解冻且**必须**入组。
    """
    model, criterion = model_and_criterion
    param_groups, _lr, _name = configure_stage_parameters(
        model, criterion, epoch=epoch, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY
    )
    covered = {id(p) for g in param_groups for p in g["params"]}
    duals = list(criterion.mal_loss_fn.parameters())
    assert duals, "AUCM 应有对偶参数(a/b/alpha);若已改实现,请同步本测试与 stages.py"

    if expect_bound:
        assert all(p.requires_grad for p in duals), f"epoch {epoch}: 对偶变量应解冻"
        unbound = [i for i, p in enumerate(duals) if id(p) not in covered]
        assert not unbound, (
            f"epoch {epoch}: AUCM 对偶变量 {unbound} 未进 optimizer —— "
            "它们会永不更新,minimax 内层失效而训练日志看不出异常"
        )
        # 对偶变量必须落在 weight_decay=0 的组里
        for g in param_groups:
            if any(id(p) in {id(d) for d in duals} for p in g["params"]):
                assert g["weight_decay"] == 0.0, (
                    "拉格朗日乘子不得被 weight_decay 衰减拉扯"
                )
    else:
        assert not any(p.requires_grad for p in duals), f"epoch {epoch}: 对偶变量应冻结"
        leaked = [i for i, p in enumerate(duals) if id(p) in covered]
        assert not leaked, f"epoch {epoch}(mal loss 关闭): 对偶变量 {leaked} 不应入组"


@pytest.mark.parametrize("epoch", [1, 41, 101])
def test_all_param_groups_share_stage_lr(model_and_criterion, epoch):
    """param_group 的 lr 与返回的 stage_lr 一致(gate7 同款断言,补进 CI 侧)。

    gate7 只在 A800/CI 跑 `verify_model.py` 时才验;这里让本地 pytest 也守住。
    """
    model, criterion = model_and_criterion
    param_groups, stage_lr, _name = configure_stage_parameters(
        model, criterion, epoch=epoch, lr_base=LR_BASE, weight_decay=WEIGHT_DECAY
    )
    optimizer = torch.optim.AdamW(param_groups, lr=stage_lr)
    for i, pg in enumerate(optimizer.param_groups):
        # LoRA 组是例外(固定 1e-4,不随调度);非 LoRA 模型下不会出现
        assert pg["lr"] == stage_lr, f"epoch {epoch}: param_group[{i}] lr={pg['lr']} != {stage_lr}"


# ═════════════════════════════════════════════════════════════════════════════
# 混训采样器:每样本期望抽中次数
# ═════════════════════════════════════════════════════════════════════════════


def test_mixtrain_sampler_expected_draws():
    """`WeightedRandomSampler` 下每样本的期望抽中次数 = num_samples × w_i / Σw。

    **背景**:`train.py` 原注释声称"TN3K 每样本 ~1.2 次、XL ~0.8 次",**算错了**
    (漏了除以 Σw = Σratio = 1.6)。真实值在生产配比(TN3K n=2742 ratio=1.0 /
    ThyroidXL n=9882 ratio=0.6)下是 **TN3K ≈ 2.88 次、XL ≈ 0.48 次**。注释已同批修正。

    注意与另一个数区分量纲:`train.py` 打印的 **62.5% : 37.5%** 是 *draw 总量分布*
    (= ratio / Σratio),那个数是对的,和"每样本抽中次数"不是一回事。

    本测试断言**不变量**而非硬编码 2.88/0.48 —— 那两个数依赖两源实际样本数,
    数据一变就漂。这里用小规模合成配比,同时做解析式与蒙特卡洛双向对照。

    权重来源(`per_sample_w = ratio / n`)由
    `tests/test_multisource_seg_smoke.py::test_sample_weights_by_ratio` 用真实
    `MultiSourceThyroidDataset` 覆盖,本测试只接续验证「权重 → 期望抽中次数」这一步,
    语义不重叠、无需真实数据目录。
    """
    n_tn, ratio_tn = 15, 1.0
    n_xl, ratio_xl = 50, 0.6
    weights = [ratio_tn / n_tn] * n_tn + [ratio_xl / n_xl] * n_xl
    num_samples = n_tn + n_xl  # train.py 用 num_samples=len(mds)
    total_w = ratio_tn + ratio_xl  # Σw = Σratio(逐源权重之和恰为其 ratio)

    assert sum(weights) == pytest.approx(total_w), "Σw 应等于 Σratio"

    exp_tn = num_samples * (ratio_tn / n_tn) / total_w
    exp_xl = num_samples * (ratio_xl / n_xl) / total_w

    # ── 不变量①:期望抽中次数之比 = ratio 之比 × 样本数之反比 ──
    assert exp_tn / exp_xl == pytest.approx((ratio_tn / ratio_xl) * (n_xl / n_tn))
    # ── 不变量②:总抽中次数守恒 ──
    assert n_tn * exp_tn + n_xl * exp_xl == pytest.approx(num_samples)
    # ── 不变量③:draw 总量分布 = ratio / Σratio(即那个 62.5% : 37.5%)──
    assert (n_tn * exp_tn) / num_samples == pytest.approx(ratio_tn / total_w)
    assert (n_xl * exp_xl) / num_samples == pytest.approx(ratio_xl / total_w)
    # ── 小样本源被抽中次数 > 1,大样本源 < 1(配比意图的方向)──
    assert exp_tn > 1.0 > exp_xl

    # ── 蒙特卡洛:真跑 PyTorch 的 sampler,验解析式没写错 ──
    generator = torch.Generator().manual_seed(1234)
    rounds = 300
    counts = torch.zeros(num_samples)
    for _ in range(rounds):
        sampler = WeightedRandomSampler(
            weights, num_samples=num_samples, replacement=True, generator=generator
        )
        for idx in sampler:
            counts[idx] += 1

    mc_tn = (counts[:n_tn].sum() / (n_tn * rounds)).item()
    mc_xl = (counts[n_tn:].sum() / (n_xl * rounds)).item()
    assert mc_tn == pytest.approx(exp_tn, rel=0.05), f"TN3K 实测 {mc_tn:.3f} vs 解析 {exp_tn:.3f}"
    assert mc_xl == pytest.approx(exp_xl, rel=0.05), f"XL 实测 {mc_xl:.3f} vs 解析 {exp_xl:.3f}"


def test_mixtrain_production_ratio_numbers():
    """生产配比下的具体数字(TN3K 2742 / ThyroidXL 9882,ratio 1.0 : 0.6)。

    锁死 `train.py` 注释里写的 2.88 / 0.48,以及"XL 每 epoch 去重可见 ≈3760 张"。
    样本数改变时本测试会红 —— 那正是提醒去同步注释的时机。
    """
    n_tn, n_xl = 2742, 9882
    ratio_tn, ratio_xl = 1.0, 0.6
    num_samples = n_tn + n_xl
    total_w = ratio_tn + ratio_xl

    exp_tn = num_samples * (ratio_tn / n_tn) / total_w
    exp_xl = num_samples * (ratio_xl / n_xl) / total_w

    assert exp_tn == pytest.approx(2.88, abs=0.01), f"TN3K 期望抽中 {exp_tn:.3f}"
    assert exp_xl == pytest.approx(0.48, abs=0.01), f"XL 期望抽中 {exp_xl:.3f}"
    assert exp_tn / exp_xl == pytest.approx(6.0, abs=0.01)

    # 去重可见张数:每样本至少被抽中一次的概率 ≈ 1 - e^(-E)(泊松近似)
    unique_xl = n_xl * (1 - math.exp(-exp_xl))
    assert unique_xl == pytest.approx(3760, abs=60), f"XL 去重可见 ≈{unique_xl:.0f} 张"

    # 原注释声称的 1.2 / 0.8 与真值差 2.4 倍 —— 钉住"不是那个数"
    assert exp_tn > 2.0, "若这里 ≈1.2,说明有人按旧错误注释改了采样逻辑"
