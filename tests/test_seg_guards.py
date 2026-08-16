"""分割护栏(`evaluate_seg_guards`)的回归测试。

🔴 **本文件存在的理由(2026-08-16 Gate2 lane3 指出的盲区)**:护栏是"防白烧 24 小时
云端训练"的唯一开关,而它此前完全没有测试 —— 需要真跑训练才触得到。

⚠️ **但测试只能钉住"给定阈值下行为对不对",钉不住"这个阈值本身对不对"**。
本轮真实发生过:用 `range(SEG_FROZEN_EPOCH, SEG_ABORT_EPOCH)` 写的遍历用例,在阈值
被改回旧值后**18 项全绿一个没红** —— 常量一改遍历范围跟着变,测试永远自洽。
故凡涉及阈值语义的用例,**epoch/占比一律硬编码,不引用常量**(见 `_HARDCODED_*` 段)。

## 判据的演进(两版被实测证伪,勿再改回去)

| 版本 | 判据 | 为何被否 |
|---|---|---|
| v1 | `val_dice < 0.05` @ep12 | hard dice 先 `p>0.5` 二值化,真实病灶仅占 512² 的 0.1-3%,分割**在学**但前景 logit 未越 0 时 dice 长期恒 0 → 必然误杀 |
| v2 | `spread = p_max − p_mean < 0.01` | **实测 3/8 场景判错**(`tmp_main_spread_reverse_hole.py`):① 有结构但整体偏低(max=0.45,spread=0.396)→ 放行,而 mask 仍全零、Sclin 仍 0;② 常量图 + 4 个异常像素(spread=0.545)→ 放行。根因:max/mean 都是单点或全局标量,几个像素就能骗过 |
| **v3(当前)** | `seg_over_half_frac >= 1e-4`(**只有下界**) | 它**直接就是**预测 mask 的前景占比,正是官方 IoU 匹配消费的量,与评分口径同构 |

阈值来源:下界 `1e-4` ≈ 26 像素,而最小真实病灶 0.1% ≈ 262 像素 → 10 倍误杀余量。

🔴 **曾加过上界 0.5(防"全图判成前景"),已被实测否掉、勿再加回**:随机初始化时输出
普遍 >0.5(frac≈1.0),收敛到真值**必然穿过 0.5** —— 实测 step0 1.000 → step10 0.926
→ step20 0.555 → step25 0.219 → step60 0.0099(精确等于真值)。上界门会在 step10-20
那段把正常收敛的模型打死。见 `test_high_foreground_during_convergence_not_killed`。

⚠️ **本地冒烟(`--mock-labels --smoke`)里 `seg_over_half` 三个 epoch 全是 0.0,
这不是 v3 会误杀真实训练的证据**(独立审查核实):mock 用 64 张随机图、`mask=None`
→ 合成的稀疏前景密度≈0 且无空间结构,每 epoch 只有 4 步。真实数据 6626 样本 /
batch 16 ≈ 414 step/epoch,ep8 累计约 3300 步,且 mask 监督齐备(`n_valid_mask`
实测恒 16/16)。**但 ep8 的真实余量仍以云端首轮实测为准** —— `seg_over_half_frac`
已写进 JSONL,首轮务必回报。
"""
import importlib.util
import math
from pathlib import Path

import pytest

# `scripts/train_official.py` 不是包的一部分,按路径加载。
_SPEC = importlib.util.spec_from_file_location(
    "train_official_mod",
    Path(__file__).resolve().parents[1] / "scripts" / "train_official.py",
)
assert _SPEC is not None and _SPEC.loader is not None
train_official = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train_official)

evaluate_seg_guards = train_official.evaluate_seg_guards
SEG_FROZEN_EPOCH = train_official.SEG_FROZEN_EPOCH
SEG_FG_FRAC_LO = train_official.SEG_FG_FRAC_LO
SEG_ABORT_EPOCH = train_official.SEG_ABORT_EPOCH
SEG_ABORT_DICE = train_official.SEG_ABORT_DICE
SEG_WARN_EPOCH = train_official.SEG_WARN_EPOCH
SEG_WARN_DICE = train_official.SEG_WARN_DICE


def _m(fg=None, pmean=None, pmax=None):
    """构造 metrics dict;None 表示该诊断量缺失(不写进 dict)。"""
    out = {}
    if fg is not None:
        out["seg_over_half_frac"] = fg
    if pmean is not None:
        out["seg_prob_mean"] = pmean
    if pmax is not None:
        out["seg_prob_max"] = pmax
    return out


# ── 1. 上一轮真实失败样本必须被抓住 ──────────────────────────────────────

def test_20260815_failure_is_caught():
    """2026-08-15 云端实测态必须 abort。

    那轮 `pred_prob` 全域 0.4425~0.4463(全 <0.5)→ 前景占比恒 0 → 预测 mask 恒空
    → 官方 IoU≥0.5 匹配零命中 → 五项分类/分期零样本 → Sclin≈0。
    模型带着这个状态跑满了 35 epoch。护栏若放过它,就等于没写。
    """
    reason, msg, _w = evaluate_seg_guards(
        epoch=SEG_FROZEN_EPOCH, val_dice=0.0007,
        metrics=_m(fg=0.0, pmean=0.4454, pmax=0.4463), epochs=35,
    )
    assert reason == "seg_no_foreground"
    assert "Sclin" in msg


@pytest.mark.parametrize("epoch", [8, 12, 20, 35])
def test_20260815_failure_caught_at_every_later_epoch(epoch):
    """失败态在 ep8 及之后的任何 epoch 都该被抓,不是只在边界那一轮。"""
    reason, _m2, _w = evaluate_seg_guards(
        epoch=epoch, val_dice=0.0007, metrics=_m(fg=0.0), epochs=35,
    )
    assert reason == "seg_no_foreground", f"epoch {epoch} 漏放"


def test_not_judged_before_gate_epoch():
    """ep8 之前不判 —— 早期随机初始化本就可能没有前景,过早开枪是误杀。"""
    reason, _m2, _w = evaluate_seg_guards(
        epoch=7, val_dice=0.0007, metrics=_m(fg=0.0), epochs=35,
    )
    assert reason is None


# ── 2. v2 判据(spread)的两个漏放场景,必须被 v3 抓住 ────────────────────
#      🔴 这两条是本轮最有价值的用例:它们是我自己写的判据被自己证伪的产物。
#      若后人想改回 spread 判据,这两条会立刻红。

def test_v2_hole_structured_but_below_half():
    """【v2 漏放场景 ①】有空间结构但整体偏低 —— mask 仍全零,必须 abort。

    构造:max=0.45 / mean=0.05 → spread=0.396(v2 判据看着很健康,放行)。
    但 `pred_mask=(p>0.5)` 依然**全零**,前景占比 0 → Sclin 照样 0。
    这是 v2 最危险的失效方向:**假的安全感**,护栏看起来在工作实则形同虚设。
    """
    reason, _m2, _w = evaluate_seg_guards(
        epoch=SEG_FROZEN_EPOCH, val_dice=0.0,
        metrics=_m(fg=0.0, pmean=0.054, pmax=0.45), epochs=35,
    )
    assert reason == "seg_no_foreground", (
        "有结构但 max<0.5 的输出被放行 —— 检查是否改回了 spread 判据")


def test_v2_hole_constant_map_with_outlier_pixels():
    """【v2 漏放场景 ②】常量图 + 极少数异常像素 —— 整体无结构,必须 abort。

    构造:512² 的图里 4 个像素是 0.99、其余恒 0.4454 → spread=0.545(v2 放行)。
    前景占比 = 4/262144 ≈ 1.5e-5 < 1e-4 → v3 正确 abort。
    根因:`max` 是全验证集单点极值,几个异常像素就能骗过它。
    """
    reason, _m2, _w = evaluate_seg_guards(
        epoch=SEG_FROZEN_EPOCH, val_dice=0.0,
        metrics=_m(fg=4 / (512 * 512), pmean=0.4454, pmax=0.99), epochs=35,
    )
    assert reason == "seg_no_foreground"


@pytest.mark.parametrize("frac,step", [
    (1.000, "step 0(随机初始化)"),
    (0.926, "step 10"),
    (0.555, "step 20"),
    (0.219, "step 25"),
])
def test_high_foreground_during_convergence_not_killed(frac, step):
    """🔴 **高前景占比是收敛的必经之路,不得 abort(2026-08-16 实测否掉上界门)**。

    曾加过"占比 >0.5 → abort"防"全图判成结节",**实测会误杀正常收敛的模型**:
    随机初始化时输出普遍 >0.5(frac≈1.0),要收敛到真值(1% 前景)**必然穿过 0.5**。
    本地实测轨迹(BCE + 1% 前景 target):
        step0 1.000 → step10 0.926 → step20 0.555 → step25 0.219 → step60 0.0099
    最后精确收敛到真值 0.0099。上界门会在 step10-20 那段把它打死。

    真正的"全图判前景"崩溃态会**一直卡在高位不下来**,由 ep20 的 dice 兜底门抓
    (frac 恒 1.0 → IoU 极低 → dice 上不去)。占比本身区分不了这两者。
    """
    reason, _m2, _w = evaluate_seg_guards(
        epoch=SEG_FROZEN_EPOCH, val_dice=0.0, metrics=_m(fg=frac), epochs=35,
    )
    assert reason is None, f"{step} frac={frac} 被上界门误杀(上界门不该存在)"


def test_no_upper_bound_constant_exists():
    """钉死"上界门已移除"这个事实本身。

    若后人重新引入 `SEG_FG_FRAC_HI`,这条会红并指向上面的实测轨迹。
    """
    assert not hasattr(train_official, "SEG_FG_FRAC_HI"), (
        "上界门被重新引入了 —— 见 test_high_foreground_during_convergence_not_killed "
        "的实测:收敛必然穿过 0.5,上界门会误杀正常训练")


# ── 3. 在学的模型不得被误杀(硬编码占比,不引用常量)──────────────────────

@pytest.mark.parametrize("fg,label", [
    (0.000549, "小病灶 0.055%(接近真实下限)"),
    (0.001,    "0.1% = 官方记录的最小病灶占比"),
    (0.0099,   "1% 典型病灶"),
    (0.03,     "3% = 官方记录的最大病灶占比"),
    (0.25,     "25% 大病灶"),
])
def test_realistic_foreground_never_killed(fg, label):
    """真实病灶占比区间(0.1%~3%,含上下各一档余量)一律放行。

    🔴 `val_dice` 刻意取 0.0 —— 这正是关键场景:分割**已经输出前景**但 IoU 还没
    对齐、hard dice 仍在地板上。v1 的 dice 门会在这里误杀,v3 不会。
    占比数字硬编码,不引用 `SEG_FG_FRAC_*`,阈值被改坏时这些用例才会红。
    """
    reason, _m2, warns = evaluate_seg_guards(
        epoch=SEG_FROZEN_EPOCH, val_dice=0.0, metrics=_m(fg=fg), epochs=35,
    )
    assert reason is None, f"{label}(占比 {fg})被误杀"
    assert any("warn" in w for w in warns)   # 仍应提示 dice 偏低,但不停机


@pytest.mark.parametrize("epoch", [8, 10, 12, 15, 19])
def test_dice_gate_hardcoded_epochs_must_not_fire(epoch):
    """dice 兜底门在 ep20 之前不得触发,epoch **硬编码**。

    ep12 是关键的那个 —— 它是 v1 被否掉的旧阈值。若谁把 `SEG_ABORT_EPOCH` 改回
    12,这条立刻红。变异实验实证:用常量表达遍历范围的版本改回 ep12 后 18 项全绿,
    完全抓不住。
    """
    reason, _m2, _w = evaluate_seg_guards(
        epoch=epoch, val_dice=0.0,
        metrics=_m(fg=0.01), epochs=35,      # 前景健康 = 在学
    )
    assert reason is None, (
        f"epoch {epoch} 在前景占比健康(1%)时被 dice 门误杀 —— "
        f"检查 SEG_ABORT_EPOCH 是否被改小")


def test_dice_gate_fires_at_gamma_window_end():
    """ep20(γ=0 干净监督窗口末尾)dice 仍 <0.02 → 兜底门开枪,即便前景占比正常。"""
    reason, msg, _w = evaluate_seg_guards(
        epoch=20, val_dice=0.019, metrics=_m(fg=0.01), epochs=35,
    )
    assert reason == "seg_not_learning"
    assert "兜底" in msg


# ── 4. NaN 门(`NaN < 阈值` 恒 False,不单独拦则后面所有门静默失效)────────

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_dice_aborts(bad):
    reason, msg, _w = evaluate_seg_guards(
        epoch=1, val_dice=bad, metrics=_m(fg=0.01), epochs=35,
    )
    assert reason == "nonfinite_dice"
    assert "静默失效" in msg


def test_nan_comparison_premise_still_holds():
    """钉住 NaN 门存在的**前提**本身。

    不是在测 Python 语义,是测"我们依赖的前提没被改掉" —— 若某次重构把 dice 换成
    自定义比较类型,这条会红并提醒复查 NaN 门是否仍必要。
    """
    assert (float("nan") < 0.02) is False
    assert not math.isfinite(float("nan"))


def test_nan_gate_precedes_all_others():
    """NaN 必须最先判:即便同时前景占比为 0,也该报 nonfinite 而非 no_foreground。

    两者处理完全不同(数值失控 vs 调度没接上),报错原因搞混会让排查从零开始。
    """
    reason, _m2, _w = evaluate_seg_guards(
        epoch=SEG_FROZEN_EPOCH, val_dice=float("nan"),
        metrics=_m(fg=0.0), epochs=35,
    )
    assert reason == "nonfinite_dice"


# ── 5. 诊断量缺失时不得开枪(不拿 None 当 0)──────────────────────────────

def test_missing_fg_frac_skips_gate():
    """`seg_over_half_frac` 缺失(mock/异常路径)时跳过主门,而非判死刑。

    ⚠️ 这条同时暴露一个**真实风险**:若某个运行模式下该键根本不写,护栏就静默
    失效。`trainer.validate_epoch` 只在 `official_metrics_out` 非 None 时写入 ——
    官方训练路径恒非 None,但改动该处时必须记得这条依赖。
    """
    reason, _m2, _w = evaluate_seg_guards(
        epoch=SEG_FROZEN_EPOCH, val_dice=0.0, metrics=_m(), epochs=35,
    )
    assert reason is None


def test_old_spread_keys_alone_do_not_trigger():
    """只有旧的 `seg_prob_mean/max` 而没有 `seg_over_half_frac` 时不触发。

    防止后人以为"留着旧键就还有保护" —— v3 只认前景占比这一个键。
    """
    reason, _m2, _w = evaluate_seg_guards(
        epoch=SEG_FROZEN_EPOCH, val_dice=0.0,
        metrics=_m(pmean=0.4454, pmax=0.4463), epochs=35,
    )
    assert reason is None


# ── 6. 健康轨迹全程放行 ───────────────────────────────────────────────────

def test_healthy_run_never_aborts():
    """分割正常学起来的完整 35 epoch 轨迹,不得有任何 abort。"""
    for epoch in range(1, 36):
        dice = min(0.85, epoch * 0.03)
        reason, _m2, _w = evaluate_seg_guards(
            epoch=epoch, val_dice=dice, metrics=_m(fg=0.012), epochs=35,
        )
        assert reason is None, f"epoch {epoch} dice={dice:.2f} 被误杀"


def test_warn_fires_but_does_not_abort():
    """warn 只提示不停机 —— 价值是让云端 agent 提前盯,不是开枪。"""
    reason, _m2, warns = evaluate_seg_guards(
        epoch=SEG_WARN_EPOCH, val_dice=SEG_WARN_DICE / 2,
        metrics=_m(fg=0.012), epochs=35,
    )
    assert reason is None
    assert len(warns) == 1 and "分割 warn" in warns[0]


def test_warn_lines_returned_even_when_aborting():
    """🔴 **abort 时也必须把已收集的 warn 一并返回**(第四轮审查抓到的回归)。

    抽成纯函数前 warn 是立即 `print` 的,abort 时 stdout 上先有 warn 再崩;抽出后
    若 abort 分支直接 return 空 list,**abort 那一轮的 warn 就丢了** —— 而那恰恰是
    最该看到的诊断(dice 偏低 + 前景占比为 0 同时出现,能区分"没学起来"与"学歪了")。

    本条钉住返回值;调用点必须在 `_abort()` **之前**打印 warn_lines(见
    `train_official.main()` 的护栏块)。
    """
    reason, _m2, warns = evaluate_seg_guards(
        epoch=20, val_dice=0.001,          # 低到同时触发 warn 和 dice 兜底门
        metrics=_m(fg=0.01), epochs=35,
    )
    assert reason == "seg_not_learning"
    assert len(warns) == 1, "abort 时 warn 被丢弃了"
    assert "分割 warn" in warns[0]


def test_boundary_values_inclusive():
    """边界值语义:恰好等于上下界时**放行**(用 `<` / `>` 而非 `<=` / `>=`)。

    钉死这个是因为边界方向错了会在真实数据上产生难以复现的偶发 abort。
    """
    for fg in (SEG_FG_FRAC_LO,):
        reason, _m2, _w = evaluate_seg_guards(
            epoch=SEG_FROZEN_EPOCH, val_dice=0.5, metrics=_m(fg=fg), epochs=35,
        )
        assert reason is None, f"边界值 {fg} 被判 abort(应放行)"
