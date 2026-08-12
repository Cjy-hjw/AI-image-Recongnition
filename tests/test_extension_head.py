"""侵犯头（ETE）契约测试（Phase2 阶段 2.2）。

守的同样是「错了不会崩、只会静默学错」那一类：

  1. 交界带的**方向**（膨胀−腐蚀，不是反过来）与**带宽**（= kernel−1，实测值）。
  2. 阴性病例（空 mask）分母为 0 时不得产出 NaN。
  3. CBM `margin` 必须 detach —— 否则 ETE 的梯度会重塑「边缘」概念、毁掉 CBM 报告。
  4. 🔴 Logit Adjustment 的**符号**：反了会放大不平衡，而损失照常下降。
     这里用收敛实验做方向断言，不是"有变化就算过"（#119 教训）。
  5. CB 权重按**真实标签**索引，不是按预测。
  6. β 的实跑数值（646.9 : 1 这种量级的坑必须钉住，防有人"顺手"抄论文默认值）。
"""

import pytest
import torch

from src.models.heads.extension import (
    ETE_IDX_TO_CODE,
    NUM_ETE_CLASSES,
    ExtensionHead,
)
from src.models.label_schema import ETE_CODES
from src.models.losses.components.extension import (
    class_balanced_weights,
    compute_extension_loss,
    log_class_prior,
)
from src.models.mask_features import boundary_band_from_mask

IN_CH = 352  # = MultiTaskThyroidModel.CLS_POOLED_DIM
# 官方全量实测（docs/训练与架构/训练数据定稿.md:85）：无 / 包膜 / 软组织 / 血管
OFFICIAL_ETE_COUNTS = [3906, 267, 67, 5]


def _square_mask(canvas=40, lo=10, hi=30):
    m = torch.zeros(1, 1, canvas, canvas)
    m[:, :, lo:hi, lo:hi] = 1.0
    return m


# ── 1. 交界带形态学 ────────────────────────────────────────────────────────


@pytest.mark.parametrize("k, expect_width", [(3, 2), (5, 4), (7, 6)])
def test_band_width_is_kernel_minus_one(k, expect_width):
    """带宽 = kernel_size − 1（实测），不是 kernel_size。

    这条钉住的是一个**容易想当然**的量：`docs/训练与架构/训练架构方案.md` 写「5px 交界带」，
    照字面把 kernel 设 5 得到的其实是 4px。数值本身不重要，重要的是别让后人
    按错误的换算去调核宽。
    """
    band = boundary_band_from_mask(_square_mask(), kernel_size=k)
    row = band[0, 0, 20]  # 穿过方块中线
    cols = (row > 0).nonzero().flatten().tolist()
    # 方块占 col 10..29，左右各一段带；每段宽度应为 expect_width
    left = [c for c in cols if c < 20]
    right = [c for c in cols if c >= 20]
    assert len(left) == expect_width, f"左侧带宽 {len(left)} != {expect_width}, cols={cols}"
    assert len(right) == expect_width, f"右侧带宽 {len(right)} != {expect_width}"
    # 且必须跨在轮廓两侧（内外各一半），不是整段都在外面
    assert min(left) < 10 <= max(left), f"带没有跨过轮廓: {left}"


def test_band_is_zero_inside_and_far_outside():
    """🔴 方向反证：若实现写成 erode − dilate，这里全 0，测试立刻红。

    同时锁住"带只在界面附近"这一语义 —— 结节实心内部的纹理与 ETE 无关。
    """
    band = boundary_band_from_mask(_square_mask(), kernel_size=5)
    assert band[0, 0, 20, 20].item() == 0.0, "结节正中心不该有带"
    assert band[0, 0, 2, 2].item() == 0.0, "远离结节处不该有带"
    assert band.max().item() > 0.0, "轮廓处必须有带（实现方向可能反了）"
    assert (band >= 0).all(), "带权重不得为负（负权重会让加权池化分子分母反号）"


@pytest.mark.parametrize(
    "mask, why",
    [
        (torch.zeros(1, 1, 20, 20), "空 mask = 阴性病例，无界面"),
        (torch.ones(1, 1, 20, 20), "全 1 mask = 分割崩溃判整图前景，同样无界面"),
    ],
)
def test_band_degenerate_masks_give_zero(mask, why):
    """两种退化 mask 都必须给 0 带 —— 下游加权池化的分母因此可能为 0，必须能吞下。"""
    assert boundary_band_from_mask(mask, kernel_size=5).sum().item() == 0.0, why


def test_band_has_no_band_along_image_border():
    """贴图像边界的结节，其贴边那一侧没有带（padding 语义，**有意保留**）。

    `F.max_pool2d` 隐式 −inf padding → 腐蚀在越界方向取不到背景值、不会内缩。
    图像边缘外的解剖结构本就未知，在那里伪造界面只会喂给 ETE 头假证据。
    若哪天 torch 改了 padding 语义，这条会红并提示复议。
    """
    m = torch.zeros(1, 1, 20, 20)
    m[:, :, 0:10, 0:10] = 1.0
    band = boundary_band_from_mask(m, kernel_size=5)
    assert band[0, 0, 0, 0].item() == 0.0, "图像角点不该有带"
    assert band[0, 0, 9, 5].item() > 0.0, "块的内侧边界必须有带"


def test_band_is_detached_even_from_grad_carrying_mask():
    m = _square_mask().requires_grad_(True)
    band = boundary_band_from_mask(m, kernel_size=5)
    assert not band.requires_grad, "带必须已 detach（三头梯度不得回流分割主干）"


def test_band_rejects_even_kernel_and_wrong_rank():
    with pytest.raises(AssertionError, match="奇数"):
        boundary_band_from_mask(_square_mask(), kernel_size=4)
    with pytest.raises(AssertionError, match=r"\[B,1,H,W\]"):
        boundary_band_from_mask(torch.zeros(1, 3, 8, 8), kernel_size=5)


# ── 2. 类别编码顺序 ────────────────────────────────────────────────────────


def test_ete_code_mapping_is_one_based_and_locked():
    """0-based 头下标 i ↔ 官方输出码 i+1。错位不会报错，只会把包膜填进"无"。"""
    assert NUM_ETE_CLASSES == 4
    assert tuple(ETE_IDX_TO_CODE) == tuple(ETE_CODES) == (1, 2, 3, 4)
    for i, code in enumerate(ETE_IDX_TO_CODE):
        assert code == i + 1


# ── 3. CB 权重与先验 ───────────────────────────────────────────────────────


def test_cb_weight_ratio_at_official_counts():
    """🔴 钉死 β 的真实后果 —— 防有人"顺手"把 β 抄成论文常用的 0.9999。

    n=5 时 β=0.9999 的权重比达 646.9:1，等于让 5 张图吃掉整个损失、必然过拟合到
    这 5 张图；这是文献里公认会崩的近似逆频率加权。默认 0.99 给出 20.4:1。
    """
    ratios = {}
    for beta in (0.9, 0.99, 0.999, 0.9999):
        w = class_balanced_weights(OFFICIAL_ETE_COUNTS, beta)
        ratios[beta] = float(w[3] / w[0])
    assert ratios[0.99] == pytest.approx(20.4, abs=0.1)
    assert ratios[0.9999] == pytest.approx(646.9, abs=1.0)
    # 单调性：β 越大越偏袒稀有类
    assert ratios[0.9] < ratios[0.99] < ratios[0.999] < ratios[0.9999]


def test_cb_weights_normalized_to_mean_one():
    """归一化到均值 1，使"开关 CB"不顺带改变 ETE 损失相对其它任务的量级。"""
    for beta in (0.0, 0.9, 0.99, 0.9999):
        w = class_balanced_weights(OFFICIAL_ETE_COUNTS, beta)
        assert float(w.mean()) == pytest.approx(1.0, abs=1e-5)


def test_cb_beta_zero_disables_weighting():
    w = class_balanced_weights(OFFICIAL_ETE_COUNTS, 0.0)
    assert torch.allclose(w, torch.ones(4))


def test_cb_rejects_zero_count_class():
    """计数为 0 → 有效样本数 0 → 权重 inf 会静默毒化损失，必须 fail-loud。"""
    with pytest.raises(AssertionError, match="必须全 > 0"):
        class_balanced_weights([100, 0, 5, 1], 0.99)


def test_cb_float32_vs_float64_agree_at_realistic_counts():
    """如实钉住：在本任务的量级上，float32 与 float64 的 CB 权重**没有实质差异**。

    本测试第一版是**假门**（断言 `w[0] != w[1]`，而 float32 同样区分得开），第二版
    想改用 n=20000 制造差异也失败了 —— 因为 `class_balanced_weights` 末尾会
    `.to(float32)`，float64 多算出的那点差异在返回时又被舍掉。

    与其把测试拧到能过，不如记录实测真相：β∈[0.99, 0.99999] × 官方计数下，
    float32 与 float64 的权重**最大相对差仅 4.2e-5**。`1 − β^n` 确实是教科书级的
    灾难性相消形式（β=0.99999,n=5 时 float32 算出 5.00679e-5、float64 算出
    4.99990e-5，差 0.14%），但归一化之后全被吃掉。

    → **float64 是零成本的正确写法，不是必需的防护**。本测试的价值是：若将来
    数据集量级变化到让这个假设失效（相对差超过 1e-3），它会立刻红，提醒去复议。
    """
    for beta in (0.99, 0.999, 0.9999, 0.99999):
        w64 = class_balanced_weights(OFFICIAL_ETE_COUNTS, beta)
        c32 = torch.as_tensor(OFFICIAL_ETE_COUNTS, dtype=torch.float32)
        eff32 = (1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), c32)) / (1.0 - beta)
        w32 = 1.0 / eff32
        w32 = w32 / w32.mean()
        rel = float(((w32 - w64).abs() / w64.abs()).max())
        assert rel < 1e-3, (
            f"β={beta} 下 float32/float64 权重相对差 {rel:.3e} 超过 1e-3 —— "
            "精度假设失效，需复议是否要在返回值上也保持 float64"
        )


def test_log_prior_matches_counts():
    lp = log_class_prior(OFFICIAL_ETE_COUNTS)
    total = sum(OFFICIAL_ETE_COUNTS)
    for i, n in enumerate(OFFICIAL_ETE_COUNTS):
        assert float(lp[i]) == pytest.approx(torch.log(torch.tensor(n / total)).item(), abs=1e-5)


# ── 4. 损失 ────────────────────────────────────────────────────────────────


def test_loss_weights_are_indexed_by_label_not_prediction():
    """🔴 权重必须按**真实标签**索引。

    按预测索引的话，模型只要"少预测稀有类"就能自己把损失权重降下来 —— 与校正
    意图完全相反，且不会有任何报错。这里用全 0 logits（各类 log_pt 相同）隔离出
    权重因子：两样本标签 0/3、权重 1/5，总损失应恰为无权重版的 3 倍。
    """
    logits = torch.zeros(2, 4)
    idx = torch.tensor([0, 3])
    v = torch.ones(2)
    w = torch.tensor([1.0, 1.0, 1.0, 5.0])
    base, _ = compute_extension_loss(logits, idx, v, focal_gamma=0.0)
    weighted, _ = compute_extension_loss(logits, idx, v, class_weights=w, focal_gamma=0.0)
    assert float(weighted / base) == pytest.approx(3.0, abs=1e-5)
    # 两个样本都标成稀有类 → 应恰为 5 倍（若按预测索引，logits 全 0 时不可能变）
    both_rare, _ = compute_extension_loss(
        logits, torch.tensor([3, 3]), v, class_weights=w, focal_gamma=0.0
    )
    assert float(both_rare / base) == pytest.approx(5.0, abs=1e-5)


def test_logit_adjustment_sign_pushes_toward_rare_class():
    """🔴 **方向断言**（不是"有变化就算过"）—— LA 的符号反了会放大不平衡。

    构造一个只有偏置的模型（logits = 可学参数，与输入无关），在 100:5 的不平衡
    数据上训到收敛。理论最优：
      - τ=0（纯 CE）：logits 收敛到 log 先验 → gap = log(100/5) ≈ **3.00**
      - τ=1（**加** log π，正确）：最优是 params + log π = log π → gap → **0**
      - τ=1 但符号写成**减**（错误）：最优 params = 2·log π → gap ≈ **6.00**
    三者两两可分，故本测试对符号错误有真实区分力（#119 教训：只验有变化不验方向
    正是 R6 FAMO 漏网的根因）。
    """
    counts = [100, 5]
    idx = torch.cat([torch.zeros(100, dtype=torch.long), torch.ones(5, dtype=torch.long)])
    v = torch.ones(105)
    lp = log_class_prior(counts)

    def train(tau, sign=+1.0):
        params = torch.zeros(1, 2, requires_grad=True)
        opt = torch.optim.Adam([params], lr=0.05)
        # 300 步足够收敛：实测 3 个 seed 下 gap 稳定在 (2.996, 0.000, 5.896)，与 2000 步
        # 的 (2.996, -0.007, 5.991) 判定完全一致，而耗时从 9.5s 降到 ~2s。反号对照组
        # 与正确实现的差距近 6 个单位，300 步远在安全边际内（50 步才开始退化）。
        for _ in range(300):
            opt.zero_grad()
            loss, _ = compute_extension_loss(
                params.expand(105, 2),
                idx,
                v,
                focal_gamma=0.0,
                la_tau=sign * tau,
                log_prior=lp,
            )
            loss.backward()
            opt.step()
        return float((params[0, 0] - params[0, 1]).detach())

    gap_plain = train(0.0)
    gap_la = train(1.0)
    # 先确认实验装置本身有效：纯 CE 必须复现 log(100/5)
    assert gap_plain == pytest.approx(3.0, abs=0.15), (
        f"实验装置失效：纯 CE 应收敛到 log 先验 gap≈3.0，实得 {gap_plain:.3f}"
    )
    # 正确符号：LA 把 gap 压向 0（稀有类相对 logit 被抬高）
    assert abs(gap_la) < 0.5, (
        f"LA(τ=1) 应把 logit gap 压到 ~0，实得 {gap_la:.3f} —— 符号可能反了"
    )
    # 区分力自证：符号写反会把 gap 放大到 ~6，与正确实现相差一个数量级
    gap_wrong_sign = train(1.0, sign=-1.0)
    assert gap_wrong_sign > gap_plain + 1.0, (
        f"反号对照组未放大 gap（{gap_wrong_sign:.3f} vs {gap_plain:.3f}），"
        "说明本测试对符号错误没有区分力，需重新设计"
    )


def test_loss_rejects_2d_validity_that_would_broadcast_silently():
    """[B,1] validity 会隐式广播成 [B,B] 静默放大 loss（QA_VP §2.2）。"""
    logits = torch.randn(3, 4)
    with pytest.raises(AssertionError, match=r"必须是 \[B\] 一维"):
        compute_extension_loss(logits, torch.tensor([0, 1, 2]), torch.ones(3, 1))


def test_loss_rejects_out_of_range_labels_both_directions():
    """越界正负不对称：idx=4 会 IndexError，idx=-1 静默取到最稀有的"血管"类。"""
    logits = torch.randn(2, 4)
    for bad in (4, -1):
        with pytest.raises(AssertionError, match="ETE 标签越界"):
            compute_extension_loss(logits, torch.tensor([bad, 0]), torch.ones(2))


def test_loss_zero_and_no_grad_when_all_invalid():
    logits = torch.randn(4, 4, requires_grad=True)
    total, _ = compute_extension_loss(
        logits, torch.tensor([0, 1, 2, 3]), torch.zeros(4)
    )
    assert total.item() == 0.0 and torch.isfinite(total)
    total.backward()
    assert logits.grad.abs().sum().item() == 0.0


def test_loss_denominator_counts_only_valid_samples():
    torch.manual_seed(0)
    logits = torch.randn(3, 4)
    idx = torch.tensor([0, 1, 2])
    masked, _ = compute_extension_loss(logits, idx, torch.tensor([0.0, 1.0, 0.0]))
    single, _ = compute_extension_loss(logits[1:2], idx[1:2], torch.ones(1))
    assert masked.item() == pytest.approx(single.item(), abs=1e-6)


def test_loss_reduces_to_plain_ce_without_focal_and_weights():
    """γ=0 且无权重时必须等于普通 masked CE —— 保证"关掉校正"是真的关掉。"""
    torch.manual_seed(0)
    logits = torch.randn(5, 4)
    idx = torch.tensor([0, 1, 2, 3, 0])
    v = torch.ones(5)
    total, detail = compute_extension_loss(logits, idx, v, focal_gamma=0.0)
    ref = torch.nn.functional.cross_entropy(logits, idx)
    assert total.item() == pytest.approx(ref.item(), abs=1e-5)
    assert detail["ete_ce"].item() == pytest.approx(ref.item(), abs=1e-5)


def test_focal_downweights_easy_samples():
    """γ>0 必须让已学好的样本贡献变小（方向断言，不只是"数值变了"）。"""
    easy = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    hard = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    idx, v = torch.tensor([0]), torch.ones(1)
    e0, _ = compute_extension_loss(easy, idx, v, focal_gamma=0.0)
    e2, _ = compute_extension_loss(easy, idx, v, focal_gamma=2.0)
    h0, _ = compute_extension_loss(hard, idx, v, focal_gamma=0.0)
    h2, _ = compute_extension_loss(hard, idx, v, focal_gamma=2.0)
    assert e2 / e0 < h2 / h0, "Focal 未按难度差异化压制（易样本应被压得更狠）"


def test_la_requires_log_prior():
    with pytest.raises(AssertionError, match="log_prior"):
        compute_extension_loss(
            torch.randn(2, 4), torch.tensor([0, 1]), torch.ones(2), la_tau=1.0
        )


def test_loss_rejects_fractional_focal_gamma_that_makes_nan_grads():
    """🔴 γ∈(0,1) 在 p_t=1.0 处梯度是 NaN，且 **loss 值看不出异常**。

    `d/dp (1-p)^γ = -γ(1-p)^(γ-1)`，γ<1 时指数为负 → 0^负 = inf。而 p_t == 1.0
    在 fp32 下真的可达（实测 logit gap ≥ 50 时 softmax 精确返回 1.0），训练中后期
    大量易样本都在这个区间。实测 γ=0.5 → loss=0.0 但 grad=[nan, nan]。
    """
    logits = torch.tensor([[200.0, 0.0, 0.0, 0.0]])
    with pytest.raises(AssertionError, match=r"focal_gamma"):
        compute_extension_loss(logits, torch.tensor([0]), torch.ones(1), focal_gamma=0.5)
    # 反证：确认这个危险确实存在（若哪天 torch 改了 0^负 的语义，本防护可复议）
    lg = logits.clone().requires_grad_(True)
    pt = torch.softmax(lg, -1)[0, 0]
    assert float(pt) == 1.0, "对照组失效：p_t 未能取到精确 1.0，本防护的前提已变"
    ((1.0 - pt).pow(0.5)).backward()
    assert not torch.isfinite(lg.grad).all(), (
        "γ=0.5 在 p_t=1.0 处竟给出有限梯度 —— 本断言的前提已变，请复议是否还需拦截"
    )


@pytest.mark.parametrize("gamma", [0.0, 1.0, 2.0, 5.0])
def test_loss_finite_under_extreme_logits(gamma):
    """QA_VP §1.3：极端 logits（±1000）下损失与梯度必须保持有限、无 NaN/Inf。

    这类值在训练早期不常见，但 fp16 推理链路 + 未校准的头很容易冲到几百量级；
    `log_softmax` 的数值稳定实现是唯一屏障，必须实证而非假定。
    """
    for scale in (1000.0, -1000.0):
        logits = (torch.eye(4) * scale).requires_grad_(True)
        total, detail = compute_extension_loss(
            logits,
            torch.tensor([0, 1, 2, 3]),
            torch.ones(4),
            class_weights=class_balanced_weights(OFFICIAL_ETE_COUNTS, 0.99),
            log_prior=log_class_prior(OFFICIAL_ETE_COUNTS),
            focal_gamma=gamma,
            la_tau=1.0,
        )
        assert torch.isfinite(total), f"scale={scale} γ={gamma} loss 非有限: {total}"
        assert torch.isfinite(detail["ete_ce"]), f"scale={scale} ce 非有限"
        total.backward()
        assert torch.isfinite(logits.grad).all(), (
            f"scale={scale} γ={gamma} 梯度含 NaN/Inf: {logits.grad}"
        )


def test_head_finite_under_bf16_autocast():
    """QA_VP §1.3 的 autocast 分支：整头在 autocast 下不得产出 NaN/Inf。

    这里用 bf16（CPU 上 torch 不支持 fp16 autocast）。**真 fp16 只有 GPU 能验**，
    由 `scripts/probe_heads_cuda.py` 覆盖 —— 那才是交付推理的真实精度
    （`model_runner.run_forward(precision="fp16")`）。两者缺一不可：本测试保证
    CI 上有 autocast 覆盖，探针保证上云前有真 fp16 覆盖。
    """
    head = ExtensionHead(in_channels=IN_CH).eval()
    feat, mask, margin = _head_inputs(b=2)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = head(feat * 100.0, mask, margin * 100.0)
    assert torch.isfinite(out["ete_logits"]).all(), "bf16 autocast 下 logits 非有限"
    assert torch.allclose(out["ete_probs"].sum(-1), torch.ones(2), atol=1e-3)
    assert out["ete_probs"].dtype == torch.float32, (
        "概率必须提升到 fp32 —— 稀有类概率在 1e-3 量级，低精度会被抹成 0"
    )


# ── 5. ExtensionHead ───────────────────────────────────────────────────────


def _head_inputs(b=2, ch=IN_CH, hw=16, img=64):
    feat = torch.randn(b, ch, hw, hw)
    mask = torch.zeros(b, 1, img, img)
    mask[:, :, img // 4 : img * 3 // 4, img // 4 : img * 3 // 4] = 1.0
    margin = torch.randn(b, 4)
    return feat, mask, margin


def test_head_forward_shapes_and_normalization():
    torch.manual_seed(0)
    head = ExtensionHead(in_channels=IN_CH).eval()
    out = head(*_head_inputs())
    assert out["ete_logits"].shape == (2, NUM_ETE_CLASSES)
    assert out["ete_probs"].shape == (2, NUM_ETE_CLASSES)
    assert out["band_ratio"].shape == (2, 1)
    assert torch.allclose(out["ete_probs"].sum(-1), torch.ones(2), atol=1e-6)
    assert (out["ete_probs"] >= 0).all()


def test_head_does_not_backprop_into_cbm_margin():
    """🔴 ETE 梯度不得重塑 CBM「边缘」概念 —— 那是交付的可解释性模块。

    三条断言缺一不可，前两条单独看都是**假门**：
      A) `margin.grad is None` —— 但「margin 压根没进 concat」同样给 None。
      B) `feat.grad` 非零 —— 证明头确实接上了梯度，A 才不是空转。
      C) **改变 margin 必须改变输出** —— 这条才排除"margin 被整个丢弃"。
    只写 A+B 时，若有人重构时把 `margin_logits` 从 `torch.cat` 里删掉，测试照样全绿
    （审查实测坐实了这个盲区）。
    """
    head = ExtensionHead(in_channels=IN_CH)
    feat, mask, margin = _head_inputs()
    feat.requires_grad_(True)
    margin.requires_grad_(True)
    head(feat, mask, margin)["ete_logits"].sum().backward()
    assert margin.grad is None, "CBM margin 收到了 ETE 的梯度 —— detach 丢了"
    assert feat.grad is not None and feat.grad.abs().sum() > 0, (
        "stage4_spatial 没有梯度 —— 头压根没接上，上一条断言失去意义"
    )
    # C) margin 确实参与前向：换一组 margin，logits 必须变
    head.eval()
    with torch.no_grad():
        f, m, mg = _head_inputs()
        a = head(f, m, mg)["ete_logits"]
        b = head(f, m, mg + 5.0)["ete_logits"]
    assert not torch.allclose(a, b, atol=1e-5), (
        "改变 margin 不改变输出 —— margin 没进 concat，detach 断言只是空转"
    )


def test_head_does_not_backprop_into_soft_mask():
    """soft_mask 通路也必须无梯度（双保险：outputs 已 detach + 头内再 detach）。"""
    head = ExtensionHead(in_channels=IN_CH)
    feat, mask, margin = _head_inputs()
    mask.requires_grad_(True)
    head(feat, mask, margin)["ete_logits"].sum().backward()
    assert mask.grad is None, "soft_mask 收到梯度 —— 会污染 seg_backbone（破坏 gate8 前提）"


def test_head_survives_empty_mask_without_nan():
    """阴性病例：带全 0 → 加权池化分母 0。必须靠 eps 兜住，不得 NaN。"""
    head = ExtensionHead(in_channels=IN_CH).eval()
    feat, _, margin = _head_inputs()
    out = head(feat, torch.zeros(2, 1, 64, 64), margin)
    assert torch.isfinite(out["ete_logits"]).all(), "空 mask 产出 NaN/Inf"
    assert float(out["band_ratio"].abs().max()) == 0.0, "空 mask 的 band_ratio 应为 0"


def test_band_ratio_tracks_nodule_size():
    """band_ratio 必须随结节增大而增大 —— 它同时充当尺度代理与"局部特征可信度"。

    ⚠️ 单调性本身**不是 band 的 oracle**：原始 mask 的面积占比也单调，若实现误把
    band 换成 mask 均值，只看单调性照样通过（审查实测坐实）。故加第二条判据 ——
    **band_ratio 与 mask 面积占比的大小关系随尺寸翻转**：小结节的带比自身面积还大
    （8px 方块：band 0.03125 > mask 0.01562），大结节则远小于自身面积
    （32px：band 0.125 < mask 0.25）。这个翻转是"周长 vs 面积"的必然结果，
    mask 均值绝不可能复现，故能真正区分两种实现。
    """
    head = ExtensionHead(in_channels=IN_CH).eval()
    feat = torch.randn(1, IN_CH, 16, 16)
    margin = torch.randn(1, 4)
    ratios, mask_means = [], []
    for size in (8, 16, 32):
        m = torch.zeros(1, 1, 64, 64)
        m[:, :, 32 - size // 2 : 32 + size // 2, 32 - size // 2 : 32 + size // 2] = 1.0
        ratios.append(float(head(feat, m, margin)["band_ratio"]))
        mask_means.append(float(m.mean()))
    assert ratios[0] < ratios[1] < ratios[2], f"band_ratio 未随尺寸单调增: {ratios}"
    assert ratios[0] > mask_means[0], (
        f"小结节的带占比应**大于**自身面积占比（周长效应），实得 band={ratios[0]} "
        f"vs mask={mask_means[0]} —— band_ratio 可能被实现成了 mask 面积占比"
    )
    assert ratios[2] < mask_means[2], (
        f"大结节的带占比应**小于**自身面积占比，实得 band={ratios[2]} "
        f"vs mask={mask_means[2]} —— 同上"
    )


def test_head_rejects_wrong_shapes():
    head = ExtensionHead(in_channels=IN_CH)
    feat, mask, margin = _head_inputs()
    with pytest.raises(AssertionError, match="空间特征图"):
        head(torch.randn(2, IN_CH), mask, margin)
    with pytest.raises(AssertionError, match="通道失配"):
        head(torch.randn(2, 208, 16, 16), mask, margin)
    with pytest.raises(AssertionError, match=r"margin_logits 期望"):
        head(feat, mask, torch.randn(2, 5))
    with pytest.raises(AssertionError, match="batch 不一致"):
        head(feat, mask[:1], margin)


def test_head_backward_reaches_all_parameters():
    """端到端反传：每个参数都要拿到梯度，防某分支被意外断开。"""
    head = ExtensionHead(in_channels=IN_CH)
    feat, mask, margin = _head_inputs(b=4)
    out = head(feat, mask, margin)
    total, _ = compute_extension_loss(
        out["ete_logits"],
        torch.tensor([0, 1, 2, 3]),
        torch.ones(4),
        class_weights=class_balanced_weights(OFFICIAL_ETE_COUNTS, 0.99),
        log_prior=log_class_prior(OFFICIAL_ETE_COUNTS),
        la_tau=1.0,
    )
    total.backward()
    missing = [n for n, p in head.named_parameters() if p.grad is None]
    assert not missing, f"这些参数没收到梯度: {missing}"


# ── 6. 真模型端到端梯度隔离（QA_VP §1.1 的正规验证形态）──────────────────


def test_end_to_end_ete_loss_leaves_seg_backbone_and_cbm_untouched():
    """🔴 用**真** `MultiTaskThyroidModel` 跑一遍：ETE 损失不得污染分割主干与 CBM。

    上面那些 head 级测试只断言"输入张量没拿到梯度"，那是接口契约；QA_VP §1.1 要求的
    是**参数级**证据（`p.grad.abs().sum() < 1e-7`）。二者不能互相替代 —— 阶段 2.4 把头
    接进训练循环后，真正会崩的是参数，不是入参。

    两条隔离各有各的通路，必须分开断言：
      - `seg_backbone`：ETE 只吃 `soft_mask_detached`，梯度到不了分割主干。
        这是 `gate8_adaptor_isolation` 的同一条红线，也是 2026-08-11 Stage2 崩塌
        四因之一（物理先验通路梯度由 0 到满强度）的预防。
      - `cbm_head`：margin 已 detach，ETE 不得重塑「边缘」概念。
        ⚠️ 注意 `seg_backbone` 那条**对 margin 是否 detach 没有区分力** —— 即使
        margin 不 detach，梯度也只会流到 `cbm_head → cls_backbone → adaptor`，
        而 adaptor 早已对 `seg_features` detach，seg_backbone 仍是 0。故必须单列
        `cbm_head` 这条，它才是 margin detach 的真实后果。
      - 反证：`cls_backbone` **必须**有梯度（`stage4_spatial` 通路是设计意图），
        否则上面两条只是"整个头没接上"的空转。
    """
    from src.models.multi_task_model import MultiTaskThyroidModel

    torch.manual_seed(0)
    model = MultiTaskThyroidModel(pretrained=False, config={"data": {"img_size": 128}})
    head = ExtensionHead(in_channels=model.CLS_POOLED_DIM)

    outputs = model(torch.randn(2, 1, 128, 128), gamma_prior=0.0)
    out = head(
        outputs["stage4_spatial"], outputs["soft_mask_detached"], outputs["margin"]
    )
    total, _ = compute_extension_loss(
        out["ete_logits"], torch.tensor([0, 3]), torch.ones(2)
    )
    total.backward()

    def grad_sum(prefix):
        return sum(
            float(p.grad.abs().sum())
            for n, p in model.named_parameters()
            if n.startswith(prefix) and p.grad is not None
        )

    assert grad_sum("seg_backbone") == 0.0, (
        f"ETE 损失污染了 seg_backbone（梯度和 {grad_sum('seg_backbone')}）"
        " —— gate8 的精确零前提被破坏"
    )
    assert grad_sum("cbm_head") == 0.0, (
        f"ETE 损失污染了 cbm_head（梯度和 {grad_sum('cbm_head')}）"
        " —— margin 的 detach 丢了，CBM 概念会被 ETE 拽着漂移"
    )
    assert grad_sum("cls_backbone") > 0.0, (
        "cls_backbone 没有梯度 —— stage4_spatial 通路断了，上面两条断言失去区分力"
    )
