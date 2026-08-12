"""Phase2 三头的 **GPU / 跨平台契约探针** —— 本地 Windows+CPU 测不到的那几条。

## 为什么单独有这个脚本

三头的单测（`tests/test_{location,extension,clinical_t}_head.py`）跑在 Windows +
CPU + fp32 上，能覆盖数学与契约，但**结构性地测不到**四类上云才会炸的问题：

1. **fp16 autocast**：torch 在 CPU 上不支持 float16 autocast，而交付推理正是
   `precision="fp16"`。三头里的 `.float()` softmax、fp32 累加器、`.to(feat.dtype)`
   回转、以及临床 T 的 `log(clamp(s))`，**只有真 CUDA 上才走到真实分支**。
2. 🔴 **`F.max_pool2d` 的隐式 −inf padding 语义**：交界带的腐蚀是 `-max_pool2d(-m)`，
   整个 ETE 局部特征都押在这个语义上。CPU 与 CUDA 是两套 kernel，一致性**必须实测**。
3. **显存峰值**：平台每容器仅约 14~16G（`docs/项目现状总纲.md` §3）。
   ⚠️ 实测下来这一维**并非风险**（三头合计 **0.536M** 参数 = 2.05MB fp32、batch=16
   峰值仅百 MB 级），本项保留的价值是"换 backbone / 提分辨率后能立刻重测"，
   不是当前有隐患。
4. **BatchNorm 在 batch=1 下的推理形态**：交付推理 batch 恒 1。

## 用法

    python scripts/probe_heads_cuda.py                # 默认 batch 4
    python scripts/probe_heads_cuda.py --batch 16     # 训练批量下的显存

**上云后第一件事就是在官方容器里跑一遍本脚本**（连同 `nvidia-smi` 记录真实
显存 total 回填总纲 §3）。

退出码有三档，**不要只看"没报错"**：
  - `0` = 全部通过且**没有任何 SKIP**（GPU 项真的验过了）
  - `2` = 有 SKIP（多半是没看到 CUDA）。这不是失败，但**绝不能当成"已验证"** ——
    fp16/显存/跨设备一致性三类恰恰是本脚本存在的理由，跳过它们等于没跑。
  - `1` = 有 FAIL

## 📌 关于 rules.md §6 的 300 行红线：本文件是**有意保留的例外**

本文件超 300 行且**不拆**，按 rules §6 的「拆分前置检查」三问逐条判定（三问全中）：
① 拆分**降低可分发性** —— 本脚本是**上云现场跑的工具**，而云平台上传要过人工审核、
每次限约 10 个文件（`memory/cloud-upload-review-constraint.md`），拆成 `probes/` 包会从
1 个文件变 3 个；且现在任何工作目录下 `python scripts/probe_heads_cuda.py` 直接可跑，
拆包后要依赖 `scripts/__init__` → `scripts/probes/__init__` 的 import 链，陌生容器里多一个
失败点。② 切出来的两半仍是同一件事（验三头 GPU 契约），**不产生认知边界**。
③ 为守行数压缩注释会撞 §5「细节优先」—— 上一轮真这么干过，是被规则记录在案的反例。

本脚本只有一个职责，行数随头数线性增长（每加一个头约 20 行），可读性不随之下降。
"""

import argparse
import sys

import torch

# Windows 控制台默认 GBK，本文件的输出含 ↔ 等非 GBK 字符会直接 UnicodeEncodeError
# （与 2026-08-12 修掉的 `scripts/verify/runner.py` GBK 崩溃同一类问题）。
# 在任何打印之前先把标准流切到 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # 非 TextIO（被重定向/被包装）时跳过
        pass

# 允许从仓库根直接 `python scripts/probe_heads_cuda.py`
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.models.heads.clinical_t import ClinicalTHead  # noqa: E402
from src.models.heads.extension import ExtensionHead  # noqa: E402
from src.models.heads.location import LocationHead  # noqa: E402
from src.models.mask_features import (  # noqa: E402
    boundary_band_from_mask,
    nodule_scale_from_mask,
)

IN_CH = 352  # MultiTaskThyroidModel.CLS_POOLED_DIM
IMG = 512
FEAT_HW = IMG // 32

_results = []


def record(name, status, detail=""):
    _results.append((name, status, detail))
    icon = {"PASS": "[OK]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
    print(f"{icon} {name}" + (f" — {detail}" if detail else ""), flush=True)


def check(name, cond, detail=""):
    record(name, "PASS" if cond else "FAIL", detail)
    return bool(cond)


def _make_inputs(batch, device, dtype=torch.float32):
    torch.manual_seed(0)
    feat = torch.randn(batch, IN_CH, FEAT_HW, FEAT_HW, device=device, dtype=dtype)
    mask = torch.zeros(batch, 1, IMG, IMG, device=device, dtype=dtype)
    mask[:, :, 180:330, 180:330] = 1.0  # 150px 结节
    margin = torch.randn(batch, 4, device=device, dtype=dtype)
    return feat, mask, margin


# ── 1. 形态学 kernel 的跨设备一致性 ────────────────────────────────────────


def probe_band_cpu_gpu_equivalence(device):
    """交界带在 CPU 与 CUDA 上必须**逐元素相同**。

    带是 0/1 阶跃的形态学结果，任何 padding 语义差异都表现为整条边界带的有无，
    而不是末位误差，故用严格相等而非 allclose。
    """
    if device.type != "cuda":
        record("band CPU↔CUDA 逐元素一致", "SKIP", "无 CUDA")
        return
    m_cpu = torch.zeros(1, 1, 64, 64)
    m_cpu[:, :, 10:40, 20:50] = 1.0
    m_cpu[:, :, 0:8, 0:8] = 1.0  # 贴边块：专门压 −inf padding 那条语义
    b_cpu = boundary_band_from_mask(m_cpu, 5)
    b_gpu = boundary_band_from_mask(m_cpu.to(device), 5).cpu()
    check(
        "band CPU↔CUDA 逐元素一致",
        torch.equal(b_cpu, b_gpu),
        f"最大差 {float((b_cpu - b_gpu).abs().max()):.3e}",
    )
    # 贴边语义本身也在 GPU 上复核一次（本地测试只验过 CPU）
    check(
        "band 贴边侧无带(−inf padding 语义, GPU)",
        float(b_gpu[0, 0, 0, 0]) == 0.0 and float(b_gpu[0, 0, 7, 4]) > 0.0,
        f"角点={float(b_gpu[0, 0, 0, 0])}, 内侧边={float(b_gpu[0, 0, 7, 4])}",
    )


# ── 2. fp16 autocast（交付推理的真实路径）────────────────────────────────


def probe_fp16_autocast(device, batch):
    """整头在真 fp16 autocast 下前向：不得 NaN/Inf，概率仍须和为 1。CPU 验不了这条。"""
    if device.type != "cuda":
        record("fp16 autocast 前向", "SKIP", "CPU 不支持 float16 autocast")
        return
    ext = ExtensionHead(in_channels=IN_CH).to(device).eval()
    loc = LocationHead(in_channels=IN_CH).to(device).eval()
    clt = ClinicalTHead(in_channels=IN_CH).to(device).eval()
    feat, mask, margin = _make_inputs(batch, device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        eo = ext(feat, mask, margin)
        lo = loc(feat)
        to = clt(feat, nodule_scale_from_mask(mask), eo["ete_logits"])
        # 阴性病例（空 mask）→ scale 恰为 0 → `log(clamp(s,1e-4))`。fp16 下 1e-4 落在
        # 非规格化边缘（最小正规数 6.1e-5），是 CPU 测不到的真实分支。
        to_empty = clt(feat, nodule_scale_from_mask(torch.zeros_like(mask)),
                       eo["ete_logits"])
    ok = (
        torch.isfinite(eo["ete_logits"]).all()
        and torch.isfinite(eo["ete_probs"]).all()
        and torch.isfinite(lo["location_probs"]).all()
        and torch.isfinite(to["t_logits"]).all()
    )
    check("fp16 autocast 前向无 NaN/Inf", bool(ok))
    check(
        "fp16 下空 mask 的 log(clamp) 不产生 -inf/NaN",
        bool(torch.isfinite(to_empty["t_logits"]).all()),
        f"t_logits[0]={[round(float(v), 3) for v in to_empty['t_logits'][0]]}",
    )
    check(
        "fp16 下 ETE 概率和为 1",
        torch.allclose(eo["ete_probs"].sum(-1).float(), torch.ones(batch, device=device), atol=1e-3),
        f"和={[round(float(v), 5) for v in eo['ete_probs'].sum(-1)[:2]]}",
    )
    check(
        "fp16 下位置概率和为 1",
        torch.allclose(lo["location_probs"].sum(-1).float(), torch.ones(batch, device=device), atol=1e-3),
    )
    check(
        "fp16 下 T 概率和为 1 且 t_conf==P(T=1)",
        torch.allclose(to["t_probs"].sum(-1).float(), torch.ones(batch, device=device), atol=1e-3)
        and torch.equal(to["t_conf"], to["t_probs"][:, 1]),
    )
    check(
        "fp16 概率输出被提升到 fp32",
        eo["ete_probs"].dtype == torch.float32
        and lo["location_probs"].dtype == torch.float32
        and to["t_probs"].dtype == torch.float32,
        f"ete={eo['ete_probs'].dtype}, loc={lo['location_probs'].dtype}, t={to['t_probs'].dtype}",
    )


def probe_fp16_sum_overflow(device):
    """fp16 下对 512² 求和的溢出防护是否**真的**在 GPU 上生效。

    区分力：同一张 fp16 图，朴素 `.sum()` 应溢出成 inf，而带 fp32 累加器的
    `nodule_scale_from_mask` 必须给出有限值。若哪天 GPU 上朴素求和不再溢出，
    本项会 FAIL 提示"防护的前提变了、去复议"，而不是默默变成永远通过的假门。
    """
    m = torch.full((1, 1, IMG, IMG), 0.5, dtype=torch.float16, device=device)
    naive = float(m.sum())
    guarded = nodule_scale_from_mask(m)
    check(
        "fp16 朴素求和确实溢出(反证前提成立)",
        not torch.isfinite(torch.tensor(naive)),
        f"naive sum={naive}",
    )
    check(
        "nodule_scale fp32 累加器有限且正确",
        bool(torch.isfinite(guarded).all()) and abs(float(guarded) - 0.5 ** 0.5) < 1e-3,
        f"scale={float(guarded):.5f} (期望 sqrt(0.5)={0.5 ** 0.5:.5f})",
    )


# ── 3. batch=1 推理形态 ────────────────────────────────────────────────────


def probe_batch_one_inference(device):
    """交付推理 batch 恒 1：BN 在 eval 下必须能跑，且不同结节给出不同结果。"""
    ext = ExtensionHead(in_channels=IN_CH).to(device).eval()
    feat, mask, margin = _make_inputs(1, device)
    with torch.no_grad():
        out1 = ext(feat, mask, margin)
        small = torch.zeros_like(mask)
        small[:, :, 250:262, 250:262] = 1.0
        out2 = ext(feat, small, margin)
    check("batch=1 推理不崩且输出有限", bool(torch.isfinite(out1["ete_logits"]).all()))
    check(
        "不同结节尺度 → 不同 band_ratio(输入真的进了头)",
        float(out1["band_ratio"]) > float(out2["band_ratio"]) > 0.0,
        f"150px={float(out1['band_ratio']):.5f} vs 12px={float(out2['band_ratio']):.5f}",
    )
    # 临床 T 在 **train 模式 + batch=1**：若有人把 BatchNorm1d 加回 MLP，这里当场
    # ValueError（"Expected more than 1 value per channel"）。训练末批可能只剩 1 样本，
    # 这不是假想。CPU 上单测已覆盖，此处确认 CUDA kernel 同样不引入 batch 依赖。
    clt = ClinicalTHead(in_channels=IN_CH).to(device).train()
    to = clt(feat, nodule_scale_from_mask(mask), torch.randn(1, 4, device=device))
    check(
        "T 头 train 模式 batch=1 不崩(BN1d 未被加回)",
        bool(torch.isfinite(to["t_logits"]).all()),
        f"t_conf={float(to['t_conf']):.5f}",
    )


# ── 4. 显存 ────────────────────────────────────────────────────────────────


def probe_memory(device, batch):
    """三头前向+反传的显存峰值。平台每容器仅约 14G，A800 的 80G 不 OOM 说明不了什么。"""
    if device.type != "cuda":
        record("显存峰值实测", "SKIP", "无 CUDA")
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ext = ExtensionHead(in_channels=IN_CH).to(device)
    loc = LocationHead(in_channels=IN_CH).to(device)
    clt = ClinicalTHead(in_channels=IN_CH).to(device)
    feat, mask, margin = _make_inputs(batch, device)
    feat.requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        eo = ext(feat, mask, margin)
        out = (
            eo["ete_logits"].sum()
            + loc(feat)["side_logits"].sum()
            + clt(feat, nodule_scale_from_mask(mask), eo["ete_logits"])["t_logits"].sum()
        )
    out.backward()
    peak_alloc = torch.cuda.max_memory_allocated() / 2**20
    peak_res = torch.cuda.max_memory_reserved() / 2**20
    total = torch.cuda.get_device_properties(device).total_memory / 2**20
    record(
        "显存峰值实测",
        "PASS",
        f"batch={batch} allocated={peak_alloc:.1f}MB reserved={peak_res:.1f}MB "
        f"/ 卡上 total={total:.0f}MB（平台容器仅约 14336MB，按 reserved 折算）",
    )
    n = {k: sum(p.numel() for p in h.parameters())
         for k, h in (("ext", ext), ("loc", loc), ("clin_t", clt))}
    record(
        "三头参数量",
        "PASS",
        f"ExtensionHead={n['ext'] / 1e6:.3f}M, LocationHead={n['loc'] / 1e6:.3f}M, "
        f"ClinicalTHead={n['clin_t'] / 1e6:.3f}M "
        f"（合计 {sum(n.values()) * 4 / 2**20:.2f}MB fp32）",
    )


# ── 5. 梯度隔离（真设备上再确认一次）──────────────────────────────────────


def probe_gradient_isolation(device):
    """三头的梯度不得到达 mask / CBM margin / 上游头 —— 换设备不该改变这些契约。"""
    ext = ExtensionHead(in_channels=IN_CH).to(device)
    feat, mask, margin = _make_inputs(2, device)
    feat.requires_grad_(True)
    mask.requires_grad_(True)
    margin.requires_grad_(True)
    ext(feat, mask, margin)["ete_logits"].sum().backward()
    check("ETE 梯度不回流 soft_mask", mask.grad is None)
    check("ETE 梯度不回流 CBM margin", margin.grad is None)
    check(
        "ETE 梯度确实到达 stage4_spatial(反证上两条非空转)",
        feat.grad is not None and float(feat.grad.abs().sum()) > 0,
    )

    # 临床 T：ete_logits 与 nodule_scale 都必须 detach（ETE 是官方独立计分维度，
    # 不能被 T 的监督拽偏；scale 那条则是分割主干隔离）。
    clt = ClinicalTHead(in_channels=IN_CH).to(device)
    feat_t = torch.randn(2, IN_CH, FEAT_HW, FEAT_HW, device=device, requires_grad=True)
    scale_t = torch.rand(2, 1, device=device, requires_grad=True)
    ete_t = torch.randn(2, 4, device=device, requires_grad=True)
    clt(feat_t, scale_t, ete_t)["t_logits"].sum().backward()
    check("T 梯度不回流 ETE logits", ete_t.grad is None)
    check("T 梯度不回流 nodule_scale", scale_t.grad is None)
    check(
        "T 梯度确实到达 stage4_spatial(反证上两条非空转)",
        feat_t.grad is not None and float(feat_t.grad.abs().sum()) > 0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    # 🔴 不能只看 `is_available()`。实测 `CUDA_VISIBLE_DEVICES=""` 时它仍可能返回
    # True，而 `get_device_name(0)` 当场 `AssertionError: Invalid device id` ——
    # 这恰恰是"云端容器没把 GPU 透传进来"的典型形态，本脚本必须优雅降级成 SKIP
    # 并给出退出码 2，而不是崩在第一行让人以为是脚本坏了。
    has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if has_cuda else "cpu")
    print(f"torch={torch.__version__} device={device} "
          f"is_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
    if has_cuda:
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception as exc:  # 驱动/可见性异常 → 退回 CPU，别让探针本身成为故障点
            print(f"⚠️ 查询 GPU 名失败({exc})，按无 CUDA 处理")
            device = torch.device("cpu")
    print("-" * 78, flush=True)

    probe_band_cpu_gpu_equivalence(device)
    probe_fp16_sum_overflow(device)
    probe_fp16_autocast(device, args.batch)
    probe_batch_one_inference(device)
    probe_gradient_isolation(device)
    probe_memory(device, args.batch)

    print("-" * 78)
    n_fail = sum(1 for _, s, _ in _results if s == "FAIL")
    n_skip = sum(1 for _, s, _ in _results if s == "SKIP")
    print(f"总计 {len(_results)} 项: {len(_results) - n_fail - n_skip} PASS / "
          f"{n_fail} FAIL / {n_skip} SKIP")
    if n_fail:
        print("失败项:")
        for name, status, detail in _results:
            if status == "FAIL":
                print(f"  - {name}: {detail}")
        return 1
    if n_skip:
        # 🔴 SKIP 不得静默计为通过。本脚本存在的全部理由就是验 CPU 验不到的东西
        # （fp16 autocast / 跨设备形态学一致性 / 显存），这些恰好都是无 CUDA 时被
        # 跳过的项 —— 若此时返回 0，上云看到"exit 0"会误以为已验证。
        print("\n⚠️ 有 SKIP 项，本次运行**不构成 GPU 验证**：")
        for name, status, detail in _results:
            if status == "SKIP":
                print(f"  - {name}: {detail}")
        print("  → 退出码 2（非失败，但也非通过）。上云请在有 GPU 的容器内复跑。")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
