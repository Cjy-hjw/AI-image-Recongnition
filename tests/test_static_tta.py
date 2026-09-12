# -*- coding: utf-8 -*-
"""静态 hflip TTA(D2,2026-09-12)——只平均分割概率,分类各头不融合。

设计契约(见 case_pipeline.diagnose_static_case 的 TTA 段注释):
1. 🔴 分类/位置/ETE/T/TI-RADS 概率必须与无 TTA 时**逐位一致**(读原图 outputs);
2. 🔴 位置头绝不能 flip(左右叶语义互换);
3. mask 概率 = 原图 sigmoid 概率与「翻转前向→逆 flip」概率的逐像素平均;
4. TTA 关(env AICOMP_STATIC_TTA=0)或翻转前向 RuntimeError → 逐位回退单前向;
5. mask_prob_override 含 NaN → decode_predictions 照常硬拦
   (NonFiniteProbabilityError,不静默产出全零 mask)。

fixture 采用「确定性的空间不对称响应」构造法(镜像测试须验区分力,memory
mirror-tests-need-distinguishing-power):模型对图像**每个像素独立**输出
`sigmoid(a·x + b·(1-x) + c)`——x 是归一化列坐标,左右不对称,翻转前向概率
≠ 原图概率,TTA 平均因此**可解析验证**;而全图的分类头输出恒定,可钉死
「分类不融合」契约。
"""
import os

import numpy as np
import pytest
import torch

from src.inference.case_pipeline import (
    _static_tta_enabled,
    diagnose_static_case,
)
from src.inference.postprocess import decode_predictions


# ---------------- fixture:确定性的空间不对称假模型 ----------------


class _AsymMaskModel(torch.nn.Module):
    """逐像素线性于归一化列坐标的假分割 + 位置敏感的分类头。

    forward(img[B,1,H,W], gamma_prior) → dict,键集覆盖 decode_predictions
    全部消费项。
    - 分割 logits = a·x_norm + b·(1-x_norm) + c(左右不对称,a/b/c 见 LOGITS)
      ——翻转输入后同一位置 logits 不同,TTA 平均 ≠ 单前向,测试有区分力;
    - 分类 logits 恒定基座 + **逐类异号的翻转敏感项** d·sign_per_class:
      d = 左右半区均值差(hflip 下 d→-d),sign 逐类固定。🔴 这是让"契约1
      (分类禁止融合)"测试有区分力的唯一构造(修复审查 F2):若实现把两路
      分类概率平均,(softmax(v+δs)+softmax(v-δs))/2 与 softmax(v) 的差
      达 0.085(实测),远超 allclose 容差——恒定 logits 或全局同加 δ 的
      fixture 对融合零区分力(softmax 平移不变,差被 float32 精度淹没)。
    """

    LOGITS = {"a": 4.0, "b": -2.0, "c": 0.5, "sens": 1.5}
    # 逐类异号敏感系数(长度=各类别数,与 _head_probs 契约一致)
    _SIGNS = {"ete": [1.0, -1.0, 0.5, -0.5],
              "location": [1.0, -1.0, 0.5, -0.5, 1.0, -1.0, 0.5],
              "t": [1.0, -1.0], "tirads": [1.0, -1.0, 0.5, -0.5, 1.0],
              "comp": [1.0, -1.0, 0.5, -0.5], "echo": [1.0, -1.0, 0.5, -0.5],
              "shape": [1.0, -1.0], "margin": [1.0, -1.0, 0.5, -0.5],
              "foci": [1.0, -1.0, 0.5, -0.5], "mal": [1.0, -1.0]}

    def _half_diff(self, img):
        """左/右半区均值差——hflip 下变号,作为翻转敏感标量。"""
        w = img.shape[-1]
        return img[..., :w // 2].mean() - img[..., w // 2:].mean()

    def forward(self, img, gamma_prior=1.0):  # noqa: ARG002
        bsz, _, h, w = img.shape
        x = torch.linspace(0.0, 1.0, w, device=img.device).view(1, 1, 1, w)
        p = self.LOGITS
        seg_logits = (p["a"] * x + p["b"] * (1.0 - x) + p["c"])
        seg_logits = seg_logits.expand(bsz, 1, h, w)

        d = self._half_diff(img) * p["sens"]

        def _sens(n, base, key):
            signs = torch.tensor(self._SIGNS[key], device=img.device,
                                 dtype=torch.float32)
            return torch.full((bsz, n), float(base), device=img.device) + d * signs

        return {
            "mask": seg_logits,
            "composition": _sens(4, 1.0, "comp"),
            "echogenicity": _sens(4, 0.5, "echo"),
            "shape": _sens(2, -0.3, "shape"),
            "margin": _sens(4, 0.2, "margin"),
            "foci": _sens(4, -1.0, "foci"),
            "malignancy": _sens(2, 0.7, "mal"),
            # 官方四头:模型内已 softmax 的概率(decode 直收,不二次 softmax)
            "ete_probs": torch.softmax(_sens(4, 0.3, "ete"), dim=1),
            "location_probs": torch.softmax(_sens(7, 0.9, "location"), dim=1),
            "t_probs": torch.softmax(_sens(2, -0.4, "t"), dim=1),
            "tirads_probs": torch.softmax(_sens(5, 0.1, "tirads"), dim=1),
            "t_conf": torch.sigmoid(
                torch.full((bsz,), 0.5, device=img.device) + d),
        }


CONFIG = {"data": {"img_size": 32}}
DEVICE = torch.device("cpu")


def _run(img_np, tta_env=None):
    """跑 diagnose_static_case,可控 TTA env。返回 preds dict。"""
    old = os.environ.pop("AICOMP_STATIC_TTA", None)
    if tta_env is not None:
        os.environ["AICOMP_STATIC_TTA"] = tta_env
    try:
        model = _AsymMaskModel().to(DEVICE).eval()
        out = diagnose_static_case(
            model=model, img_np=img_np, config=CONFIG,
            calibrator=None, precision="fp32", device=DEVICE)
        return out["preds"]
    finally:
        if tta_env is not None:
            os.environ.pop("AICOMP_STATIC_TTA", None)
        if old is not None:
            os.environ["AICOMP_STATIC_TTA"] = old


IMG = np.tile(np.linspace(0, 255, 30, dtype=np.uint8), (40, 1))   # H≠W 且左右不对称:
# 🔴 必须左右不对称(修复审查 F2 实证:常量图 d=0 → 敏感项恒 0,契约1 零区分力;
# 渐变图 d=-0.52,逐类异号敏感项被激活,融合变异可测)


# ---------------- 开关解析 ----------------


@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("", True), ("true", True), ("TRUE", True), ("garbage", True),
    ("0", False), ("false", False), ("off", False), ("no", False),
])
def test_tta_env_switch(monkeypatch, raw, expected):
    monkeypatch.setenv("AICOMP_STATIC_TTA", raw)
    assert _static_tta_enabled() is expected
    monkeypatch.delenv("AICOMP_STATIC_TTA")
    assert _static_tta_enabled() is True   # 缺省=开


# ---------------- 契约 1&2:分类各头与无 TTA 逐位一致 ----------------


_CLASS_KEYS = ["comp_idx", "comp_prob", "echo_prob", "shape_prob", "margin_prob",
               "foci_prob", "mal_prob", "mal_probs",
               "ete_probs", "location_probs", "t_probs", "t_conf", "tirads_probs"]


def test_classification_heads_identical_with_and_without_tta():
    off = _run(IMG, tta_env="0")
    on = _run(IMG, tta_env="1")
    for k in _CLASS_KEYS:
        a, b = off[k], on[k]
        assert np.allclose(a, b, atol=0, rtol=0), (
            f"TTA 改变了分类输出 {k}:无TTA={a} 有TTA={b} —— 分类各头禁止融合")


# ---------------- 契约 3:mask = 两路 sigmoid 概率的逐像素平均 ----------------


def test_mask_is_pixelwise_average_of_sigmoid_probs():
    p = _AsymMaskModel.LOGITS
    w = CONFIG["data"]["img_size"]
    x = np.linspace(0.0, 1.0, w, dtype=np.float64)
    # 假模型 logits 只随列坐标变化;翻转图的列坐标 x' = 1-x(连续 linspace 近似)。
    # sigmoid 解析式:
    s = 1.0 / (1.0 + np.exp(-(p["a"] * x + p["b"] * (1 - x) + p["c"])))
    s_flip = 1.0 / (1.0 + np.exp(-(p["a"] * (1 - x) + p["b"] * x + p["c"])))
    expected = (s + s_flip) / 2.0
    got = _run(IMG, tta_env="1")["pred_mask"]
    assert got.shape == (CONFIG["data"]["img_size"],) * 2
    # 每行应等于 expected(linspace 逆序与 1-x 在采样点上有 O(1/w) 离散差,
    # 容差放 5e-2)
    assert np.allclose(got, expected[None, :], atol=5e-2), (
        f"TTA mask 与解析期望不符:max|diff|="
        f"{np.abs(got - expected[None, :]).max():.4f}")


def test_tta_changes_mask_vs_no_tta():
    """镜像测试区分力:TTA mask 必须与单前向 mask **不同**(fixture 不对称)。
    若二者逐位相同,说明 fixture 退化(对称响应)或 TTA 实际没生效。"""
    off = _run(IMG, tta_env="0")["pred_mask"]
    on = _run(IMG, tta_env="1")["pred_mask"]
    assert not np.allclose(off, on), "TTA 未改变 mask——fixture 无区分力或 TTA 未生效"


# ---------------- 契约 4:翻转前向异常的分流(OOM 回退/非 OOM 冒泡) ----------------


class _FlipOOMModel(_AsymMaskModel):
    """对**翻转后的输入**(用预张量指纹识别)抛 OOM,原图正常——精确模拟 TTA 故障面。

    指纹:原始输入是 resize 后的渐变图,首列像素恒 0;hflip 后首列像素恒 255。
    判据「首列均值 > 0.5」只在翻转副本上为真(渐变图左右不均),不会误伤原图。
    """

    def forward(self, img, gamma_prior=1.0):
        if img[..., 0].mean() > 0.5:   # 首列亮 = 已被 hflip 的副本
            raise RuntimeError("CUDA out of memory (fake)")
        return super().forward(img, gamma_prior)


def test_flip_forward_failure_falls_back_to_single_forward():
    """翻转前向 OOM → warning + 回退,mask 与无 TTA 单前向逐位一致,不炸。"""
    grad = np.tile(np.linspace(0, 255, 30, dtype=np.uint8), (40, 1))
    old = os.environ.pop("AICOMP_STATIC_TTA", None)
    os.environ["AICOMP_STATIC_TTA"] = "1"
    try:
        model = _FlipOOMModel().to(DEVICE).eval()
        out = diagnose_static_case(model=model, img_np=grad, config=CONFIG,
                                   calibrator=None, precision="fp32",
                                   device=DEVICE)
        preds = out["preds"]
    finally:
        os.environ.pop("AICOMP_STATIC_TTA", None)
        if old is not None:
            os.environ["AICOMP_STATIC_TTA"] = old
    off = _run(grad, tta_env="0")
    assert np.array_equal(preds["pred_mask"], off["pred_mask"]), (
        "翻转前向失败后应逐位回退单前向 mask")


class _FlipRealBugModel(_AsymMaskModel):
    """翻转副本上抛**非 OOM** RuntimeError(weight shape mismatch 类真故障)。

    Gate2 审查 F2:except 收窄到 OOM 后,真故障必须冒泡,不得静默劣化成
    "无 TTA 照常跑"——否则整批半 TTA 且对拍判据失真。
    """

    def forward(self, img, gamma_prior=1.0):
        if img[..., 0].mean() > 0.5:
            raise RuntimeError("Error(s) in loading state_dict: size mismatch")
        return super().forward(img, gamma_prior)


def test_flip_forward_real_bug_must_propagate():
    """非 OOM 的翻转前向 RuntimeError 必须冒泡(整例 ERROR),不得静默回退。"""
    grad = np.tile(np.linspace(0, 255, 30, dtype=np.uint8), (40, 1))
    old = os.environ.pop("AICOMP_STATIC_TTA", None)
    os.environ["AICOMP_STATIC_TTA"] = "1"
    try:
        with pytest.raises(RuntimeError, match="size mismatch"):
            diagnose_static_case(model=_FlipRealBugModel().to(DEVICE).eval(),
                                 img_np=grad, config=CONFIG, calibrator=None,
                                 precision="fp32", device=DEVICE)
    finally:
        os.environ.pop("AICOMP_STATIC_TTA", None)
        if old is not None:
            os.environ["AICOMP_STATIC_TTA"] = old


# ---------------- 契约 5:override 含 NaN → 硬拦 / 形状非法 → 硬拦 ----------------


def test_nan_override_raises():
    outputs = _AsymMaskModel().forward(torch.zeros(1, 1, 8, 8))
    with pytest.raises(Exception) as ei:
        decode_predictions(outputs, mask_prob_override=torch.full((8, 8), float("nan")))
    assert "NaN" in str(ei.value) or "nan" in str(ei.value).lower()


@pytest.mark.parametrize("shape", [(1, 1, 8, 8), (4, 8, 8), (8,), (8, 8, 8, 8)])
def test_bad_shape_override_raises(shape):
    """Gate2 审查 F3:3D/4D override 在下游 cv2.resize 会被当多通道图静默放行
    或直接 cv2.error——与 t_probs 长度校验同一风险面,源头硬拦。"""
    outputs = _AsymMaskModel().forward(torch.zeros(1, 1, 8, 8))
    with pytest.raises(ValueError, match="维度契约"):
        decode_predictions(outputs, mask_prob_override=torch.rand(shape))


# ---------------- 默认路径回归:override=None 逐位等价旧版 ----------------


def test_override_none_matches_legacy_sigmoid():
    outputs = _AsymMaskModel().forward(torch.zeros(1, 1, 8, 8))
    legacy = torch.sigmoid(outputs["mask"].float())[0, 0].numpy()
    got = decode_predictions(outputs)["pred_mask"]
    assert np.array_equal(got, legacy)
