# -*- coding: utf-8 -*-
"""静态 2D 形态学清理契约测试(2026-09-12,Fable 低成本点)。

nodule_split._split_2d 的 AICOMP_CLEAN_2D 开关分支锁定:
1. 🔴 默认关:env 未设时输出与旧版(无清理)**逐位一致**——这是离线对拍前的
   安全态,没验证双升之前不改变任何生产行为;
2. 开("1")时:1px 孔洞被填(环形 mask → 实心)、细缝闭合(断开的两半 → 一个实例);
3. 开时不误伤:已经实心的 mask 清理后逐位不变(闭运算+填孔对实心凸区域是恒等);
4. 边界 flood 起点是前景时跳过填孔(宁少填不误扩);
5. 垃圾 env 值("true"/"on"/"")= 关(回安全默认)。
"""
import os
import sys

import numpy as np
import pytest

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.nodule_split import (  # noqa: E402
    _clean_2d_enabled,
    split_nodules,
)


def _ring(h=64, w=64, cy=32, cx=32, r_out=10, r_in=5):
    """环形前景(中间是孔):清理应把孔填上。"""
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return ((d2 <= r_out ** 2) & (d2 >= r_in ** 2)).astype(np.uint8) * 255


def _solid(h=64, w=64, cy=32, cx=32, r=10):
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return (d2 <= r ** 2).astype(np.uint8) * 255


def _cracked(h=64, w=64):
    """直径 20 的圆,竖缝从 y=22 到 y=43 穿透圆(上下都切穿)。"""
    m = _solid(h, w)
    m[22:43, 32] = 0     # 竖缝全穿圆心
    return m


def _run(mask, clean):
    old = os.environ.pop("AICOMP_CLEAN_2D", None)
    if clean:
        os.environ["AICOMP_CLEAN_2D"] = "1"
    try:
        return split_nodules(mask)
    finally:
        os.environ.pop("AICOMP_CLEAN_2D", None)
        if old is not None:
            os.environ["AICOMP_CLEAN_2D"] = old


@pytest.mark.parametrize("raw,expected", [
    ("1", True), (" 1 ", True),
    ("0", False), ("true", False), ("on", False), ("", False), ("garbage", False),
])
def test_clean_env_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("AICOMP_CLEAN_2D", raw)
    assert _clean_2d_enabled() is expected
    monkeypatch.delenv("AICOMP_CLEAN_2D")
    assert _clean_2d_enabled() is False   # 缺省=关


def test_default_off_is_bitwise_identical():
    """🔴 默认关 = 逐位等价旧版(安全态,对拍前的红线)。"""
    ring = _ring()
    off = _run(ring, clean=False)
    # 旧版行为:环形是一个连通域(孔不拆连通性),面积 = 环面积
    assert len(off) == 1
    import math
    expect_area = int(np.count_nonzero(_ring()))
    assert off[0]["area"] == expect_area


def test_clean_fills_holes():
    """开清理:环形 → 实心(孔被填),面积 = 外圆面积。"""
    ring = _ring()
    on = _run(ring, clean=True)
    assert len(on) == 1
    expect_filled = int(np.count_nonzero(_solid()))
    assert on[0]["area"] == expect_filled
    # mask 内部不再有 0(实心)
    assert on[0]["mask"].sum() == expect_filled


def test_clean_closes_thin_crack():
    """开清理:1px 穿透竖缝被闭运算缝合——关时圆被切成两个实例(实测 148+148),
    开时合回一个。缝合后面积 = 裂缝圆(296) + 闭运算补的缝像素,略小于实心圆
    (317,圆边界处缝像素补不满)——断言用"单实例 + 面积介于两者"而非精确值。"""
    m = _cracked()
    off, on = _run(m, False), _run(m, True)
    assert len(off) == 2, "关时穿透缝应把圆切成两半(fixture 前提)"
    assert len(on) == 1, "开时闭运算应缝合穿透缝"
    cracked_px = int(np.count_nonzero(m))
    solid_px = int(np.count_nonzero(_solid()))
    assert cracked_px <= on[0]["area"] <= solid_px, (
        f"缝合后面积 {on[0]['area']} 应在 [{cracked_px}, {solid_px}] 内")


def test_clean_fills_holes_when_corner_foreground():
    """🔴 Gate2 P1 回归:前景竖带贯通上下边时,(0,0) 与右侧背景不连通,
    旧版单种子实现把整块右侧背景当"孔洞"灌成前景(实测单实例 53% 图幅;
    左缘条带+真结节场景 98%)。四角种子修复后:竖带只填它围住的真洞,
    不吞外部背景。"""
    h, w = 64, 64
    band = np.zeros((h, w), np.uint8)
    band[:, 20:30] = 255            # 竖带贯通上下:(0,0) 左侧与右侧背景隔离
    band[28:36, 40:48] = 255        # 右侧放一个实心块(正常前景)
    os.environ["AICOMP_CLEAN_2D"] = "1"
    try:
        out = split_nodules(band)
    finally:
        os.environ.pop("AICOMP_CLEAN_2D")
    # 前景占比必须仍在 ~带+块量级(约 (10*64+64-10*8)/4096 ≈ 9%),绝不能 53%
    total_fg = sum(o["area"] for o in out)
    assert total_fg < 0.2 * h * w, f"清理吞了外部背景:前景占比 {total_fg/(h*w):.2f}"


def test_clean_flood_failover_guard():
    """🔴 Gate2 兜底:flood 静默失败(或异常大洞)导致填孔后前景>0.5 时,
    回退原始 mask 而非产出全前景。用 monkeypatch 把 floodFill 替换为
    no-op 模拟,断言结果仍是原 mask 的实例(不是整图)。"""
    import src.inference.nodule_split as ns
    ring = _ring()
    real_flood = cv2.floodFill

    def noop_flood(image, mask, seed, newval):
        return 0, image                   # 什么都不灌

    os.environ["AICOMP_CLEAN_2D"] = "1"
    try:
        orig = ns.cv2.floodFill
        ns.cv2.floodFill = noop_flood
        try:
            out = split_nodules(ring)
        finally:
            ns.cv2.floodFill = orig
    finally:
        os.environ.pop("AICOMP_CLEAN_2D")
        del real_flood
    assert len(out) == 1
    assert out[0]["area"] == int(np.count_nonzero(_ring())), (
        "flood no-op 时应整体回退原 mask(环面积),不能变全前景")


def test_clean_is_identity_on_solid():
    """开清理对已实心 mask 是恒等(不误扩)。"""
    solid = _solid()
    off, on = _run(solid, False), _run(solid, True)
    assert len(on) == len(off) == 1
    assert np.array_equal(on[0]["mask"], off[0]["mask"])
    assert on[0]["area"] == off[0]["area"]


def test_clean_skips_fill_when_corner_is_foreground():
    """单角前景不再全局跳填(四角种子修复后的新语义):(0,0) 是前景,但其余
    三角有背景 → 仍从可用角灌、环孔被填(实测 317=实心面积)。该行为比旧版
    (单角命中失败就整图跳过)更符合"填孔"语义;真整图四角全前景才跳过。"""
    ring = _ring()
    ring[0, 0] = 255      # 只有左上角是前景
    on = _run(ring, clean=True)
    assert len(on) == 2   # 填孔后的实心环(317) + 1px 角点(min_area=1 不滤)
    assert on[0]["area"] == 317, "其余三角可用 → 孔应被填(实心面积)"


def test_clean_off_env_variants_keep_legacy():
    """非 "1" 的任何值 = 旧行为(垃圾配置回安全默认,不静默采信)。"""
    ring = _ring()
    for v in ("0", "true", ""):
        old = os.environ.pop("AICOMP_CLEAN_2D", None)
        os.environ["AICOMP_CLEAN_2D"] = v
        try:
            out = split_nodules(ring)
        finally:
            os.environ.pop("AICOMP_CLEAN_2D", None)
            if old is not None:
                os.environ["AICOMP_CLEAN_2D"] = old
        assert out[0]["area"] == int(np.count_nonzero(_ring())), f"env={v!r} 应走旧路径"
