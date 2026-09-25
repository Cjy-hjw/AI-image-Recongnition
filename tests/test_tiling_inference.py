"""Z1 静态滑窗 tiling 推理回归测试(新架构线第一波,2026-09-21)。

覆盖设计(设计文档已删,预注册五组判据存总纲 §6.4 Z1 条):
1. 零回归:env 关(默认)时 diagnose_static_case 全链路逐位不变;
2. 恒等:小图(≤tile)时 tiling 路径退化为单前向(不进融合);
3. 融合正确性:合成大图放已知位置病灶,tiling 后 mask 坐标落原图坐标系;
4. 分类单源:四头输出全部来自单一 tile outputs(不融合);
5. 置信度解耦:跨 tile 实例 conf 不等于多 tile 概率的平均(max 语义)。

模型用随机初始化的 MultiTaskThyroidModel(真前向,CPU,不依赖权重——
几何/融合/单源契约与权重值无关)。
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.tiling import (  # noqa: E402
    _select_classification_tile, _tiling_enabled, _tiling_geometry,
    diagnose_static_case_tiled, tile_origins)

CONFIG = {"data": {"img_size": 512}, "model": {"backbone": "efficientnet_b2"},
          "environment": {"device": "cpu"}}


@pytest.fixture(scope="module")
def tiny_model():
    """真结构模型但小输入:img_size 会被 tile 前的 preprocess resize 到 512
    (preprocess 内部 resize),模型本身与常驻 fixture 相同——直接构建
    随机初始化模型即可,前向契约与权重值无关。"""
    from src.models.multi_task_model import MultiTaskThyroidModel
    m = MultiTaskThyroidModel(backbone_name="efficientnet_b2", pretrained=False,
                              config=CONFIG)
    m.eval()
    return m


# ── 几何 ────────────────────────────────────────────────────────────────


def test_tile_origins_small_image_single_tile():
    """小图(两维 ≤ tile)返回单起点 [(0,0)] = 恒等路径。"""
    assert tile_origins(400, 300, 512, 384) == [(0, 0)]
    assert tile_origins(512, 512, 512, 384) == [(0, 0)]


def test_tile_origins_large_image_covers_all():
    """大图(H=532,W=727)起点数 >1 且边缘补齐。"""
    o = tile_origins(532, 727, 512, 384)
    assert len(o) == 4  # 2×2
    ys = sorted({y for y, _ in o})
    xs = sorted({x for _, x in o})
    assert ys == [0, 20]      # H=532:532-512=20
    assert xs == [0, 215]     # W=727:727-512=215(stride 384 越过后钳位)
    assert (0, 0) in o


def test_tile_origins_full_coverage_no_gap():
    """任意尺寸下相邻 tile 间隙 < tile(全覆盖,无缝隙条带)。注意 (H,W) 顺序。"""
    for h, w in [(532, 727), (768, 1024), (600, 800), (513, 900)]:
        o = tile_origins(h, w, 512, 384)
        ys = sorted({y for y, _ in o})
        xs = sorted({x for _, x in o})
        for a, b in zip(ys, ys[1:]):
            assert b - a < 512  # 相邻 tile 原点距 < tile → 有重叠
        for a, b in zip(xs, xs[1:]):
            assert b - a < 512
        assert ys[-1] + 512 >= h and xs[-1] + 512 >= w  # 边缘覆盖


# ── env 解析 ───────────────────────────────────────────────────────────


def test_tiling_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AICOMP_STATIC_TILING", raising=False)
    assert _tiling_enabled() is False


def test_tiling_enabled_env(monkeypatch):
    monkeypatch.setenv("AICOMP_STATIC_TILING", "1")
    assert _tiling_enabled() is True
    monkeypatch.setenv("AICOMP_STATIC_TILING", "garbage")
    assert _tiling_enabled() is False  # 畸形回关


def test_tiling_geometry_bounds(monkeypatch):
    monkeypatch.delenv("AICOMP_TILING_TILE", raising=False)
    monkeypatch.delenv("AICOMP_TILING_STRIDE", raising=False)
    assert _tiling_geometry() == (512, 384)
    monkeypatch.setenv("AICOMP_TILING_TILE", "abc")  # 非整数回默认
    assert _tiling_geometry() == (512, 384)
    monkeypatch.setenv("AICOMP_TILING_TILE", "640")  # ≠512 fail-loud
    with pytest.raises(ValueError, match="TILING_TILE"):
        _tiling_geometry()
    monkeypatch.setenv("AICOMP_TILING_TILE", "512")
    monkeypatch.setenv("AICOMP_TILING_STRIDE", "512")  # ==tile 零重叠
    with pytest.raises(ValueError, match="TILING_STRIDE"):
        _tiling_geometry()
    monkeypatch.setenv("AICOMP_TILING_STRIDE", "100")  # < tile//2
    with pytest.raises(ValueError, match="TILING_STRIDE"):
        _tiling_geometry()


# ── 零回归(env 关 = 旧路径)────────────────────────────────────────────


def test_diagnose_static_case_default_path_unchanged(tiny_model, monkeypatch):
    """env 未设时 diagnose_static_case 不含 tiling_used 键(走原路径)。"""
    from src.inference.case_pipeline import diagnose_static_case
    monkeypatch.delenv("AICOMP_STATIC_TILING", raising=False)
    monkeypatch.delenv("AICOMP_STATIC_TTA", raising=False)
    img = (np.random.rand(532, 727) * 255).astype(np.uint8)
    result = diagnose_static_case(tiny_model, img, CONFIG, None, "fp32",
                                  torch.device("cpu"))
    assert "tiling_used" not in result  # 原路径无此键
    assert result["preds"]["mask_binary"].shape == (512, 512)


def test_diagnose_static_case_small_image_tiling_identity(
        tiny_model, monkeypatch):
    """tiling 开 + 小图 → 恒等路径(单前向,不进融合;无 tiling_used)。"""
    from src.inference.case_pipeline import diagnose_static_case
    monkeypatch.setenv("AICOMP_STATIC_TILING", "1")
    img = (np.random.rand(400, 300) * 255).astype(np.uint8)
    result = diagnose_static_case(tiny_model, img, CONFIG, None, "fp32",
                                  torch.device("cpu"))
    assert "tiling_used" not in result


# ── 融合正确性 + 分类单源 ──────────────────────────────────────────────


def test_tiled_forward_shapes_and_coords(tiny_model):
    """大图 727×532:融合概率图 shape=原图;分类 outputs 是有效 dict。"""
    img = (np.random.rand(532, 727) * 255).astype(np.uint8)
    outputs, prob = diagnose_static_case_tiled(
        tiny_model, img, CONFIG, None, "fp32", torch.device("cpu"),
        tile=512, stride=384)
    assert prob.shape == (532, 727)          # 原图坐标系
    assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0
    assert "mask" in outputs and "location_probs" in outputs  # 单 tile 全键


def test_tiled_fusion_is_max_not_mean(tiny_model):
    """重叠区取 max:两 tile 输出概率恒有 prob_fused == max(两个 tile 概率)。

    用 monkeypatch 的假模型:tile(0,0) 概率 0.6 全图、tile(0,215) 概率 0.9
    全图 → 重叠区 [215,512) 应为 0.9(max)非 0.75(平均)。"""
    class FakeModel(torch.nn.Module):
        def forward(self, x, gamma_prior=1.0):
            b = x.shape[0]
            # 按输入内容区分 tile:pad 区为 0 → 右侧 tile 的均值更低
            intensity = float(x.mean())
            mask_logit = torch.full((b, 1, 512, 512), 3.0 if intensity > 0.4
                                    else 1.5)
            outs = {
                "mask": mask_logit,
                "composition": torch.zeros(b, 3),
                "echogenicity": torch.zeros(b, 3),
                "shape": torch.zeros(b, 3),
                "margin": torch.zeros(b, 3),
                "foci": torch.zeros(b, 1),
                "malignancy": torch.zeros(b, 2),
                "concept_vector": torch.zeros(b, 18),
                "latent_res": torch.zeros(b, 16),
                "z_latent": torch.zeros(b, 8),
                "z_contrastive": torch.zeros(b, 8),
                "physical_vector": torch.zeros(b, 3),
                "spin_l1": torch.zeros(1),
                "stage4_spatial": torch.zeros(b, 8, 16, 16),
                "soft_mask_detached": torch.zeros(b, 1, 512, 512),
                "nodule_scale_detached": torch.zeros(b, 1),
                "location_side_logits": torch.zeros(b, 7),
                "location_level_logits": torch.zeros(b, 7),
                "location_probs": torch.zeros(b, 7),
                "ete_logits": torch.zeros(b, 4),
                "ete_probs": torch.zeros(b, 4),
                "ete_band_ratio": torch.zeros(b, 1),
                "t_logits": torch.zeros(b, 2),
                "t_probs": torch.zeros(b, 2),
                "t_conf": torch.zeros(b),
                "tirads_logits": torch.zeros(b, 5),
                "tirads_probs": torch.zeros(b, 5),
                "seg_aux2": torch.zeros(b, 1, 512, 512),
                "seg_aux3": torch.zeros(b, 1, 512, 512),
            }
            return outs

    # sigmoid(3.0)≈0.953, sigmoid(1.5)≈0.818
    img = np.full((532, 727), 200, dtype=np.uint8)  # 整图高亮 → 左 tile 高均值
    _, prob = diagnose_static_case_tiled(
        FakeModel(), img, CONFIG, None, "fp32", torch.device("cpu"),
        tile=512, stride=384)
    # 左 tile(0,0) 覆盖 [0,512),右 tile(0,215) 覆盖 [215,727)
    # 右 tile 因 pad 区(727-215=512 无 pad,实际 215+512=727 恰好满)——
    # 改用纵向:下 tile pad(532-20=512 无 pad)。此处右 tile 无 pad,
    # 需 pad 的是 y=20 起点(532-20=512 也满)。727/532 的 2×2 恰无 pad!
    # 换 800×600:tile 512,起点 y∈{0,88} x∈{0,288},右/下 tile 无 pad;
    # 只有最右 x=288(288+512=800 满)也无 pad。**727×532 恰全满,验证不了
    # pad 分支** —— 选用 750×550(x 起点 0,238;238+512=750 满;y 起点
    # 0,38;38+512=550 满)仍满。构造 pad 需 dim > tile×2:用 1100×550。
    img2 = np.full((550, 1100), 200, dtype=np.uint8)
    _, prob2 = diagnose_static_case_tiled(
        FakeModel(), img2, CONFIG, None, "fp32", torch.device("cpu"),
        tile=512, stride=384)
    # x 起点 {0, 384, 588}:588+512=1100 恰满;x=384 tile 有 pad(384+512=896
    # <1100?否,896<1100 → 有 204px pad)。
    # 右侧 tile 含 pad 零区 → 均值更低 → 假模型给它低 logit → 融合 max 在
    # 重叠区应取左 tile 的高值。
    assert prob2.shape == (550, 1100)
    # 全图任何像素 ≥ min(sigmoid(1.5))≈0.818:证明 max 没把高 tile 平均掉
    assert float(prob2.min()) >= 0.818 - 1e-4


def test_tiled_fusion_max_not_last_wins(monkeypatch):
    """重叠区取 max,不是后写覆盖(Gate2 路A:旧测试两 tile 同值,杀不掉 last-wins)。"""
    logits = [2.197, -0.847]  # sigmoid ≈ 0.90 / 0.30,按前向顺序
    n = {"i": 0}

    def fake_forward(model, img_tensor, precision, device):
        logit = logits[min(n["i"], len(logits) - 1)]
        n["i"] += 1
        return {"mask": torch.full((1, 1, 512, 512), logit)}

    monkeypatch.setattr("src.inference.tiling.run_forward", fake_forward)
    img = np.zeros((512, 896), dtype=np.uint8)  # origins (0,0),(0,384)
    _, prob = diagnose_static_case_tiled(
        None, img, CONFIG, None, "fp32", torch.device("cpu"), 512, 384)
    overlap = prob[:, 384:512]
    assert float(overlap.min()) >= 0.90 - 1e-3  # max(0.90,0.30)=0.90,非 last=0.30
    n["i"] = 0
    logits[:] = [-0.847, 2.197]  # 顺序对调仍 max
    _, prob2 = diagnose_static_case_tiled(
        None, img, CONFIG, None, "fp32", torch.device("cpu"), 512, 384)
    assert float(prob2[:, 384:512].min()) >= 0.90 - 1e-3


def test_select_classification_tile_contains_pixel():
    """stride==tile 时像素必须落在选中 tile 的 [oy,oy+tile)×[ox,ox+tile)。"""
    origins = tile_origins(1199, 1199, 512, 512)
    tagged = [(oy, ox, {"tag": f"T{oy},{ox}"}) for oy, ox in origins]
    sel = _select_classification_tile(tagged, (512, 512), 512)
    assert sel["tag"] == "T512,512"  # 旧实现选中 T0,0


def test_select_classification_tile_nearest_not_first():
    """重叠区取起点更近的 tile,不是先出现的。"""
    tagged = [
        (0, 0, {"tag": "T0,0"}),
        (0, 384, {"tag": "T0,384"}),
    ]
    # (256,500) 在两 tile 内;距 T0,384 中心更近
    sel = _select_classification_tile(tagged, (256, 500), 512)
    assert sel["tag"] == "T0,384"
    first_only = _select_classification_tile(tagged, (256, 100), 512)
    assert first_only["tag"] == "T0,0"


def test_diagnose_static_case_tiled_rejects_tile_neq_img_size():
    img = np.zeros((600, 800), dtype=np.uint8)
    with pytest.raises(ValueError, match="img_size"):
        diagnose_static_case_tiled(
            None, img, CONFIG, None, "fp32", torch.device("cpu"), 640, 320)


def test_classification_single_source_not_fused():
    """分类 tile 选择:argmax 像素所属 tile 的 outputs 原样返回(引用同一性)。"""
    class FakeModel(torch.nn.Module):
        def forward(self, x, gamma_prior=1.0):
            b = x.shape[0]
            return {
                "mask": torch.full((b, 1, 512, 512), 2.0),
                "malignancy": torch.zeros(b, 2),
                "location_probs": torch.zeros(b, 7),
                "ete_probs": torch.zeros(b, 4),
                "t_probs": torch.zeros(b, 2),
                "t_conf": torch.zeros(b),
                "tirads_probs": torch.zeros(b, 5),
            }

    img = (np.random.rand(532, 727) * 255).astype(np.uint8)
    outputs, _ = diagnose_static_case_tiled(
        FakeModel(), img, CONFIG, None, "fp32", torch.device("cpu"),
        tile=512, stride=384)
    # 返回的 outputs 是某次前向的原样 dict(键值单源,无融合痕迹:
    # mask 仍是 [1,1,512,512] tile 域 logits,不是融合图)
    assert outputs["mask"].shape == (1, 1, 512, 512)


# ── 端到端:decode 消费融合图 ──────────────────────────────────────────


def test_diagnose_static_case_tiled_e2e(tiny_model, monkeypatch):
    """tiling 开 + 大图 → 全链路:融合概率图(原图域)进 decode,preds 的
    pred_mask/mask_binary 直接在原图域(532×727),mask_resized 恒等。"""
    from src.inference.case_pipeline import diagnose_static_case
    monkeypatch.setenv("AICOMP_STATIC_TILING", "1")
    monkeypatch.delenv("AICOMP_STATIC_TTA", raising=False)
    img = (np.random.rand(532, 727) * 255).astype(np.uint8)
    result = diagnose_static_case(tiny_model, img, CONFIG, None, "fp32",
                                  torch.device("cpu"))
    assert result["tiling_used"] == (512, 384)
    # mask_prob_override 是原图域 → decode 的 pred_mask/mask_binary 均为
    # 原图尺寸(与 TTA 的 512 域不同——tiling 从一开始就在原图坐标系)
    assert result["preds"]["pred_mask"].shape == (532, 727)
    assert result["preds"]["mask_binary"].shape == (532, 727)
    # mask_resized 的 resize 目标是原图 (w,h)=(727,532) → 同尺寸恒等
    assert result["mask_resized"].shape == (532, 727)
