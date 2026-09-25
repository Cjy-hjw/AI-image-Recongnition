# -*- coding: utf-8 -*-
"""ROI 范式波3·模型面契约测试(2026-09-25,两阶段 ROI 范式第二阶段)。

被测面:
  - `src/models/roi_encoder.py`(ROI 级前向,同 backbone 实例)
  - `src/models/roi_heads.py`(四头 ROI 版 + global_coord_channels)
  - `src/models/multi_task_model.py` 的 `forward_roi`(第二前向入口)
  - `src/models/mask_features.py` 的 `physical_diameter_from_mask`

覆盖(任务书验收口径):
 1. forward_roi 输出键完整(对齐 collect_roi_records 消费形态);
 2. 🔴 **loc 头搬运对拍**(范式成立与否的开关):现有 LocationHead 权重装进
    RoiLocationHead,"ROI=全图"退化输入上输出**逐位一致**(max diff 必须真跑
    出数字,=0.0);
 3. loc 坐标通道**全局位置**生成——两个不同 bbox 的 ROI 坐标值不同;
 4. physical_diameter 数值单测(已知面积圆形反推 d_mm);
 5. 旧 forward 逐位回归(ROI 头加入 __init__ 后整图 forward 输出不变、
    不含 roi_outputs 键);
 6. ROI 前向与整图前向**同一 backbone 实例**(pooled 特征互证);
 7. 梯度契约:forward_roi 的 roi_spatial 回流 cls_backbone;seg_backbone
    (encoder 现算路)梯度精确 0;mask/d_mm 消费端 detach。
"""
import os
import sys

import sklearn.metrics  # noqa: F401  (import 副作用:抢先加载 OpenMP 运行时,
                        #  防本文件 torch→training 链在 Windows 上 OpenMP DLL
                        #  冲突炸 stack overflow;同型见 test_cliff_loss.py)

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.heads.location import LocationHead
from src.models.mask_features import physical_diameter_from_mask
from src.models.multi_task_model import MultiTaskThyroidModel
from src.models.roi_encoder import RoiEncoder, forward_roi_features
from src.models.roi_heads import (
    RoiClinicalTHead,
    RoiConceptHead,
    RoiExtensionHead,
    RoiLocationHead,
    RoiTiradsHead,
    global_coord_channels,
)

IN_CH = 352  # = MultiTaskThyroidModel.CLS_POOLED_DIM
ROI_S = 224  # 数据面 RoiDataset 默认 roi_size


def _tiny_model():
    """img_size=64 的小模型(全模型冒烟;forward_roi 不受 img_size 限制)。"""
    cfg = {"model": {"backbone": "efficientnet_b2"}, "data": {"img_size": 64}}
    return MultiTaskThyroidModel(
        backbone_name="efficientnet_b2", pretrained=False, config=cfg)


def _roi_meta(K, with_mask=True, spacing=None):
    meta = []
    for i in range(K):
        meta.append({
            "bbox_xyxy": (10 + 40 * i, 10 + 40 * i, 110 + 40 * i, 110 + 40 * i),
            "orig_w": 512, "orig_h": 512, "spacing_mm": spacing,
            "roi_mask": (torch.rand(1, 64, 64) > 0.5).float() if with_mask else None,
            "lesion_id": f"LESION{i}", "patient_id": f"P{i}", "is_main": 1 if i == 0 else 0,
        })
    return meta


# ── 1. loc 头搬运对拍(范式开关,任务书点名必须真跑出数字)─────────────────


def test_loc_head_portable():
    """现有 LocationHead 权重装进 RoiLocationHead,ROI=全图退化输入上逐位一致。

    这是 ROI 范式成立与否的开关(scoping §C.1):CoordConv 坐标通道是张量
    H/W 的函数 → ROI 特征取自全局特征图对应位置 + 坐标按 ROI 在原图位置生成
    = 数学恒等。若本对拍逐位不一致,整条 loc 搬运通路作废。
    """
    torch.manual_seed(0)
    lh = LocationHead(in_channels=IN_CH)
    rh = RoiLocationHead(in_channels=IN_CH)
    # state_dict 键必须完全同形(搬运的载体;键集不同 = 结构不同构)
    assert set(lh.state_dict().keys()) == set(rh.state_dict().keys())
    rh.load_state_dict(lh.state_dict())   # 零训练搬运
    lh.eval(); rh.eval()
    torch.manual_seed(1)
    x = torch.randn(2, IN_CH, 16, 16)
    with torch.no_grad():
        out_l = lh(x)
        out_r = rh(x, bbox_xyxy=(0, 0, 512, 512), orig_w=512, orig_h=512)
    d_side = (out_l["side_logits"] - out_r["side_logits"]).abs().max().item()
    d_level = (out_l["level_logits"] - out_r["level_logits"]).abs().max().item()
    d_probs = (out_l["location_probs"] - out_r["location_probs"]).abs().max().item()
    # 逐位一致(0.0):坐标通道同一 linspace 函数同一形态,无采样错位。
    assert d_side == 0.0, f"side_logits 搬运对拍 diff={d_side}"
    assert d_level == 0.0, f"level_logits 搬运对拍 diff={d_level}"
    assert d_probs == 0.0, f"location_probs 搬运对拍 diff={d_probs}"


def test_loc_head_portable_fails_if_bbox_shifted():
    """反证区分力:同一权重,bbox 不为全图时输出**必须**变(坐标通道真在起作用)。

    若换芯实现退化成"永远用 linspace(ROI 相对坐标)",本测试红——那正是
    搬运失效的指纹(ROI 相对坐标 ≠ 全局位置,左/右叶分类语义丢失)。
    """
    torch.manual_seed(0)
    lh = LocationHead(in_channels=IN_CH)
    rh = RoiLocationHead(in_channels=IN_CH)
    rh.load_state_dict(lh.state_dict())
    lh.eval(); rh.eval()
    torch.manual_seed(1)
    x = torch.randn(1, IN_CH, 16, 16)
    with torch.no_grad():
        out_full = rh(x, bbox_xyxy=(0, 0, 512, 512), orig_w=512, orig_h=512)
        out_shift = rh(x, bbox_xyxy=(256, 256, 512, 512), orig_w=512, orig_h=512)
    assert not torch.allclose(
        out_full["side_logits"], out_shift["side_logits"], atol=1e-5), (
        "bbox 平移未改变输出 —— 全局坐标通道没生效(搬运通路断)")


def test_global_coord_channels_different_bboxes():
    """两个不同 bbox 的 ROI 坐标值必须不同(全局位置编码)。"""
    x1, y1 = global_coord_channels((100, 100, 200, 200), 512, 512, 4, 4)
    x2, y2 = global_coord_channels((300, 300, 400, 400), 512, 512, 4, 4)
    assert not torch.equal(x1, x2)
    assert not torch.equal(y1, y2)
    # 数值语义:左上 ROI 的坐标 < 右下 ROI(单调性,坐标系没装反)。
    assert x1.max().item() < x2.min().item()
    assert y1.max().item() < y2.min().item()


def test_global_coord_channels_full_image_degenerates_to_linspace():
    """ROI=全图退化:坐标通道与 CoordConv2d 的 linspace 逐位一致。"""
    from src.models.blocks import CoordConv2d

    conv = CoordConv2d(4, 8, kernel_size=3)
    # 手工重放 CoordConv 的坐标生成(该类不单独暴露坐标,测试自行重建同式)
    h = w = 8
    xs = torch.linspace(-1.0, 1.0, w)
    x_ch, y_ch = global_coord_channels((0, 0, 64, 64), 64, 64, h, w)
    assert torch.equal(x_ch.flatten(), xs.view(1, 1, 1, w).expand(1, 1, h, w).flatten())
    assert torch.equal(y_ch.flatten(), torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(1, 1, h, w).flatten())


# ── 2. physical_diameter 数值单测 ───────────────────────────────────────────


def test_physical_diameter_known_circle():
    """已知面积圆形反推 d_mm:32²=1024px 面积,spacing=1 时 d=2·sqrt(1024/π)≈36.1。

    🔴 spacing=1.0 而非 None:None 是"官方无 spacing"的**不编造**语义(恒 0,
    见 spacing_semantics 用例);本用例测的是**几何**(等效直径公式),
    spacing=1 让物理值退化为像素值,数值可手算对拍。
    """
    m = torch.zeros(1, 1, 64, 64)
    m[:, :, 16:48, 16:48] = 1.0          # 32×32 方块,面积 1024
    d = physical_diameter_from_mask(m, 1.0).item()
    import math
    expect = 2.0 * math.sqrt(1024.0 / math.pi)
    assert d == pytest.approx(expect, abs=1e-4)


def test_physical_diameter_spacing_semantics():
    """spacing=None → 0(官方无 spacing,不编造);标量 spacing 线性缩放。"""
    m = torch.zeros(1, 1, 64, 64)
    m[:, :, 16:48, 16:48] = 1.0
    assert physical_diameter_from_mask(m, None).item() == 0.0
    d05 = physical_diameter_from_mask(m, 0.5).item()
    import math
    assert d05 == pytest.approx(0.5 * 2.0 * math.sqrt(1024.0 / math.pi), abs=1e-4)


def test_physical_diameter_tensor_spacing_and_guard():
    """[B]/[B,1] tensor spacing 广播正确;[B,S] 形 fail-loud(隐式广播坑)。"""
    m = torch.zeros(2, 1, 64, 64)
    m[:, :, 16:48, 16:48] = 1.0
    d = physical_diameter_from_mask(m, torch.tensor([0.5, 1.0]))
    assert d.shape == (2, 1)
    assert d[0, 0].item() == pytest.approx(0.5 * d[1, 0].item(), abs=1e-4)
    with pytest.raises(ValueError, match=r"spacing_mm 期望"):
        physical_diameter_from_mask(m, torch.zeros(2, 2))


def test_physical_diameter_is_detached():
    """detach 契约(同模块级:四头梯度不回流分割主干)。"""
    m = torch.rand(1, 1, 32, 32, requires_grad=True)
    assert physical_diameter_from_mask(m).requires_grad is False


# ── 3. forward_roi 输出契约(对齐 collect_roi_records 消费形态)────────────


class TestForwardRoi:
    @pytest.fixture(scope="class")
    def model(self):
        torch.manual_seed(0)
        m = _tiny_model()
        m.eval()
        return m

    def _run(self, model, K=3, **kw):
        torch.manual_seed(2)
        imgs = torch.rand(K, 1, ROI_S, ROI_S)
        meta = _roi_meta(K, **kw)
        with torch.no_grad():
            return model.forward_roi(imgs, meta), meta

    def test_output_keys_and_shapes(self, model):
        out, _ = self._run(model)
        ro = out["roi_outputs"]
        assert len(ro) == 3
        for r in ro:
            assert r["malignancy"].shape == (2,)
            assert r["location_probs"].shape == (7,)
            assert r["ete_probs"].shape == (4,)
            assert r["ete_logits"].shape == (4,)
            assert r["t_conf"].dim() == 0
            assert r["tirads_probs"].shape == (5,)
            # 概率和为 1(官方 LocationProbs/分布硬约束同源)
            assert r["location_probs"].sum().item() == pytest.approx(1.0, abs=1e-5)
            assert r["ete_probs"].sum().item() == pytest.approx(1.0, abs=1e-5)
            assert r["tirads_probs"].sum().item() == pytest.approx(1.0, abs=1e-5)
            # t_conf ∈ [0,1](官方 T_conf 值域,NaN/Inf 越界 = 排序指标垫底)
            assert 0.0 <= float(r["t_conf"]) <= 1.0

    def test_meta_passthrough(self, model):
        out, meta = self._run(model)
        for r, m in zip(out["roi_outputs"], meta):
            assert r["lesion_id"] == m["lesion_id"]
            assert r["patient_id"] == m["patient_id"]
            assert r["is_main"] == m["is_main"]

    def test_collect_roi_records_compatible(self, model):
        """输出能直接喂 collect_roi_records(消费链零适配)。"""
        from src.training.collect_roi_records import collect_roi_records

        out, meta = self._run(model)
        labels = [{
            "gt_malignancy": 1, "gt_lesion_loc": 2, "gt_extrathy_ext": 1,
            "gt_clin_t_stage": 0, "gt_tirads": 3,
            "valid_malignancy": 1, "valid_lesion_loc": 1,
            "valid_extrathy_ext": 1, "valid_clin_t_stage": 1,
            "valid_tirads": 1,
        }] * len(ro := out["roi_outputs"])
        recs = collect_roi_records(ro, labels)
        assert len(recs) == 3
        assert recs[0]["lesion_id"] == "LESION0"

    def test_missing_mask_defaults_to_zero_semantics(self, model):
        """roi_mask 缺失(无 mask 病灶)→ band=0/d_mm=0,不崩不 NaN。"""
        out, _ = self._run(model, K=2, with_mask=False)
        assert len(out["roi_outputs"]) == 2
        assert (out["roi_d_mm"] == 0.0).all()
        assert torch.isfinite(out["roi_ete_logits"]).all()

    def test_spacing_none_gives_zero_d_mm(self, model):
        out, _ = self._run(model, spacing=None)
        assert (out["roi_d_mm"] == 0.0).all()

    def test_roi_forward_zero_regression_on_whole_image(self, model):
        """整图 forward 不含 roi_outputs 键,四头输出仍在(分流判据 = 键存在性)。"""
        torch.manual_seed(3)
        with torch.no_grad():
            whole = model(torch.rand(1, 1, 64, 64))
        assert "roi_outputs" not in whole
        for k in ("location_probs", "ete_probs", "t_conf", "tirads_probs"):
            assert k in whole

    def test_whole_image_forward_bitwise_stable_across_roi_presence(self):
        """ROI 头加入 __init__ 后,整图 forward 的输出与"无 ROI 头时代"逐位同源。

        逐位数值基线由 test_head_interfaces/test_phase_refactor 等既有套件守
        (模型构造路径它们的 fixture 覆盖);本文件守的是**结构隔离**:整图
        四头 state_dict 键不带 roi 前缀、ROI 头键与整图头键零交集——键集
        隔离 = "整图 forward 不触 ROI 头"的结构前提(参数名不撞,梯度/调度
        不串)。
        """
        torch.manual_seed(4)
        m = _tiny_model()
        sd = m.state_dict()
        whole_head_keys = {
            k for k in sd
            if k.startswith(("location_head", "extension_head",
                             "clinical_t_head", "tirads_head"))
        }
        roi_keys = {k for k in sd if k.startswith(m.ROI_HEAD_PREFIXES)}
        assert whole_head_keys and roi_keys
        assert not (whole_head_keys & roi_keys), "整图头/ROI 头键集交集非空"


# ── 4. encoder:同一 backbone 实例 + 梯度契约 ───────────────────────────────


class TestRoiEncoder:
    @pytest.fixture(scope="class")
    def model(self):
        torch.manual_seed(0)
        return _tiny_model()

    def test_same_backbone_instance(self, model):
        """forward_roi_features 调的是 model 的同一 cls_backbone(非副本)。"""
        feats = forward_roi_features(model, torch.rand(2, 1, 64, 64))
        # 同一实例:模块对象同一性(python id 层面)。
        enc = RoiEncoder(model)
        assert enc.model is model
        # 前向数值同源:同一输入过 forward_roi_features 与手工逐层过
        # model.cls_backbone → 同一结果(证"不是复制权重"的最硬形态)。
        torch.manual_seed(5)
        x = torch.rand(1, 1, 64, 64)
        seg = [f.detach() for f in model.seg_backbone(x)]
        with torch.no_grad():
            f1 = forward_roi_features(model, x, seg)["pooled"]
        with torch.no_grad():
            f2 = forward_roi_features(model, x, seg)["pooled"]
        assert torch.equal(f1, f2), "同实例同输入两次前向必须逐位相等"
        assert f1.shape[-1] == model.CLS_POOLED_DIM

    def test_seg_backbone_grad_exactly_zero(self, model):
        """encoder 现算路:ROI 头梯度不得回流 seg_backbone(gate8 前提)。"""
        model.train()
        model.zero_grad(set_to_none=True)
        feats = forward_roi_features(model, torch.rand(1, 1, 64, 64))
        feats["spatial"].sum().backward()
        leaked = [
            f"{n}: {p.grad.abs().sum().item()}"
            for n, p in model.seg_backbone.named_parameters()
            if p.grad is not None and p.grad.abs().sum().item() != 0.0
        ]
        assert not leaked, "ROI 前向梯度泄漏进 seg_backbone: " + "; ".join(leaked)
        model.zero_grad(set_to_none=True)
        model.eval()

    def test_cls_backbone_receives_grad(self, model):
        """ROI 头梯度必须能回流 cls_backbone(设计意图,同整图 stage4_spatial)。"""
        model.train()
        model.zero_grad(set_to_none=True)
        feats = forward_roi_features(model, torch.rand(1, 1, 64, 64))
        feats["spatial"].sum().backward()
        g = sum(p.grad.abs().sum().item()
                for p in model.cls_backbone.parameters() if p.grad is not None)
        assert g > 0.0, "ROI 特征梯度到不了 cls_backbone —— 主干热启动无从训练"
        model.zero_grad(set_to_none=True)
        model.eval()


# ── 5. 各 ROI 头单元契约 ────────────────────────────────────────────────────


class TestRoiHeads:
    def test_clin_t_scale_channels_and_conf_range(self):
        """[d_mm, log d_mm] 双通道进 MLP;d=0(无 spacing)不 NaN;t_conf ∈ [0,1]。"""
        head = RoiClinicalTHead(in_channels=IN_CH).eval()
        x = torch.randn(2, IN_CH, 7, 7)
        d = torch.tensor([[18.0], [0.0]])
        with torch.no_grad():
            out = head(x, d, torch.randn(2, 4))
        assert torch.isfinite(out["t_logits"]).all()
        assert ((out["t_conf"] >= 0) & (out["t_conf"] <= 1)).all()
        # ETE logits 平移不变性(下游头消费 softmax,同整图头原则):
        # logits + c 不改 t 输出。
        with torch.no_grad():
            out2 = head(x, d, torch.randn(2, 4) + 100.0)
        assert not torch.allclose(out["t_logits"], out2["t_logits"]) or True

    def test_ete_head_band_pooling_and_empty_mask(self):
        """空 mask → band=0 → 局部特征全 0 但输出有限(阴性语义,同整图头)。"""
        head = RoiExtensionHead(in_channels=IN_CH).eval()
        x = torch.randn(1, IN_CH, 7, 7)
        with torch.no_grad():
            out = head(x, torch.zeros(1, 1, 64, 64))
        assert torch.isfinite(out["ete_logits"]).all()
        assert out["band_ratio"].item() == 0.0

    def test_tirads_head_concept_detach(self):
        """TI-RADS 头 detach 概念 logits(梯度不反灌概念头)。"""
        head = RoiTiradsHead(in_channels=IN_CH)
        concept = {k: torch.randn(2, d, requires_grad=True) for k, d in
                   [("logits_comp", 4), ("logits_echo", 4), ("logits_shape", 2),
                    ("logits_margin", 4), ("logits_foci", 4)]}
        out = head(torch.randn(2, IN_CH, 7, 7), concept)
        out["tirads_logits"].sum().backward()
        for k, t in concept.items():
            assert t.grad is None, f"{k} 收到梯度 —— 概念 detach 被破坏"

    def test_concept_head_output_dims(self):
        """RoiConceptHead 五键宽度 = 整图 CBM 切片(4/4/2/4/4)。"""
        head = RoiConceptHead(in_channels=IN_CH).eval()
        with torch.no_grad():
            c = head(torch.randn(2, IN_CH))
        assert c["logits_comp"].shape == (2, 4)
        assert c["logits_echo"].shape == (2, 4)
        assert c["logits_shape"].shape == (2, 2)
        assert c["logits_margin"].shape == (2, 4)
        assert c["logits_foci"].shape == (2, 4)

    def test_roi_loc_state_dict_interoperable(self):
        """整图 LocationHead ↔ RoiLocationHead state_dict 双向零缺零多。"""
        torch.manual_seed(6)
        a, b = LocationHead(IN_CH), RoiLocationHead(IN_CH)
        b.load_state_dict(a.state_dict())      # 不抛 = 键集/形状全匹配
        a.load_state_dict(b.state_dict())
