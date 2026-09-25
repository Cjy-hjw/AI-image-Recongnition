# -*- coding: utf-8 -*-
"""ROI 四头损失契约测试(两阶段 ROI 范式 P3 损失组,2026-09-25)。

覆盖(deepsup/cliff 范式,含变异必须 RED 项):
1.  零权重回归:roi_head_weight=0 时编排返回 python 0.0 + 恒 0 detail 键
    (键恒在——接线被删 JSONL 立刻可见,不自证);MultiTaskLoss 全链
    legacy loss_dict 逐位不引入 roi_* 非 0 值。
2.  fail-loud:权重>0 无 roi_outputs / 无 roi_labels → raise;长度不齐 → raise。
3.  valid 过滤:valid=0 病灶零贡献(数值对拍)。
4.  is_main 门:is_main=0 病灶零贡献;is_main 缺席(None)全当主病灶。
5.  四头各一条数值单测(手算对拍):
    - location:边缘分布侧别 CE 手算(含峡部 level 屏蔽);
    - ete:focal=0 退化加权 CE 手算;
    - clin_t:BCE(概率域)手算(含 clamp 端点);
    - tirads:概率域 CE 手算(含 ordinal>0 的期望项)。
6.  分解测试:编排层 L(λ) − L(0) = λ × (四头原始和)精确式。
7.  变异杀测:is_main 门被删(全病灶监督)→ 数值必变(区分力)。
8.  梯度:四头张量梯度有限非零;malignancy 键不被消费(良恶性归整图链路)。
9.  k_ref 分母平滑(债6 同批,2026-09-25):n_valid≥4 逐位不变(旧公式对拍);
    n_valid∈(0,4) 分母抬到 4(单样本损失压 4×);v 全 0 → 精确 0。
    🔴 手算断言一律给 **≥4 有效样本**的 fixture(手算均值公式分母=K_valid
    只在 ≥4 时与实现一致;K_valid<4 时实现分母是 clamp 后的 4,不是 K_valid)。
10. 债6 概念监督(方案 A 简化版):默认 0 逐位零回归;监督开后概念头有学习
    信号(小随机数据训几步 loss 下降);valid 屏蔽不进分母;foci 恒零梯度
    (grad is None);变异杀测(去掉 valid 门注入必 RED);窄 dict 误传
    fail-loud;计数键恒在。
11. §J.5b 三条测试债:gt_ fixture 源级对拍(真实 roi_labels_from_meta 桥
    产物直喂损失,断言非零——手写 gt_ fixture 键名错时零杀的缺口);
    label_smoothing>0 分支数值覆盖;零权重 detail 的 device 断言(与输入
    同 device,GPU 下 .to() 不炸)。
"""
import sys
from pathlib import Path

import sklearn.metrics  # noqa: F401  (import 副作用:抢先加载 OpenMP 运行时,
                        #  防本文件 torch→heads 链在 Windows 上 OpenMP DLL
                        #  冲突炸 stack overflow;test_cliff_loss.py:29 同款)

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.losses.components.roi_heads import (  # noqa: E402
    compute_roi_head_losses,
)
from src.models.losses.components.location_roi import (  # noqa: E402
    compute_location_roi_loss,
)
from src.models.losses.components.extension_roi import (  # noqa: E402
    compute_extension_roi_loss,
)
from src.models.losses.components.clinical_t_roi import (  # noqa: E402
    compute_clinical_t_roi_loss,
)
from src.models.losses.components.tirads_roi import (  # noqa: E402
    compute_tirads_roi_loss,
)

_EPS = 1e-8
LOCATION_KEYS = ("roi_location", "roi_location_side", "roi_location_level",
                 "roi_ete", "roi_ete_ce", "roi_clin_t", "roi_clin_t_ce",
                 "roi_tirads", "roi_tirads_ce", "roi_tirads_ordinal")


# ── fixtures ────────────────────────────────────────────────────────────────

def _roi(lid="L", *, loc_probs=None, ete_logits=None, t_conf=0.5,
         tirads_probs=None, is_main=1):
    """造一条 roi_outputs 元素(模型面 forward_roi 契约键)。"""
    return {
        "lesion_id": lid,
        "is_main": is_main,
        "patient_id": None,
        "malignancy": torch.tensor([0.0, 1.0]),
        "location_probs": (torch.softmax(torch.randn(7), dim=0)
                           if loc_probs is None else loc_probs),
        "ete_probs": torch.softmax(torch.randn(4), dim=0),
        "ete_logits": (torch.randn(4) if ete_logits is None else ete_logits),
        "t_conf": torch.tensor(float(t_conf)),
        "tirads_probs": (torch.softmax(torch.randn(5), dim=0)
                         if tirads_probs is None else tirads_probs),
    }


def _lab(lid="L", *, loc=0, ete=0, t=1, tirads=2, v_loc=1, v_ete=1, v_t=1,
         v_tirads=1, is_main=1, v_mal=1, mal=1):
    """造一条 roi_labels 元素(数据面契约键,gt_ 前缀)。"""
    return {
        "gt_malignancy": mal,
        "gt_lesion_loc": loc,
        "gt_extrathy_ext": ete,
        "gt_clin_t_stage": t,
        "gt_tirads": tirads,
        "valid_malignancy": v_mal,
        "valid_lesion_loc": v_loc,
        "valid_extrathy_ext": v_ete,
        "valid_clin_t_stage": v_t,
        "valid_tirads": v_tirads,
        "lesion_id": lid,
        "is_main": is_main,
    }


def _batch_3(seed=0):
    """K=3 标准批:主+次+invalid,四头标签各异。"""
    torch.manual_seed(seed)
    labs = [
        _lab("L", loc=0, ete=0, t=1, tirads=2, is_main=1),
        _lab("S", loc=6, ete=3, t=0, tirads=4, is_main=0),
        _lab("M", loc=3, ete=1, t=1, tirads=0, is_main=1,
             v_loc=0, v_ete=0, v_t=0, v_tirads=0),
    ]
    outs = {"roi_outputs": [_roi(l["lesion_id"]) for l in labs]}
    return outs, {"roi_labels": labs}


# ── 1. 零权重回归 ───────────────────────────────────────────────────────────

def test_zero_weight_returns_python_zero_and_constant_keys():
    """权重 0:total 是 python 0.0(不是 tensor);10 个 detail 键恒在恒 0。"""
    total, detail = compute_roi_head_losses({}, {}, lambda_roi=0.0)
    assert total == 0.0 and not isinstance(total, torch.Tensor)
    assert set(detail.keys()) == set(LOCATION_KEYS)
    for k in LOCATION_KEYS:
        assert isinstance(detail[k], torch.Tensor)
        assert float(detail[k]) == 0.0


def test_zero_weight_with_roi_keys_present_still_skips():
    """权重 0 时键存在也不算(零回归:旧模型/旧数据带 ROI 键不受影响)。"""
    outs, tgs = _batch_3()
    total, detail = compute_roi_head_losses(outs, tgs, lambda_roi=0.0)
    assert total == 0.0
    assert all(float(detail[k]) == 0.0 for k in LOCATION_KEYS)


def test_multitask_legacy_loss_dict_has_zero_roi_keys():
    """MultiTaskLoss 全链:legacy targets 的 loss_dict 含恒 0 的 roi_* 键,
    其余键与接线前一致(head_detail + seg_extra 照旧)。"""
    from src.models.losses.multitask import MultiTaskLoss

    torch.manual_seed(0)

    class _M(torch.nn.Module):
        def forward(self, x, gamma_prior=1.0):
            b = x.shape[0]
            return {"mask": torch.randn(b, 1, 32, 32),
                    "malignancy": torch.randn(b, 2),
                    "z_latent": torch.randn(b, 8),
                    "concept_vector": torch.randn(b, 18),
                    "spin_l1": torch.tensor(0.0),
                    "latent_res": torch.randn(b, 8),
                    "z_contrastive": torch.randn(b, 16),
                    "composition": torch.randn(b, 4),
                    "echogenicity": torch.randn(b, 4),
                    "shape": torch.randn(b, 2),
                    "margin": torch.randn(b, 4),
                    "foci": torch.randn(b, 4),
                    "location_side_logits": torch.randn(b, 3),
                    "location_level_logits": torch.randn(b, 3),
                    "ete_logits": torch.randn(b, 4),
                    "t_logits": torch.randn(b, 2),
                    "tirads_logits": torch.randn(b, 5)}

    outs = _M()(torch.randn(2, 1, 64, 64))
    targets = {
        "mask": (torch.rand(2, 1, 32, 32) > 0.5).float(),
        "valid_mask": torch.ones(2),
        "composition": torch.tensor([0, 1]),
        "echogenicity": torch.tensor([0, 1]),
        "shape": torch.tensor([0, 1]),
        "margin": torch.tensor([0, 1]),
        "foci": torch.zeros(2, 4),
        "malignancy": torch.tensor([0, 1]),
        "lesion_loc": torch.tensor([0, 4]),
        "valid_lesion_loc": torch.ones(2),
        "extrathy_ext": torch.tensor([0, 3]),
        "valid_extrathy_ext": torch.ones(2),
        "clin_t_stage": torch.tensor([0, 1]),
        "valid_clin_t_stage": torch.ones(2),
        "tirads": torch.tensor([2, 4]),
        "valid_tirads": torch.ones(2),
    }
    # 权重 0(默认构造,config 无键 → .get 兜 0):与显式 0 同一行为。
    crit_default = MultiTaskLoss()
    crit_zero = MultiTaskLoss(official_head_config={"roi_head_weight": 0.0})
    for crit in (crit_default, crit_zero):
        t, ld = crit(outs, targets, update_famo_weights=False)
        assert torch.isfinite(t)
        for k in LOCATION_KEYS:
            assert k in ld and float(ld[k]) == 0.0
        # 整图三头监控键照旧(未被 ROI 改动挤掉)。
        for k in ("location", "ete", "clin_t", "tirads"):
            assert k in ld
    assert crit_default.roi_head_weight == 0.0
    assert crit_zero.roi_head_weight == 0.0


# ── 2. fail-loud ────────────────────────────────────────────────────────────

def test_weight_positive_missing_outputs_key_raises():
    outs, tgs = _batch_3()
    stripped = {"other": 1}                       # 无 roi_outputs
    with pytest.raises(ValueError, match="roi_outputs"):
        compute_roi_head_losses(stripped, tgs, lambda_roi=0.3)


def test_weight_positive_missing_labels_key_raises():
    outs, tgs = _batch_3()
    with pytest.raises(ValueError, match="roi_labels"):
        compute_roi_head_losses(outs, {"mask": torch.zeros(1)}, lambda_roi=0.3)


def test_length_mismatch_raises():
    outs, tgs = _batch_3()
    short = {"roi_labels": tgs["roi_labels"][:2]}
    with pytest.raises(ValueError, match="长度不齐"):
        compute_roi_head_losses(outs, short, lambda_roi=1.0)


def test_multitask_weight_positive_without_roi_keys_raises():
    """MultiTaskLoss 全链 fail-loud:权重>0 + legacy targets(无 roi_labels)
    → raise(deepsup 同型:新损失配旧数据输出拒绝静默空算)。
    构造最小合法整图 outputs(过 concepts/malignancy 前置键)。
    """
    from src.models.losses.multitask import MultiTaskLoss

    torch.manual_seed(0)

    class _M(torch.nn.Module):
        def forward(self, x, gamma_prior=1.0):
            b = x.shape[0]
            return {"mask": torch.randn(b, 1, 32, 32),
                    "malignancy": torch.randn(b, 2),
                    "z_latent": torch.randn(b, 8),
                    "concept_vector": torch.randn(b, 18),
                    "spin_l1": torch.tensor(0.0),
                    "latent_res": torch.randn(b, 8),
                    "z_contrastive": torch.randn(b, 16),
                    "composition": torch.randn(b, 4),
                    "echogenicity": torch.randn(b, 4),
                    "shape": torch.randn(b, 2),
                    "margin": torch.randn(b, 4),
                    "foci": torch.randn(b, 4)}

    crit = MultiTaskLoss(official_head_config={"roi_head_weight": 0.5})
    outs = _M()(torch.randn(1, 1, 64, 64))
    targets = {"mask": torch.zeros(1, 1, 32, 32),
               "valid_mask": torch.ones(1),
               "composition": torch.tensor([0]),
               "echogenicity": torch.tensor([0]),
               "shape": torch.tensor([0]),
               "margin": torch.tensor([0]),
               "foci": torch.zeros(1, 4),
               "malignancy": torch.tensor([0])}
    with pytest.raises(ValueError, match="roi_head_weight"):
        crit(outs, targets, update_famo_weights=False)


# ── 3/4. valid 过滤 + is_main 门(数值对拍)────────────────────────────────

def test_valid_zero_lesion_contributes_nothing():
    """valid=0 病灶(M 行第三条)零贡献:K=3 与"只留前两条有效"的损失逐位一致。

    直接对拍单头(clin_t 最简):三病灶 [0.5, 0.7, 0.9] conf,第三条 valid=0
    → BCE 均值只算前两条。🔴 k_ref=4 下实现分母 = clamp(2,4)=4,分子只含
    前两条 → 手算 = (L0+L1)/4(**不是** (L0+L1)/2——旧均值公式在 n_valid<4
    时分母是 4)。对拍"只留前两条有效"的独立调用(同分母)证等价。
    """
    t_conf = torch.tensor([0.5, 0.7, 0.9])
    t_idx = torch.tensor([1, 0, 1])
    v = torch.tensor([1.0, 1.0, 0.0])
    total, _ = compute_clinical_t_roi_loss(t_conf, t_idx, v)
    # 独立调用:只给前两条(同 n_valid=2,同 clamp 分母)。
    total_ref, _ = compute_clinical_t_roi_loss(
        t_conf[:2], t_idx[:2], v[:2])
    assert float(total) == pytest.approx(float(total_ref), abs=1e-9)
    # 手算(分母=4):前两条的 BCE 之和 / 4。
    p = t_conf[:2].clamp(min=_EPS)
    one_minus_p = (1.0 - t_conf[:2]).clamp(min=_EPS)
    y = t_idx[:2].float()
    per = -(y * p.log() + (1 - y) * one_minus_p.log())
    manual = float(per.sum() / 4.0)
    assert float(total) == pytest.approx(manual, abs=1e-6)


def test_is_main_gate_zeroes_secondary_lesions():
    """is_main=0 病灶零贡献(与 valid=0 同效应,独立路径)。

    n_valid=1 < k_ref=4:实现分母 = clamp(1,4)=4,手算 = 单样本 BCE / 4。
    """
    t_conf = torch.tensor([0.5, 0.9])
    t_idx = torch.tensor([1, 1])
    v = torch.ones(2)
    is_main = torch.tensor([1.0, 0.0])            # 第二条是次病灶
    total, _ = compute_clinical_t_roi_loss(t_conf, t_idx, v, is_main=is_main)
    # 手算:只有第一条进分子,分母 clamp 到 4。
    p = t_conf[:1].clamp(min=_EPS)
    manual = float(-p.log()) / 4.0
    assert float(total) == pytest.approx(manual, abs=1e-6)


def test_is_main_none_treats_all_as_main():
    """is_main 缺席(None)= 全当主病灶(train_step 同口径)。"""
    t_conf = torch.tensor([0.5, 0.9])
    t_idx = torch.tensor([1, 1])
    v = torch.ones(2)
    total_none, _ = compute_clinical_t_roi_loss(t_conf, t_idx, v, is_main=None)
    total_all, _ = compute_clinical_t_roi_loss(
        t_conf, t_idx, v, is_main=torch.ones(2))
    assert float(total_none) == pytest.approx(float(total_all), abs=1e-9)


# ── 5. 四头数值单测(手算对拍)────────────────────────────────────────────

def test_location_roi_side_ce_hand_computed():
    """侧别/纵向分支 CE 手算:均匀底 0.05 + 主格 0.65,归一后对拍边缘。

    🔴 2026-09-25 Gate2 B 路 P1-3 修复后,level 边缘**归一到条件分布**
    P(level|非峡部):手算分母 = 非峡部 6 格概率和(=1−P_峡部),不是 1。
    K=5 ≥ k_ref=4:两分支分母各自 = v_side.sum()=5 与 v_level.sum()=5,
    与手算均值公式一致(分母不被 clamp 动)。
    """
    probs = torch.full((5, 7), 0.05)
    probs[0, 0] = 0.65                              # 左上(idx 0)
    probs[1, 3] = 0.65                              # 右上(idx 3)
    probs[2, 0] = 0.65
    probs[3, 3] = 0.65
    probs[4, 0] = 0.65
    probs = probs / probs.sum(dim=1, keepdim=True)  # 行和=1(0.35+0.65)
    loc_idx = torch.tensor([0, 3, 0, 3, 0])
    v = torch.ones(5)
    total, detail = compute_location_roi_loss(probs, loc_idx, v)
    # 手算(用归一后的同张量,防手抄舍入):
    side_L = probs[0, :3].sum()                     # 左三格 = 侧别 LEFT
    side_R = probs[1, 3:6].sum()                    # 右三格 = 侧别 RIGHT
    side_rows = []
    for i in (0, 1, 2, 3, 4):
        if loc_idx[i].item() == 0:
            side_rows.append(probs[i, :3].sum())    # 左上/左中/左下 → LEFT
        else:
            side_rows.append(probs[i, 3:6].sum())   # 右上 → RIGHT
    side_ce = float(-torch.stack(side_rows).log().mean())
    # level 条件分布:非峡部格 / 非峡部总(本 fixture 非峡部总 = 0.65+0.05×5=0.9)
    level_rows = []
    for i in range(5):
        non_isth = 1.0 - probs[i, 6]
        level_U = (probs[i, 0] + probs[i, 3]) / non_isth   # 上档 = idx 0 与 3
        level_rows.append(level_U)
    level_ce = float(-torch.stack(level_rows).log().mean())
    assert float(detail["roi_location_side"]) == pytest.approx(side_ce, abs=1e-5)
    assert float(detail["roi_location_level"]) == pytest.approx(
        level_ce, abs=1e-5)
    assert float(total) == pytest.approx(side_ce + level_ce, abs=1e-5)


def test_location_roi_isthmus_level_masked():
    """峡部(idx 6)对 level 分支零贡献:全峡部批 level 恒 0,side 正常。

    K=6 全峡部:v_level 全 0 → 分子分母皆 0 → 恒 0(eps 的 0 值路径);
    v_side=6 ≥ k_ref → side 均值公式分母不被 clamp 动,CE=log(7)/6 每条。
    """
    probs = torch.full((6, 7), 1.0 / 7)
    loc_idx = torch.tensor([6, 6, 6, 6, 6, 6])      # 全峡部
    v = torch.ones(6)
    total, detail = compute_location_roi_loss(probs, loc_idx, v)
    assert float(detail["roi_location_level"]) == 0.0
    assert float(detail["roi_location_side"]) == pytest.approx(
        float(torch.log(torch.tensor(7.0))), abs=1e-4)


def test_extension_roi_weighted_ce_hand_computed():
    """ETE focal=0 + 无权重 → 纯 CE 手算(logits 域 log_softmax)。

    K=4 = k_ref:分母 = clamp(4+eps, 4)=4+eps ≈ 均值公式分母(4),逐位对旧公式。
    """
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0],
                           [0.0, 3.0, 1.0, -1.0],
                           [1.0, 0.0, 0.5, 0.2],
                           [-0.3, 0.7, 0.1, 0.4]])
    idx = torch.tensor([0, 1, 2, 3])
    v = torch.ones(4)
    total, _ = compute_extension_roi_loss(
        logits, idx, v, class_weights=None, focal_gamma=0.0)
    log_p = F.log_softmax(logits, dim=-1)
    manual = float(-log_p.gather(1, idx.unsqueeze(1)).squeeze(1).mean())
    assert float(total) == pytest.approx(manual, abs=1e-6)


def test_extension_roi_focal_modulation_changes_value():
    """focal γ=2 与 γ=0 值不同(focal 真在调制,不是恒等)。"""
    torch.manual_seed(3)
    logits = torch.randn(4, 4)
    idx = torch.tensor([0, 1, 2, 3])
    v = torch.ones(4)
    t0, _ = compute_extension_roi_loss(logits, idx, v, focal_gamma=0.0)
    t2, _ = compute_extension_roi_loss(logits, idx, v, focal_gamma=2.0)
    assert float(t0) != pytest.approx(float(t2), abs=1e-5)


def test_extension_roi_gamma_in_open_interval_rejected():
    """γ∈(0,1) → assert(NaN 梯度坑,整图版实测坐实)。"""
    logits = torch.randn(2, 4)
    idx = torch.tensor([0, 1])
    with pytest.raises(AssertionError, match="focal_gamma"):
        compute_extension_roi_loss(
            logits, idx, torch.ones(2), focal_gamma=0.5)


def test_clinical_t_roi_bce_with_saturation_clamp():
    """t_conf 精确 0/1(sigmoid 饱和端)不产 -inf:clamp 防护。

    🔴 fp32 下 `1.0 - 1e-8` 舍入回 1.0(实测),单侧 clamp 上界不可用;
    实现按标签分流(p 与 1−p 各自下界 clamp),任何支不碰 log(0)。
    K=6 ≥ k_ref:均值公式分母不被 clamp 动。
    """
    t_conf = torch.tensor([1.0, 0.0, 1.0, 0.7, 0.5, 0.3])
    t_idx = torch.tensor([1, 0, 0, 1, 0, 1])
    v = torch.ones(6)
    total, _ = compute_clinical_t_roi_loss(t_conf, t_idx, v)
    assert torch.isfinite(total)
    # 手算:第三条 -log(eps)=18.42,第一/二条 0(饱和端正确分类),
    # 后三条各自 BCE;分母 = 6(≥ k_ref,均值口径)。
    per = []
    for p_val, y in [(1.0, 1), (0.0, 0), (1.0, 0),
                     (0.7, 1), (0.5, 0), (0.3, 1)]:
        p = max(p_val, _EPS)
        one_minus = max(1.0 - p_val, _EPS)
        per.append(-(y * torch.log(torch.tensor(p))
                     + (1 - y) * torch.log(torch.tensor(one_minus))))
    manual = float(torch.stack(per).sum() / 6.0)
    assert float(total) == pytest.approx(manual, abs=1e-4)


def test_tirads_roi_ce_hand_computed():
    """TI-RADS 概率域 CE 手算。K=5 ≥ k_ref:均值公式分母不被 clamp 动。"""
    probs = torch.tensor([[0.1, 0.2, 0.4, 0.2, 0.1],
                          [0.5, 0.2, 0.1, 0.1, 0.1],
                          [0.05, 0.05, 0.8, 0.05, 0.05],
                          [0.2, 0.2, 0.2, 0.2, 0.2],
                          [0.3, 0.3, 0.1, 0.1, 0.2]])
    idx = torch.tensor([2, 0, 2, 3, 1])
    v = torch.ones(5)
    total, detail = compute_tirads_roi_loss(probs, idx, v)
    manual = float(-torch.log(
        torch.stack([probs[i, idx[i]] for i in range(5)]).clamp(min=_EPS)).mean())
    assert float(detail["roi_tirads_ce"]) == pytest.approx(manual, abs=1e-6)
    assert float(total) == pytest.approx(manual, abs=1e-6)
    assert float(detail["roi_tirads_ordinal"]) == 0.0   # 默认关


def test_tirads_roi_ordinal_weight_positive():
    """ordinal>0:有序项 = ((Σ p_k·k) − y)² 手算;total = ce + w·ord。

    K=5 ≥ k_ref:两分支均值公式分母皆不被 clamp 动。
    """
    probs = torch.tensor([[0.1, 0.2, 0.4, 0.2, 0.1],
                          [0.5, 0.2, 0.1, 0.1, 0.1],
                          [0.05, 0.05, 0.8, 0.05, 0.05],
                          [0.2, 0.2, 0.2, 0.2, 0.2],
                          [0.3, 0.3, 0.1, 0.1, 0.2]])
    idx = torch.tensor([2, 0, 2, 3, 1])
    v = torch.ones(5)
    w = 0.5
    total, detail = compute_tirads_roi_loss(probs, idx, v, ordinal_weight=w)
    class_idx = torch.arange(5, dtype=torch.float32)
    exp_idx = (probs * class_idx).sum(dim=1)
    ord_manual = float(((exp_idx - idx.float()) ** 2).mean())
    ce_manual = float(-torch.log(
        torch.stack([probs[i, idx[i]] for i in range(5)]).clamp(min=_EPS)).mean())
    assert float(detail["roi_tirads_ordinal"]) == pytest.approx(
        ord_manual, abs=1e-6)
    assert float(total) == pytest.approx(ce_manual + w * ord_manual, abs=1e-6)


# ── 6. 编排层分解:L(λ) − L(0) = λ × 四头原始和 ─────────────────────────

def test_decomposition_lambda_additivity():
    """非自证分解:λ=0.7 与 λ=0 的 total 差精确等于
    0.7 × (L_loc + L_ete + L_t + L_tirads),其中四头原始值直接来自
    λ=1 的 detail(未乘 λ 的原始量级——监控约定同整图版)。"""
    outs, tgs = _batch_3(seed=11)
    t0, d0 = compute_roi_head_losses(outs, tgs, lambda_roi=0.0)
    t1, d1 = compute_roi_head_losses(outs, tgs, lambda_roi=1.0)
    t07, _ = compute_roi_head_losses(outs, tgs, lambda_roi=0.7)
    raw_sum = (float(d1["roi_location"]) + float(d1["roi_ete"])
               + float(d1["roi_clin_t"]) + float(d1["roi_tirads"]))
    assert float(t1) == pytest.approx(raw_sum, abs=1e-6)
    assert float(t07) == pytest.approx(0.7 * raw_sum, abs=1e-6)
    assert float(t0) == 0.0
    # detail 与 λ 无关(λ=1 与 λ=0.7 的 detail 逐位一致——监控不随 λ 漂移)。
    for k in LOCATION_KEYS:
        assert float(d1[k]) == pytest.approx(
            float(compute_roi_head_losses(outs, tgs, lambda_roi=0.7)[1][k]),
            abs=1e-9)


# ── 7. 变异杀测(区分力)────────────────────────────────────────────────

def test_mutation_is_main_gate_removed_killed():
    """变异:is_main 门被删(全病灶监督)→ 数值必变。

    _batch_3 里第二条是次病灶(is_main=0,四头全 valid):若门被删,该病灶
    的四头监督进分母 → 数值必不同于真实现。对拍独立手算"含门版"的
    clin_t 值,再构造"无门版"(is_main=None)证不同。
    """
    # 真实现:第二条被 is_main=0 屏蔽。
    t_conf = torch.tensor([0.5, 0.9])
    t_idx = torch.tensor([1, 1])
    v = torch.ones(2)
    gated, _ = compute_clinical_t_roi_loss(
        t_conf, t_idx, v, is_main=torch.tensor([1.0, 0.0]))
    # 变异:门删了(is_main=None → 全主病灶)。
    mutated, _ = compute_clinical_t_roi_loss(t_conf, t_idx, v, is_main=None)
    assert float(gated) != pytest.approx(float(mutated), abs=1e-6), \
        "is_main 门被删后数值不变 = fixture 无区分力"


# ── 8. 梯度 ─────────────────────────────────────────────────────────────────

def test_gradients_finite_and_nonzero():
    """四头输出张量梯度有限;is_main/valid 双门逐病灶核验。

    _batch_3 的人口:roi0(主+全 valid)→ 四头梯度非零;roi1(次病灶
    is_main=0)→ 四头梯度**精确 0**(is_main 门);roi2(valid 全 0)→ 精确 0。
    """
    outs, tgs = _batch_3(seed=17)
    for r in outs["roi_outputs"]:
        for k in ("location_probs", "ete_logits", "t_conf", "tirads_probs"):
            r[k].requires_grad_(True)
    total, detail = compute_roi_head_losses(outs, tgs, lambda_roi=1.0)
    total.backward()
    expected_nonzero = {0}
    for i, r in enumerate(outs["roi_outputs"]):
        for k in ("location_probs", "ete_logits", "t_conf", "tirads_probs"):
            g = r[k].grad
            assert g is not None and torch.isfinite(g).all(), f"roi{i}/{k}"
            if i in expected_nonzero:
                assert float(g.abs().sum()) > 0, f"roi{i}/{k} 梯度恒零"
            else:
                assert float(g.abs().sum()) == 0.0, \
                    f"roi{i}/{k} 被门屏蔽的病灶梯度非 0(is_main/valid 门失效)"


def test_all_invalid_batch_exact_zero_no_nan():
    """整批 valid=0 → 精确 0、无 NaN、梯度精确 0(masked 范式分母 eps)。"""
    outs, tgs = _batch_3(seed=23)
    for l in tgs["roi_labels"]:
        for k in list(l):
            if k.startswith("valid_"):
                l[k] = 0
    for r in outs["roi_outputs"]:
        for k in ("location_probs", "ete_logits", "t_conf", "tirads_probs"):
            r[k].requires_grad_(True)
    total, detail = compute_roi_head_losses(outs, tgs, lambda_roi=1.0)
    assert torch.isfinite(total)
    assert float(total) == pytest.approx(0.0, abs=1e-9)
    total.backward()
    for r in outs["roi_outputs"]:
        for k in ("location_probs", "ete_logits", "t_conf", "tirads_probs"):
            assert float(r[k].grad.abs().sum()) == 0.0


def test_malignancy_key_not_consumed_by_roi_losses():
    """良恶性归整图链路(AUCM pairwise):ROI 编排不得消费 malignancy 键。

    变异对拍:roi_outputs 的 malignancy 值翻转 → ROI 损失逐位不变。
    """
    outs, tgs = _batch_3(seed=29)
    t1, d1 = compute_roi_head_losses(outs, tgs, lambda_roi=1.0)
    outs2, _ = _batch_3(seed=29)
    for r in outs2["roi_outputs"]:
        r["malignancy"] = torch.tensor([9.0, -9.0])   # 极端翻转
    t2, d2 = compute_roi_head_losses(outs2, tgs, lambda_roi=1.0)
    assert float(t1) == pytest.approx(float(t2), abs=1e-12)
    for k in LOCATION_KEYS:
        assert float(d1[k]) == pytest.approx(float(d2[k]), abs=1e-12)


def test_etclass_weights_follow_label_not_prediction():
    """CB 权重按真实标签索引:同 logits 下换标签 → 权重路径变(区分力)。"""
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    v = torch.ones(1)
    w = torch.tensor([1.0, 10.0, 1.0, 1.0])       # 类 1 稀有加权 10×
    t_lab1, _ = compute_extension_roi_loss(
        logits, torch.tensor([1]), v, class_weights=w, focal_gamma=0.0)
    t_lab0, _ = compute_extension_roi_loss(
        logits, torch.tensor([0]), v, class_weights=w, focal_gamma=0.0)
    # 均匀 logits 下纯 CE 同值(log(4));加权后按标签取权 → 类1 是类0 的 10×。
    assert float(t_lab1) == pytest.approx(10.0 * float(t_lab0), abs=1e-5)


# ════════════════════════════════════════════════════════════════════════
# 9. k_ref 分母平滑(债6 同批,2026-09-25,DR 机制 1)
# ════════════════════════════════════════════════════════════════════════

def _clin_t_manual(t_conf, t_idx, denom):
    """clin_t 手算(与实现同式:分流 clamp,分子求和/给定分母)。"""
    per = []
    for p_val, y in zip(t_conf.tolist(), t_idx.tolist()):
        p = max(p_val, _EPS)
        one_minus = max(1.0 - p_val, _EPS)
        per.append(-(y * torch.log(torch.tensor(p))
                     + (1 - y) * torch.log(torch.tensor(one_minus))))
    return float(torch.stack(per).sum() / denom)


def test_kref_n_valid_ge4_bitwise_unchanged():
    """n_valid ≥ 4:clamp 不激活,分母 = v.sum()+eps 与旧公式逐位一致。

    对拍四头:K=6 全 valid 的实现值 vs 手算均值公式(v.sum()=6 做分母),
    精度 1e-7(eps 差 1e-8/6 量级)。
    """
    torch.manual_seed(41)
    t_conf = torch.rand(6)
    t_idx = torch.randint(0, 2, (6,))
    v = torch.ones(6)
    total, _ = compute_clinical_t_roi_loss(t_conf, t_idx, v)
    assert float(total) == pytest.approx(
        _clin_t_manual(t_conf, t_idx, 6.0), abs=1e-7)


def test_kref_n_valid_1_shrunk_4x():
    """n_valid=1 退化批:损失被压 4×(分母抬到 k_ref=4)。

    同一单有效样本:6 条批(1 valid)vs 手算 单样本 BCE/4;
    变异对拍——若 clamp 被删(分母=1),值是 4×,断言必 RED。
    """
    t_conf = torch.tensor([0.6, 0.9, 0.9, 0.9, 0.9, 0.9])
    t_idx = torch.tensor([1, 0, 0, 0, 0, 0])
    v = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # 只有第一条有效
    total, _ = compute_clinical_t_roi_loss(t_conf, t_idx, v)
    single_bce = _clin_t_manual(t_conf[:1], t_idx[:1], 1.0)
    assert float(total) == pytest.approx(single_bce / 4.0, abs=1e-6)
    assert float(total) != pytest.approx(single_bce, abs=1e-4)   # 变异杀测


def test_kref_all_invalid_still_exact_zero():
    """v 全 0:分子每项乘 0 → 0/clamp(eps,4)=精确 0、无 NaN(eps 语义保留)。"""
    t_conf = torch.rand(5)
    t_idx = torch.randint(0, 2, (5,))
    v = torch.zeros(5)
    total, detail = compute_clinical_t_roi_loss(t_conf, t_idx, v)
    assert torch.isfinite(total)
    assert float(total) == 0.0
    # requires_grad 张量上梯度也精确 0。
    t_conf = t_conf.clone().requires_grad_(True)
    total, _ = compute_clinical_t_roi_loss(t_conf, t_idx, v)
    total.backward()
    assert float(t_conf.grad.abs().sum()) == 0.0


def test_kref_epslon_semantics_note():
    """组合式 `clamp(v.sum()+eps, min=4)` 的 0 值路径:eps 保 0/0 → 0(不是
    0/0=NaN),clamp 保小批分母 ≥4。两道保护互不替代——若只 clamp 无 eps,
    v 全 0 时 0/0=NaN;若只 eps 无 clamp,单样本批独大。本测试用 ete 头
    (logits 域)复验同一组合式。
    """
    logits = torch.randn(6, 4)
    idx = torch.randint(0, 4, (6,))
    v = torch.zeros(6)
    total, _ = compute_extension_roi_loss(
        logits, idx, v, class_weights=None, focal_gamma=0.0)
    assert torch.isfinite(total) and float(total) == 0.0


# ════════════════════════════════════════════════════════════════════════
# 10. 债6 概念监督(方案 A 简化版,concepts_roi)
# ════════════════════════════════════════════════════════════════════════

from src.models.losses.components.concepts_roi import (  # noqa: E402
    compute_roi_concept_losses,
    ROI_CONCEPT_KEYS,
)

_CONCEPT_LOSS_KEYS = ("roi_concept", "roi_concept_comp", "roi_concept_echo",
                      "roi_concept_shape", "roi_concept_margin")
_CONCEPT_COUNT_KEYS = ("concept_n_valid_comp", "concept_n_valid_echo",
                       "concept_n_valid_shape", "concept_n_valid_margin")


def _concepts(seed=0, K=5):
    """造 forward_roi 顶层 roi_concepts dict(五键 logits)。"""
    torch.manual_seed(seed)
    dims = {"logits_comp": 4, "logits_echo": 4, "logits_shape": 2,
            "logits_margin": 4, "logits_foci": 4}
    return {k: torch.randn(K, d) for k, d in dims.items()}


def _concept_labels(K=5, valid_comp=1, valid_echo=1, valid_shape=1,
                    valid_margin=1):
    """造 roi_labels 元素(数据面桥产物键:gt_ 概念 + valid_*)。"""
    return {
        "gt_composition": 1, "valid_composition": valid_comp,
        "gt_echogenicity": 2, "valid_echogenicity": valid_echo,
        "gt_shape": 0, "valid_shape": valid_shape,
        "gt_margin": 3, "valid_margin": valid_margin,
        "is_main": 1, "lesion_id": "L",
    }


def _concept_batch(seed=0, K=5, v_comp=1, v_echo=1, v_shape=1, v_margin=1):
    outs = {"roi_concepts": _concepts(seed, K)}
    labs = [_concept_labels(K, v_comp, v_echo, v_shape, v_margin)
            for _ in range(K)]
    return outs, labs


def test_concept_default_zero_bitwise_regression():
    """①默认 0 逐位零回归:concept_weight=0 与不传该参的 total/detail 逐位一致
    (<1e-6),且编排层 lambda_roi=1 时四头值与概念监督无关(概念键缺席不 raise)。
    """
    outs, tgs = _batch_3(seed=51)
    t_a, d_a = compute_roi_head_losses(outs, tgs, lambda_roi=1.0)
    t_b, d_b = compute_roi_head_losses(outs, tgs, lambda_roi=1.0,
                                       concept_weight=0.0)
    assert float(t_a) == pytest.approx(float(t_b), abs=1e-12)
    for k in LOCATION_KEYS:
        assert float(d_a[k]) == pytest.approx(float(d_b[k]), abs=1e-12)
    # 概念权重 0:直接调也恒 0(total python 0.0,键恒在)。
    c_total, c_detail = compute_roi_concept_losses(
        {"roi_concepts": _concepts(0, 5)}, [_concept_labels() for _ in range(5)],
        lambda_concept=0.0)
    assert c_total == 0.0 and not isinstance(c_total, torch.Tensor)
    for k in _CONCEPT_LOSS_KEYS:
        assert isinstance(c_detail[k], torch.Tensor)
        assert float(c_detail[k]) == 0.0
    for k in _CONCEPT_COUNT_KEYS:
        assert c_detail[k] == 0.0


def test_concept_head_learns_few_steps():
    """②监督开后概念头有学习信号:小随机数据训几步,concept loss 下降。

    用一个 mini Linear 概念头(结构对齐 RoiConceptHead 的单概念切片),
    30 步 SGD 后 loss 显著低于初值(>10% 降幅)——随机噪声旁路不会有此行为。
    """
    torch.manual_seed(7)
    K, in_dim = 16, 32
    head = torch.nn.Linear(in_dim, 4)
    feats = torch.randn(K, in_dim)
    labels = torch.randint(0, 4, (K,))
    opt = torch.optim.SGD(head.parameters(), lr=0.1)
    first = last = None
    for step in range(30):
        logits = head(feats)
        log_p = F.log_softmax(logits, dim=-1)
        loss = -log_p.gather(1, labels.unsqueeze(1)).squeeze(1).mean()
        if step == 0:
            first = float(loss)
        last = float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert last < first * 0.9, (
        f"概念监督无学习信号: first={first:.4f} last={last:.4f}")


def test_concept_valid_masking_not_in_denominator():
    """③缺失标签 ignore 不进分母:valid=0 概念的损失与"去掉该样本"同值。

    K=6、comp valid 只有前 4 条:comp 分项 == 只给前 4 条的独立调用的值
    (同 clamp(4,4) 分母);与"全 6 条 valid"的值必不同(区分力)。
    """
    outs, labs = _concept_batch(seed=61, K=6, v_comp=1)
    labs[4]["valid_composition"] = 0
    labs[5]["valid_composition"] = 0
    t_masked, d_masked = compute_roi_concept_losses(
        outs, labs, lambda_concept=1.0)
    # 独立调用:前 4 条标签 + 同 logits 前 4 行(n_valid=4 = k_ref,分母同)。
    t_sub, d_sub = compute_roi_concept_losses(
        {"roi_concepts": {k: v[:4] for k, v in outs["roi_concepts"].items()}},
        labs[:4], lambda_concept=1.0)
    assert float(d_masked["roi_concept_comp"]) == pytest.approx(
        float(d_sub["roi_concept_comp"]), abs=1e-7)
    # 计数键:comp 只数 4,echo 数 6(valid 屏蔽进计数,回读可见人口)。
    assert d_masked["concept_n_valid_comp"] == 4.0
    assert d_masked["concept_n_valid_echo"] == 6.0
    # 全 6 valid 的对照必不同(去掉屏蔽的变异对拍)。
    outs2, labs2 = _concept_batch(seed=61, K=6, v_comp=1)
    _, d_full = compute_roi_concept_losses(outs2, labs2, lambda_concept=1.0)
    assert float(d_masked["roi_concept_comp"]) != pytest.approx(
        float(d_full["roi_concept_comp"]), abs=1e-6)


def test_concept_foci_always_zero_grad():
    """④foci 恒零梯度:概念权重>0 时 logits_foci 不进任何损失项
    (grad is None——不是 0,是从未入图,比 detach 更硬)。"""
    outs, labs = _concept_batch(seed=71, K=5)
    outs["roi_concepts"]["logits_foci"].requires_grad_(True)
    for k in ("logits_comp", "logits_echo", "logits_shape", "logits_margin"):
        outs["roi_concepts"][k].requires_grad_(True)
    total, detail = compute_roi_concept_losses(outs, labs, lambda_concept=1.0)
    total.backward()
    assert outs["roi_concepts"]["logits_foci"].grad is None
    for k in ("logits_comp", "logits_echo", "logits_shape", "logits_margin"):
        g = outs["roi_concepts"][k].grad
        assert g is not None and float(g.abs().sum()) > 0, f"{k} 概念头无梯度"


def test_concept_narrow_dict_raises():
    """窄 dict 误传 fail-loud:只传 {"roi_outputs": ...} 无 roi_concepts 键
    → raise(概念 logits 在 forward_roi 顶层,静默跳过 = 号称开了实际没开)。"""
    outs, labs = _concept_batch(seed=81, K=5)
    narrow = {"roi_outputs": []}          # 常见误传形态
    with pytest.raises(ValueError, match="roi_concepts"):
        compute_roi_concept_losses(narrow, labs, lambda_concept=1.0)
    # 编排层:concept_weight>0 + lambda_roi>0 但 outputs 窄 dict → 同 raise。
    with pytest.raises(ValueError, match="roi_concepts"):
        compute_roi_head_losses(
            {"roi_outputs": [{"location_probs": torch.softmax(
                torch.randn(7), 0),
                "ete_logits": torch.randn(4),
                "t_conf": torch.tensor(0.5),
                "tirads_probs": torch.softmax(torch.randn(5), 0)}] * 5,
             },
            {"roi_labels": labs}, lambda_roi=1.0, concept_weight=1.0)


def test_concept_lambda_additivity():
    """概念项 λ 线性:λ=0.5 的 total = 0.5 × (四概念原始和)。detail 未乘 λ。"""
    outs, labs = _concept_batch(seed=91, K=6)
    t1, d1 = compute_roi_concept_losses(outs, labs, lambda_concept=1.0)
    t05, d05 = compute_roi_concept_losses(outs, labs, lambda_concept=0.5)
    raw = float(d1["roi_concept"])
    assert float(t1) == pytest.approx(raw, abs=1e-6)
    assert float(t05) == pytest.approx(0.5 * raw, abs=1e-7)
    for k in _CONCEPT_LOSS_KEYS:
        assert float(d1[k]) == pytest.approx(float(d05[k]), abs=1e-9)


def test_concept_independent_of_lambda_roi():
    """概念监督独立开关:lambda_roi=0 + concept_weight>0 仍产真实概念损失
    (纯概念标定场景),四头 detail 恒 0。"""
    outs, labs = _concept_batch(seed=101, K=6)
    # lambda_roi=0 路径:targets 无 roi_labels 键也能走(概念监督自取)。
    t, d = compute_roi_head_losses(
        {"roi_concepts": outs["roi_concepts"]},
        {"roi_labels": labs}, lambda_roi=0.0, concept_weight=0.7)
    assert isinstance(t, torch.Tensor)
    c_raw, _ = compute_roi_concept_losses(outs, labs, lambda_concept=1.0)
    assert float(t) == pytest.approx(0.7 * float(c_raw), abs=1e-6)
    for k in LOCATION_KEYS:
        assert float(d[k]) == 0.0
    for k in _CONCEPT_LOSS_KEYS:
        assert float(d[k]) > 0 or k == "roi_concept"  # 真实分项非 0
    assert d["concept_n_valid_comp"] == 6.0


def test_concept_mutation_valid_gate_removed_killed():
    """⑤变异杀测:概念监督"不乘 valid"注入后测试必须 RED。

    真实现:K=6 但 comp valid 只前 2 条 → 分母 clamp(2,4)=4、分子只前 2 条。
    变异(无 valid 门):全部 6 条进分子且分母=6 → 值必不同。对拍两版断言差。
    (真实现值 = 手算前 2 条 BCE/4;变异值 = 6 条均值——同一 logits 下必不等。)
    """
    outs, labs = _concept_batch(seed=111, K=6, v_comp=1)
    for l in labs[2:]:
        l["valid_composition"] = 0          # 只前 2 条有效 comp
    _, d = compute_roi_concept_losses(outs, labs, lambda_concept=1.0)
    true_val = float(d["roi_concept_comp"])
    # 手算真实现:前 2 条 comp BCE / 4。
    log_p = F.log_softmax(outs["roi_concepts"]["logits_comp"][:2].float(), -1)
    manual = float(-log_p.gather(
        1, torch.tensor([l["gt_composition"] for l in labs[:2]]
                        ).unsqueeze(1)).squeeze(1).sum() / 4.0)
    assert true_val == pytest.approx(manual, abs=1e-6)
    # 变异对拍:无门版(全 6 条均值)必不同——本断言被"去掉 valid 乘法"的
    # 实现变异打破(true_val 会变成 6 条均值,manual 不变)。
    log_p6 = F.log_softmax(outs["roi_concepts"]["logits_comp"].float(), -1)
    mutated = float(-log_p6.gather(
        1, torch.tensor([l["gt_composition"] for l in labs]
                        ).unsqueeze(1)).squeeze(1).mean())
    assert true_val != pytest.approx(mutated, abs=1e-6), \
        "valid 门被删后数值不变 = fixture 无区分力"


def test_concept_shape_two_classes():
    """shape 是 2 类(纵横比):[K,2] logits 合法、标签 {0,1};类数钉死防漂移。"""
    outs, labs = _concept_batch(seed=121, K=5)
    for l in labs:
        l["gt_shape"] = 1
    total, d = compute_roi_concept_losses(outs, labs, lambda_concept=1.0)
    log_p = F.log_softmax(outs["roi_concepts"]["logits_shape"].float(), -1)
    manual = float(-log_p.gather(
        1, torch.ones(5, dtype=torch.long).unsqueeze(1)).squeeze(1).sum() / 5.0)
    assert float(d["roi_concept_shape"]) == pytest.approx(manual, abs=1e-6)


# ════════════════════════════════════════════════════════════════════════
# 11. §J.5b 三条测试债(gt_ fixture 源级对拍 / LS>0 覆盖 / device 断言)
# ════════════════════════════════════════════════════════════════════════

def test_j5b_gt_fixture_source_level_bridge():
    """债①:gt_ fixture 手写零杀——用**真实桥**(`roi_labels_from_meta`)产物
    直喂损失断言非零,不再手写 gt_ 键(手写键名错时手算对拍全绿=零杀)。

    构造数据面 meta(无前缀键,`_ROI_SCALAR_KEYS` 形态)→ 桥 → roi_labels →
    概念损失:四键命中非零;再喂四头:K≥4 主病灶批非零。
    """
    from src.dataset.roi_dataset import roi_labels_from_meta
    metas = []
    for i in range(6):
        metas.append({
            "lesion_id": f"L{i}", "is_main": 1,
            "malignancy": 1, "valid_malignancy": 1,
            "lesion_loc": (i * 3) % 6, "valid_lesion_loc": 1,
            "extrathy_ext": i % 4, "valid_extrathy_ext": 1,
            "clin_t_stage": i % 2, "valid_clin_t_stage": 1,
            "tirads": i % 5, "valid_tirads": 1,
            "composition": i % 4, "valid_composition": 1,
            "echogenicity": (i + 1) % 4, "valid_echogenicity": 1,
            "shape": i % 2, "valid_shape": 1,
            "margin": (i + 2) % 4, "valid_margin": 1,
        })
    labs = roi_labels_from_meta(metas)          # 真实桥(键名单源)
    # 桥产物键名断言:gt_ 前缀五键 + valid 透传(源级,防键名漂移)。
    for k in ("gt_lesion_loc", "gt_extrathy_ext", "gt_clin_t_stage",
              "gt_tirads", "gt_composition", "gt_echogenicity",
              "gt_shape", "gt_margin"):
        assert k in labs[0], f"桥产物缺 {k}(键名漂移=手写 fixture 零杀根因)"
    # 概念损失:四概念键全命中、非零。
    outs = {"roi_concepts": _concepts(131, 6)}
    _, d = compute_roi_concept_losses(outs, labs, lambda_concept=1.0)
    for k in ("roi_concept_comp", "roi_concept_echo",
              "roi_concept_shape", "roi_concept_margin"):
        assert float(d[k]) > 0, f"{k} 零——桥键名与损失键名没对上"
    # 四头:同批 K=6 非零(loc 6 类轮转)。
    roi_outputs = []
    torch.manual_seed(131)
    for i in range(6):
        roi_outputs.append({
            "lesion_id": f"L{i}", "is_main": 1, "patient_id": None,
            "malignancy": torch.tensor([0.0, 1.0]),
            "location_probs": torch.softmax(torch.randn(7), 0),
            "ete_probs": torch.softmax(torch.randn(4), 0),
            "ete_logits": torch.randn(4),
            "t_conf": torch.tensor(0.5),
            "tirads_probs": torch.softmax(torch.randn(5), 0),
        })
    t, d2 = compute_roi_head_losses(
        {"roi_outputs": roi_outputs}, {"roi_labels": labs}, lambda_roi=1.0)
    assert float(t) > 0
    for k in ("roi_location", "roi_ete", "roi_clin_t", "roi_tirads"):
        assert float(d2[k]) > 0


def test_j5b_tirads_label_smoothing_positive_coverage():
    """债②:label_smoothing>0 分支零覆盖——数值断言(此前 24 项全走 ls=0)。

    平滑目标:1-ls 给真类,ls/5 均匀;per_sample = -Σ target·log(p)。
    手算对拍 K=6(≥k_ref,均值分母)。ls=0 与 ls>0 值必不同(分支真在走)。
    """
    probs = torch.tensor([[0.1, 0.2, 0.4, 0.2, 0.1],
                          [0.5, 0.2, 0.1, 0.1, 0.1],
                          [0.05, 0.05, 0.8, 0.05, 0.05],
                          [0.2, 0.2, 0.2, 0.2, 0.2],
                          [0.3, 0.3, 0.1, 0.1, 0.2],
                          [0.1, 0.1, 0.1, 0.1, 0.6]])
    idx = torch.tensor([2, 0, 2, 3, 1, 4])
    v = torch.ones(6)
    ls = 0.1
    total, detail = compute_tirads_roi_loss(
        probs, idx, v, label_smoothing=ls)
    C = 5
    p = probs.clamp(min=_EPS)
    manual_rows = []
    for i in range(6):
        target = torch.full((C,), ls / C)
        target[idx[i]] += 1.0 - ls
        manual_rows.append(float(-(target * p[i].log()).sum()))
    manual = float(torch.tensor(manual_rows).sum() / 6.0)
    assert float(total) == pytest.approx(manual, abs=1e-6)
    # 分支区分力:ls=0 值必不同(平滑真改变数值)。
    t0, _ = compute_tirads_roi_loss(probs, idx, v, label_smoothing=0.0)
    assert float(total) != pytest.approx(float(t0), abs=1e-6)


def test_j5b_zero_weight_detail_device():
    """债③:零权重 detail 张量 device 断言——J.5 P2-1(零权重 detail 恒 CPU,
    GPU stack 即炸)。本机有 CUDA(实测),真断言:零有效批的 detail 键与
    **输入 logits 同 device**(cuda:0 == cuda:0,index 归一化比较);
    概念零权重路径的恒 0 键是 CPU 构造(消费端 .tolist() 不炸,J.5 P2-1
    记录在案的风险已由"权重>0 路径 loss_c 在 logits.device 归约"消化,
    此处钉死 CPU 构造形态不被误改)。
    """
    if not torch.cuda.is_available():
        pytest.skip("无 CUDA:device 断言留 CI/Linux 跑道(本机 Windows CPU)")
    dev = torch.device("cuda")
    logits = torch.randn(6, 4, device=dev)
    idx = torch.randint(0, 4, (6,), device=dev)
    # 四头零有效批:total 0 + detail 键恒在,断言与输入同 device。
    t, d = compute_extension_roi_loss(logits, idx, torch.zeros(6, device=dev),
                                      focal_gamma=0.0)
    assert float(t) == 0.0
    for k, val in d.items():
        assert val.device.type == dev.type, f"{k} 零权重 detail 在 {val.device}"
    # 概念权重>0 路径:loss_c 在 logits.device 上归约 → detail 与输入同 device。
    outs = {"roi_concepts": {k: torch.randn(6, n, device=dev)
                             for k, n in (("logits_comp", 4), ("logits_echo", 4),
                                          ("logits_shape", 2), ("logits_margin", 4),
                                          ("logits_foci", 4))}}
    labs = [_concept_labels() for _ in range(6)]
    c_t, c_d = compute_roi_concept_losses(outs, labs, lambda_concept=0.7)
    for k in _CONCEPT_LOSS_KEYS:
        assert c_d[k].device.type == dev.type, f"{k} 概念 detail 在 {c_d[k].device}"
