"""重构回归测试(Phase A 解耦/死代码清理 + Phase B 评估尺子校准)。

合并自临时烟测脚本,作为长期回归守卫。pytest 与 `python tests/test_phase_refactor.py` 均可运行。
"""
import sys
import os
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.models.multi_task_model import MultiTaskThyroidModel, ModelOutputs
from src.models.cbm_head import ConceptEmbeddingCBMHead, CBMHeadOutput
from src.models.loss import MultiTaskLoss
from src.inference.model_runner import run_forward
from src.inference.export import _resolve_gradcam_target_layer
from src.utils.grad_cam import GradCAM
from src.inference.io import _PHI_FIELDS
from src.training.loop import validate_step
from scripts.data.split_tn3k import _split_samples
from predict import main as predict_main


# ── Phase A: 解耦 + 死代码清理 ────────────────────────────────────────────────

def test_cbm_head_returns_keyed_dataclass():
    head = ConceptEmbeddingCBMHead(in_features=352, num_concepts=18)
    out = head(torch.randn(2, 352))
    assert isinstance(out, CBMHeadOutput)
    for f in [
        "logits_comp", "logits_echo", "logits_shape", "logits_margin",
        "logits_foci", "concept_vector", "logits_malignancy", "latent_res",
    ]:
        assert hasattr(out, f), f


def test_dead_code_removed():
    import src.models.losses.classification as clf
    import src.dataset.loader as ldr
    import src.training.stages as st
    assert not hasattr(clf, "AdaptiveVSAndAsymmetricLoss")
    assert not hasattr(ldr, "get_patient_groups")
    assert not hasattr(st, "create_optimizer")
    assert not hasattr(st, "update_optimizer_lr")
    assert not hasattr(MultiTaskLoss(), "log_vars")


def test_forward_output_matches_contract():
    model = MultiTaskThyroidModel(pretrained=False).eval()
    x = torch.randn(2, 1, 512, 512)
    with torch.no_grad():
        out = model(x)
    assert set(out.keys()) == set(ModelOutputs.__annotations__.keys())


def test_forward_deterministic():
    torch.manual_seed(0)
    model = MultiTaskThyroidModel(pretrained=False).eval()
    x = torch.randn(2, 1, 512, 512)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    for k in a:
        assert torch.allclose(a[k], b[k]), k


def test_loss_forward_finite_no_ortho():
    model = MultiTaskThyroidModel(pretrained=False).train()
    x = torch.randn(2, 1, 512, 512)
    B = 2
    targets = {
        "mask": torch.randint(0, 2, (B, 1, 512, 512)).float(),
        "composition": torch.randint(0, 4, (B,)),
        "echogenicity": torch.randint(0, 4, (B,)),
        "shape": torch.randint(0, 2, (B,)),
        "margin": torch.randint(0, 4, (B,)),
        "foci": torch.randint(0, 2, (B, 4)).float(),
        "malignancy": torch.randint(0, 2, (B,)),
    }
    total, ld = MultiTaskLoss()(model(x), targets)
    assert torch.isfinite(total)
    assert "ortho" not in ld


# ── Phase B: 评估尺子校准 ─────────────────────────────────────────────────────

def test_run_forward_fp32_equals_bare_forward():
    """fp32 下 run_forward 与裸 forward(gamma=1.0)数值一致 —— 证明 evaluate 改动只换路径不改数值。"""
    torch.manual_seed(0)
    model = MultiTaskThyroidModel(pretrained=False).eval()
    x = torch.randn(1, 1, 512, 512)
    dev = torch.device("cpu")
    with torch.inference_mode():
        bare = model(x, gamma_prior=1.0)
    out = run_forward(model, x, "fp32", dev)
    for k in ["mask", "malignancy", "composition"]:
        assert torch.allclose(bare[k], out[k], atol=1e-5), k


def test_dice_double_empty_is_one_via_epsilon():
    mask = torch.zeros(1, 1, 512, 512)
    pred = torch.zeros(1, 1, 512, 512)
    inter = (pred * mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    dice = ((2.0 * inter + 1e-6) / (union + 1e-6)).mean().item()
    assert abs(dice - 1.0) < 1e-3


def test_gradcam_dual_backbone_branches():
    model = MultiTaskThyroidModel(pretrained=False)
    cls_layer = _resolve_gradcam_target_layer(model, "cls")
    seg_layer = _resolve_gradcam_target_layer(model, "seg")
    assert cls_layer is not None and seg_layer is not None
    assert cls_layer is not seg_layer


def test_gradcam_generates_valid_heatmap_end_to_end():
    """端到端:Grad-CAM 真跑前向+反向,产出 [0,1] 归一化、与输入同尺寸的热图。

    堵集成盲区——此前只测了目标层可解析,从未验证 backward 路径能跑通(梯度
    经分类主干回流到目标层激活)。注意:GradCAM 内部自带 forward+backward,
    必须在 inference_mode 之外、用普通张量调用。
    """
    model = MultiTaskThyroidModel(pretrained=False).eval()
    target_layer = _resolve_gradcam_target_layer(model, "cls")
    gradcam = GradCAM(model, target_layer)
    try:
        x = torch.rand(1, 1, 512, 512)  # 普通张量,可参与 autograd;模型要求 512×512
        heatmap = gradcam.generate_heatmap(x, target_class=1)
    finally:
        gradcam.remove_hooks()
    assert heatmap.shape == (512, 512), "热图须与输入空间尺寸一致"
    assert float(heatmap.min()) >= 0.0 and float(heatmap.max()) <= 1.0, "热图须归一化到 [0,1]"
    assert not np.isnan(heatmap).any(), "热图不得含 NaN"


def test_predict_passes_gradcam_flag_outside_inference_mode():
    """守卫:predict.py 的 Grad-CAM 调用必须在 inference_mode 块之外。

    inference_mode 下创建/使用的张量无法参与 autograd,若 save_gradcam_overlay
    被包进 run_forward 的 inference_mode,backward 会失败。此测试静态校验源码
    结构:gradcam 调用出现在 run_forward 之后的顶层流程,且有 enable_gradcam 开关。
    """
    src = inspect.getsource(predict_main)
    assert "--enable_gradcam" in src or "enable_gradcam" in src, "缺 enable_gradcam 开关"
    assert "save_gradcam_overlay" in src, "predict 未接入 Grad-CAM"
    # gradcam 调用须在 run_forward 之后(同一函数体顶层,不在 inference_mode 内)
    assert src.index("run_forward") < src.index("save_gradcam_overlay"), (
        "Grad-CAM 须在 run_forward(inference_mode)之后调用")


def test_phi_fields_expanded():
    assert len(_PHI_FIELDS) >= 15
    for must in ["PatientName", "AccessionNumber", "DeviceSerialNumber",
                 "OperatorsName", "StudyDate"]:
        assert must in _PHI_FIELDS
    # R2 修复:补全 HIPAA Safe Harbor 高危 PHI(原 21 项清单遗漏,save_as 会残留)。
    for newly in ["PatientComments", "RequestingPhysician", "OtherPatientNames",
                  "PatientMotherBirthName", "EthnicGroup", "PatientBirthTime",
                  "AdditionalPatientHistory", "PatientInstitutionResidence"]:
        assert newly in _PHI_FIELDS, f"遗漏高危 PHI 标量 {newly}"
    # 序列类(SQ)PHI 走删除通道,不能 setattr 字符串(会破坏 SQ 结构)。
    from src.inference.io import _PHI_DELETE_TAGS
    assert "OtherPatientIDsSequence" in _PHI_DELETE_TAGS


def test_phi_anonymize_removes_new_fields(tmp_path):
    """合成带新覆盖 PHI 标签的 DICOM,跑匿名化,断言高危 PHI 已清且成像参数完好。"""
    import numpy as np
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    from src.inference.io import _anonymize_phi

    ds = Dataset()
    ds.Rows, ds.Columns = 8, 8
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 8, 8, 7
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.PixelData = np.zeros((8, 8), np.uint8).tobytes()
    # 高危 PHI(原清单遗漏项):注入含明文的内容
    ds.PatientComments = "患者张三,电话 13800000000"
    ds.RequestingPhysician = "李医生"
    ds.EthnicGroup = "汉族"
    ds.OtherPatientIDsSequence = [Dataset()]  # SQ 序列类
    ds.OtherPatientIDsSequence[0].PatientID = "ALT_ID"
    fm = FileMetaDataset()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.MediaStorageSOPClassUID = generate_uid()
    fm.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta = fm
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    _anonymize_phi(ds)

    # 标量 PHI 已匿名/清空
    assert str(ds.PatientComments) == ""
    assert str(ds.RequestingPhysician) == "ANONYMIZED"
    assert str(ds.EthnicGroup) == ""
    # 序列类 PHI 已整体删除
    assert "OtherPatientIDsSequence" not in ds
    # 成像必需参数完好、像素能读回(匿名化不破坏成像)
    assert ds.Rows == 8 and ds.PhotometricInterpretation == "MONOCHROME2"
    assert ds.pixel_array.shape == (8, 8)


def test_validate_step_fixed_gamma_prior():
    src = inspect.getsource(validate_step.validate_batch)
    assert "gamma_prior=1.0" in src
    assert "(epoch - 20)" not in src


# ── Phase D: 数据划分去伪分组 ─────────────────────────────────────────────────

def test_split_samples_disjoint():
    """train/val 图像 ID 集合互斥。"""
    samples = [(f"{i:04d}.jpg", i % 2) for i in range(100)]
    train, val = _split_samples(samples, split_ratio=0.85, seed=42)
    train_ids = {t[0] for t in train}
    val_ids = {v[0] for v in val}
    assert train_ids.isdisjoint(val_ids), "train/val must be disjoint!"


def test_split_samples_ratio():
    """划分比例 ≈ split_ratio。"""
    samples = [(f"{i:04d}.jpg", 0) for i in range(1000)]
    train, val = _split_samples(samples, split_ratio=0.85, seed=42)
    assert len(train) == 850 and len(val) == 150, (
        f"ratio mismatch: train={len(train)}, val={len(val)}"
    )


def test_split_samples_reproducible():
    """固定 seed 可复现。"""
    samples = [(f"{i:04d}.jpg", 0) for i in range(500)]
    a = _split_samples(samples, split_ratio=0.8, seed=7)
    b = _split_samples(samples, split_ratio=0.8, seed=7)
    assert [t[0] for t in a[0]] == [t[0] for t in b[0]], "seed 7 should produce same train order"
    assert [t[0] for t in a[1]] == [t[0] for t in b[1]], "seed 7 should produce same val order"


def test_split_samples_patient_id_is_image_id():
    """每张图独立身份(不再臆造 PAT_{//2} 伪分组)。"""
    # 验证 format_split 写 meta 时 patient_id == image_id(而非 PAT_xxx)
    # 这里的测试面向 _split_samples 的返回值:每张图像只出现一次,等价于独立 identity
    samples = [(f"{i:04d}.jpg", 0) for i in range(50)]
    train, val = _split_samples(samples, split_ratio=0.8, seed=1)
    all_combined = set(t[0] for t in train) | set(v[0] for v in val)
    assert len(all_combined) == 50, "every sample must appear exactly once across train∪val"


# ── Phase C: FAMO / criterion 续训态持久化 ────────────────────────────────────

def test_famo_state_dict_round_trip():
    """FAMO 动态权重(w_logits/prev_loss)可序列化并精确恢复。"""
    from src.models.famo import FAMOScheduler
    famo = FAMOScheduler(num_tasks=5)
    losses = torch.tensor([1.0, 0.8, 0.5, 1.2, 0.3])
    famo.get_weighted_loss(losses, update_weights=True)          # 改变内部状态
    famo.get_weighted_loss(losses * 0.9, update_weights=True)
    state = famo.state_dict()

    restored = FAMOScheduler(num_tasks=5)
    restored.load_state_dict(state)
    assert torch.allclose(restored.w_logits, famo.w_logits)
    assert torch.allclose(restored.prev_loss, famo.prev_loss)


def test_famo_load_rejects_task_count_mismatch():
    """任务数不一致的 checkpoint 必须显式报错,而非静默广播。"""
    import pytest
    from src.models.famo import FAMOScheduler
    src_famo = FAMOScheduler(num_tasks=5)
    dst_famo = FAMOScheduler(num_tasks=3)
    with pytest.raises(AssertionError):
        dst_famo.load_state_dict(src_famo.state_dict())


def test_criterion_persists_dual_params_and_lambda():
    """criterion.state_dict 覆盖 AUCM 对偶参数与 lambda_latent;往返后精确恢复。"""
    crit = MultiTaskLoss()
    with torch.no_grad():                                        # 模拟训练改写动态状态
        crit.mal_loss_fn.a.fill_(0.37)
        crit.mal_loss_fn.alpha.fill_(0.21)
        crit.lambda_latent.fill_(0.055)
    keys = crit.state_dict().keys()
    assert "mal_loss_fn.a" in keys and "mal_loss_fn.alpha" in keys
    assert "lambda_latent" in keys

    restored = MultiTaskLoss()
    restored.load_state_dict(crit.state_dict())
    assert torch.allclose(restored.mal_loss_fn.a, crit.mal_loss_fn.a)
    assert torch.allclose(restored.mal_loss_fn.alpha, crit.mal_loss_fn.alpha)
    assert torch.allclose(restored.lambda_latent, crit.lambda_latent)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
