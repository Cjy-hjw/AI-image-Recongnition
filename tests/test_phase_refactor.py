"""重构回归测试(Phase A 解耦/死代码清理 + Phase B 评估尺子校准)。

合并自临时烟测脚本,作为长期回归守卫。pytest 与 `python tests/test_phase_refactor.py` 均可运行。
"""
import sys
import os
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.models.multi_task_model import MultiTaskThyroidModel, ModelOutputs
from src.models.cbm_head import ConceptEmbeddingCBMHead, CBMHeadOutput
from src.models.loss import MultiTaskLoss
from src.inference.model_runner import run_forward
from src.inference.export import _resolve_gradcam_target_layer
from src.inference.io import _PHI_FIELDS
from src.training.loop import validate_step


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


def test_phi_fields_expanded():
    assert len(_PHI_FIELDS) >= 15
    for must in ["PatientName", "AccessionNumber", "DeviceSerialNumber",
                 "OperatorsName", "StudyDate"]:
        assert must in _PHI_FIELDS


def test_validate_step_fixed_gamma_prior():
    src = inspect.getsource(validate_step.validate_batch)
    assert "gamma_prior=1.0" in src
    assert "(epoch - 20)" not in src


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
