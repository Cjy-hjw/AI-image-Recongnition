"""Decode model outputs and compute TI-RADS breakdown."""

import numpy as np
import torch

from src.utils.report_generator import calculate_tirads_points


def decode_predictions(outputs: dict, calibrator=None) -> dict:
    """Extract all prediction fields from raw model outputs.

    Args:
        outputs: 原始模型输出(各头 logits)。
        calibrator: 可选 `TemperatureScaler`。每个分类头在 softmax/sigmoid **之前**
            先经 `calibrator.apply(task, logits)` 做温度缩放。为 None 或全 T=1.0 时
            **无任何行为变化**(逐位等同无校准)。argmax 预测类别不受 T 影响。

    Returns dict with keys:
        pred_mask, mask_binary, comp_idx, echo_idx, shape_idx, margin_idx,
        foci_prob, mal_prob
    """
    def _cal(task, logits):
        return calibrator.apply(task, logits) if calibrator is not None else logits

    # mask 为分割,不走温度缩放(分割校准是另一套,本期不碰)。
    pred_mask = torch.sigmoid(outputs["mask"].float())[0, 0].cpu().numpy()
    mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255

    # argmax 对温度缩放不变(T>0 单调),保留校准以与概率刻度一致、便于上层取置信度。
    # 各头统一 .float():AMP fp16 下温度除法 logits/T 在半精度有舍入损失,先升 fp32 再缩放/softmax,
    # 与 malignancy 的精度路径对齐(reviewer 建议),不改 argmax 结果。
    comp_idx = torch.argmax(_cal("composition", outputs["composition"].float()), dim=1)[0].item()
    echo_idx = torch.argmax(_cal("echogenicity", outputs["echogenicity"].float()), dim=1)[0].item()
    shape_idx = torch.argmax(_cal("shape", outputs["shape"].float()), dim=1)[0].item()
    margin_idx = torch.argmax(_cal("margin", outputs["margin"].float()), dim=1)[0].item()
    foci_prob = torch.sigmoid(_cal("foci", outputs["foci"].float()))[0].cpu().numpy()
    mal_prob = torch.softmax(_cal("malignancy", outputs["malignancy"].float()), dim=1)[0, 1].item()

    return {
        "pred_mask": pred_mask,
        "mask_binary": mask_binary,
        "comp_idx": comp_idx,
        "echo_idx": echo_idx,
        "shape_idx": shape_idx,
        "margin_idx": margin_idx,
        "foci_prob": foci_prob,
        "mal_prob": mal_prob,
    }


def build_tirads_breakdown(comp_idx, echo_idx, shape_idx, margin_idx, foci_prob):
    """Compute TI-RADS point breakdown using calculate_tirads_points."""
    return calculate_tirads_points(
        comp_idx, echo_idx, shape_idx, margin_idx, foci_prob
    )
