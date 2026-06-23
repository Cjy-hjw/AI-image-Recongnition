"""Decode model outputs and compute TI-RADS breakdown."""

import numpy as np
import torch

from src.utils.report_generator import calculate_tirads_points


def decode_predictions(outputs: dict) -> dict:
    """Extract all prediction fields from raw model outputs.

    Returns dict with keys:
        pred_mask, mask_binary, comp_idx, echo_idx, shape_idx, margin_idx,
        foci_prob, mal_prob
    """
    pred_mask = torch.sigmoid(outputs["mask"].float())[0, 0].cpu().numpy()
    mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255

    comp_idx = torch.argmax(outputs["composition"], dim=1)[0].item()
    echo_idx = torch.argmax(outputs["echogenicity"], dim=1)[0].item()
    shape_idx = torch.argmax(outputs["shape"], dim=1)[0].item()
    margin_idx = torch.argmax(outputs["margin"], dim=1)[0].item()
    foci_prob = torch.sigmoid(outputs["foci"])[0].cpu().numpy()
    mal_prob = torch.softmax(outputs["malignancy"].float(), dim=1)[0, 1].item()

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
