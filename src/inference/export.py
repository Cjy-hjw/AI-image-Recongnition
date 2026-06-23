"""Export prediction artifacts, rejection JSON, and Grad-CAM overlays."""

import json
import os

import cv2
import torch.nn as nn

from src.utils.grad_cam import GradCAM, overlay_heatmap


def save_prediction_artifacts(output_dir, mask_resized, report_txt, out_json):
    """Save mask image, clinical report text, and result JSON."""
    cv2.imwrite(os.path.join(output_dir, "mask_predicted.png"), mask_resized)

    with open(
        os.path.join(output_dir, "clinical_report.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(report_txt)

    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(out_json, f, indent=2)


def save_rejection_json(output_dir, rejection_json):
    """Save gatekeeper rejection result to result.json."""
    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(rejection_json, f, indent=2)


def _resolve_gradcam_target_layer(model, branch="cls"):
    """Resolve the last conv layer for Grad-CAM on the requested branch.

    双主干架构:`cls_backbone`(EfficientNet-B2)负责良恶性分类,`seg_backbone`
    (ConvNeXt-Nano)负责分割。Grad-CAM 默认解释恶性预测(target_class=1),故针对
    分类分支 `cls_backbone`;传 branch="seg" 可改为解释分割主干。
    """
    attr = "cls_backbone" if branch == "cls" else "seg_backbone"
    backbone = getattr(model, attr, None)
    if backbone is None:
        raise AttributeError(f"Model has no {attr} for Grad-CAM (branch={branch})")
    if hasattr(backbone, "blocks"):
        last_block = backbone.blocks[-1]
        if isinstance(last_block, nn.Sequential) and len(last_block) > 0:
            return last_block[-1]
    for module in reversed(list(backbone.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise RuntimeError(f"No Conv2d layer found for Grad-CAM target on {attr}")


def save_gradcam_overlay(model, img_tensor, img_np, output_path, branch="cls"):
    """Generate and save a Grad-CAM heatmap overlay.

    Discovers the last convolutional layer of the requested backbone branch
    ("cls" for malignancy classification, "seg" for segmentation).
    """
    gradcam = None
    try:
        target_layer = _resolve_gradcam_target_layer(model, branch)
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate_heatmap(img_tensor, target_class=1)
        overlay = overlay_heatmap(img_np, heatmap)
        cv2.imwrite(output_path, overlay)
        print(f"--> Saved Grad-CAM heatmap to {output_path}")
    except Exception as e:
        print(f"[Grad-CAM Error] Could not generate heatmap: {e}")
    finally:
        if gradcam is not None:
            gradcam.remove_hooks()
