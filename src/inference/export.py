"""Export prediction artifacts, rejection JSON, and Grad-CAM overlays."""

import json
import os

import cv2
import torch.nn as nn

from src.utils.grad_cam import GradCAM, overlay_heatmap


def _write_json_atomic(path, payload):
    """原子写 JSON:先写临时文件再 `os.replace`,避免 SIGTERM/OOM-kill 恰好在
    `json.dump` 写盘期间到达时,在磁盘上留下半写/截断的 `result.json`。

    背景(2026-07-12 Gate2 复审发现):entrypoint.sh 的崩溃兜底逻辑只检查
    `result.json` 是否存在,不校验内容是否完整——半写文件会被误判为"本次已
    正常产出",比文件完全不存在更隐蔽危险。`os.replace` 在同目录内是原子操作
    (POSIX `rename(2)` / Windows `MoveFileEx` 均保证同文件系统内原子性),
    要么整个文件原子出现要么不出现,不存在"写到一半"的中间态可被读到。
    """
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def save_prediction_artifacts(output_dir, mask_resized, report_txt, out_json):
    """Save mask image, clinical report text, and result JSON."""
    with open(
        os.path.join(output_dir, "clinical_report.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(report_txt)

    _write_json_atomic(os.path.join(output_dir, "result.json"), out_json)

    mask_path = os.path.join(output_dir, "mask_predicted.png")
    if not cv2.imwrite(mask_path, mask_resized):
        raise RuntimeError(f"cv2.imwrite failed: {mask_path} (check permissions or non-ASCII path)")


def save_rejection_json(output_dir, rejection_json):
    """Save gatekeeper rejection result to result.json."""
    _write_json_atomic(os.path.join(output_dir, "result.json"), rejection_json)


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


def maybe_save_gradcam(model, img_tensor, img_np, output_dir, branch="cls"):
    """收敛 predict / video_predict 共用的 Grad-CAM 落盘+存在性判定。

    在 output_dir 下生成 gradcam_cls.png 并返回 (gradcam_done, gradcam_path)。
    gradcam_done=1 仅当文件确实写出(save_gradcam_overlay 内部失败兜底不抛)。
    报告附注文字因单图/关键帧语义不同,由各调用方自行追加。
    """
    gradcam_path = os.path.join(output_dir, "gradcam_cls.png")
    save_gradcam_overlay(model, img_tensor, img_np, gradcam_path, branch=branch)
    gradcam_done = 1 if os.path.exists(gradcam_path) else 0
    return gradcam_done, gradcam_path
