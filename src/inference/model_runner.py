"""Model construction, preprocessing, and forward pass with precision control."""

import os
import cv2
import torch

from src.models.multi_task_model import MultiTaskThyroidModel


def build_model(config: dict, weights: str, device: torch.device):
    """Build MultiTaskThyroidModel and load weights if available."""
    model = MultiTaskThyroidModel(
        backbone_name=config["model"]["backbone"],
        pretrained=False,
        config=config,
    ).to(device)
    if os.path.exists(weights):
        # weights_only=True:best_model.pth 按设计是纯 model state_dict(续训态另存
        # training_state.pth),torch≥2.6 下安全加载并消除 FutureWarning。
        model.load_state_dict(
            torch.load(weights, map_location=device, weights_only=True), strict=False)
        print(f"Loaded model weights from {weights}")
    else:
        print(
            "[WARNING] Trained checkpoint not found. Using default initialized weights."
        )
    model.eval()
    return model


def preprocess_image(img_np, img_size: int, device: torch.device):
    """Resize grayscale image and convert to normalized tensor [B,1,H,W]."""
    img_resized = cv2.resize(img_np, (img_size, img_size))
    img_tensor = (
        torch.tensor(img_resized, dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
        / 255.0
    )
    return img_tensor


def run_forward(model, img_tensor, precision: str, device: torch.device):
    """Run model forward pass with inference_mode, autocast, and clamp [-15,15].

    Args:
        precision: one of "fp32", "fp16", "bf16".
    """
    device_type = device.type
    # CPU does not support float16 autocast — fallback to float32
    is_fp16_on_cpu = device_type == "cpu" and precision == "fp16"

    autocast_ctx = torch.autocast(
        device_type=device_type,
        enabled=(precision in ["fp16", "bf16"] and not is_fp16_on_cpu),
        dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
    )
    if is_fp16_on_cpu:
        print(
            "[WARNING] float16 autocast is not supported on CPU. "
            "Falling back to float32 precision for stability."
        )

    with torch.inference_mode():
        with autocast_ctx:
            outputs = model(img_tensor, gamma_prior=1.0)
            # Numeric safety guarding for FP16 clamp logic
            if precision in ["fp16", "bf16"]:
                outputs["malignancy"] = torch.clamp(
                    outputs["malignancy"].float(), min=-15.0, max=15.0
                )

    return outputs
