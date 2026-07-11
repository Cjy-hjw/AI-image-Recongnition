"""Model construction, preprocessing, and forward pass with precision control."""

import os
import cv2
import torch

from src.models.multi_task_model import MultiTaskThyroidModel


def build_model(config: dict, weights: str, device: torch.device):
    """Build MultiTaskThyroidModel and load weights if available.

    推理侧模型**不感知 LoRA**——训练脚本（如 train_thyroidxl.py）产出的 LoRA 结构
    checkpoint 必须先用 scripts/merge_lora_checkpoint.py 合并（LoRAConv2d.merge_weights()
    数学等价合并回普通 Conv2d）再喂给这里，保持 model_runner.py 对训练期 PEFT 技巧无感。
    """
    model = MultiTaskThyroidModel(
        backbone_name=config["model"]["backbone"],
        pretrained=False,
        config=config,
    ).to(device)
    if os.path.exists(weights):
        # weights_only=True:best_model.pth 按设计是纯 model state_dict(续训态另存
        # training_state.pth),torch≥2.6 下安全加载并消除 FutureWarning。
        raw = torch.load(weights, map_location=device, weights_only=True)
        # 兼容训练脚本另存的包装格式 {"epoch":, "stage":, "state_dict":, ...}
        # （如 train_thyroidxl.py）——不解包会把整个 dict 当 state_dict 喂进去，
        # 710 个参数全部 key 不匹配，静默变成随机初始化（本次事故根因之一）。
        state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # strict=False 静默吞掉不匹配 key 是本次事故的核心教训：LoRA checkpoint 误喂
        # 非 LoRA 模型时，144 个 lora_A/lora_B key 全部落入 unexpected 却不报错，
        # backbone 实际只剩原始冻结预训练权重。改为硬校验——一旦有 missing/unexpected，
        # 说明 checkpoint 与模型结构不匹配（如喂了未合并的 LoRA checkpoint），直接报错，
        # 不允许静默降级为部分/随机权重进入生产推理。
        if missing or unexpected:
            raise RuntimeError(
                f"[FATAL] 权重与模型结构不匹配，拒绝静默部分加载：{weights}\n"
                f"  missing keys ({len(missing)}): {missing[:10]}\n"
                f"  unexpected keys ({len(unexpected)}): {unexpected[:10]}\n"
                "  若 unexpected 中含 'lora_' 前缀：说明喂了未合并的 LoRA 训练 checkpoint，"
                "先跑 scripts/merge_lora_checkpoint.py 合并后再用。"
            )
        print(f"Loaded model weights from {weights}")
    else:
        print(
            "[WARNING] Trained checkpoint not found. Using default initialized weights."
        )
    model.eval()
    return model


def preprocess_image(img_np, img_size: int, device: torch.device):
    """Resize grayscale image and convert to normalized tensor [B,1,H,W]."""
    assert img_np.ndim == 2, f"preprocess_image expects 2D grayscale, got shape {img_np.shape}"
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

    # 陷阱：torch.autocast 在 device_type="cpu" 时,进入 with 块(__enter__ →
    # torch.set_autocast_cpu_dtype)才会真正校验 dtype,且**不看 enabled 参数**——
    # 只要 dtype=torch.float16 就必然 TORCH_CHECK 失败("AutocastCPU only support
    # Bfloat16"),不是在 torch.autocast(...) 构造这一步就崩。之前 enabled=False
    # 但 dtype 仍传 float16,导致下面的 [WARNING] 打印了"已降级",但进 with 块时仍
    # 必然崩溃,降级从未真正生效(Gate2 review 2026-07-11 修正措辞:崩溃点在
    # __enter__ 不在构造函数)。
    # 修复：is_fp16_on_cpu 时把 dtype 也一并避开 float16(enabled=False 时这个
    # dtype 值不参与任何实际 cast,纯粹是为了避开触发 TORCH_CHECK 的组合)。
    # 注：precision=="fp32" 分支下 enabled 恒为 False(CPU/GPU 皆然),此时
    # dtype 取 torch.float16 只是占位值、不影响任何数值路径,并非刻意选择。
    autocast_ctx = torch.autocast(
        device_type=device_type,
        enabled=(precision in ["fp16", "bf16"] and not is_fp16_on_cpu),
        dtype=torch.bfloat16 if (precision == "bf16" or is_fp16_on_cpu) else torch.float16,
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
