"""Model construction, preprocessing, and forward pass with precision control."""

import os
import cv2
import torch
import yaml

from src.models.multi_task_model import MultiTaskThyroidModel


def load_config(config_path: str) -> dict:
    """Load YAML config and eagerly validate the keys run_forward/build_model 实际会用到。

    2026-07-11 修复：predict.py/video_predict.py 此前在 try 块**外**直接
    `open(args.config)` → `yaml.safe_load` → `config["environment"]["device"]`，
    路径不存在(`FileNotFoundError`)/YAML 格式错误(`yaml.YAMLError`)/键缺失
    (`KeyError`)/内容为空导致 `yaml.safe_load` 返回 `None`(`TypeError`：
    `None["environment"]` 不可下标)都会导致进程裸崩溃、非零 exit code、零
    输出——与 `io.py`/`video_io.py` 的输入加载失败是同一类"try 外裸操作"
    问题，那次已修（见 `load_input_image` 契约），这次补上配置加载这一处。
    调用方**必须** `try/except Exception`（不能只 `except KeyError`，上述
    `TypeError` 等其它异常类型也要兜住）包裹本函数，兜成 `status="ERROR"`
    的 `result.json`（复用 `export.save_rejection_json`），不能让异常裸传播。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # 提前触达下游实际会用到的键：让 KeyError 在这里发生，而不是推理中途才炸，
    # 这样调用方的 try/except 边界能覆盖到（此处仅探测存在性，不改变 config）。
    _ = config["environment"]["device"]
    _ = config["data"]["img_size"]
    return config


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
        # weights_only=True:部署默认权重 thyroidxl_merged_v2.pth(merge 产出的纯 Conv2d
        # state_dict)与 train.py 的 best_model.pth 同规格,都是纯 model state_dict(续训态
        # 另存 training_state.pth),torch≥2.6 下安全加载并消除 FutureWarning。
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
        # 2026-08-11 #115: 主诊断模型(计分组件)权重文件缺失的分流——
        #   PRODUCTION=1(交付容器)→ 硬报错。随机初始化的主模型会输出"结构合法但
        #     数值全错"的垃圾诊断,比崩溃更阴(见 memory/delivery-container-missing-weights):
        #     容器漏 COPY 权重(#111 场景)时 os.path.exists 为 False,静默随机初始化会让
        #     整批病例拿到看似专业实则随机的 TI-RADS/良恶性,零分且无人察觉。故 PRODUCTION 下抛错拒绝随机初始化。
        #     ⚠️ 局限(#115/#123 未闭环):HTTP 服务惰性构建(runner 首次 /call 才调 build_model),
        #     此 raise 会被 tasks._run 兜成病例级 FAILED、容器 /health 仍 200——即"伪装健康跑零分",
        #     并非真正"拒启"。真正的服务启动硬门(main() 启动期主动 build 一次、失败即退出容器)见 #123。
        #   非 PRODUCTION(CI 冒烟/本地调试)→ 保留随机降级 + warning:CI 无 checkpoint 环境
        #     靠此跑机制层冒烟(test_smoke_entrypoints / test_model_runner 缺权重降级用例)。
        #   与 QA_VP §1.7「计分组件权重故障仍硬报错」口径一致(零计分的 authenticity 才降级 skipped)。
        if os.environ.get("PRODUCTION") == "1":
            raise RuntimeError(
                f"[FATAL] 主诊断模型权重文件不存在,PRODUCTION 模式拒绝随机初始化启动:{weights}\n"
                "  随机初始化的主模型会输出结构合法但数值全错的垃圾诊断(整批零分且难察觉)。\n"
                "  核对交付容器 Dockerfile 是否 COPY 了 checkpoints/ 权重(见交付 ToDO #111)。"
            )
        print(
            "[WARNING] Trained checkpoint not found. Using default initialized weights. "
            "(非 PRODUCTION 调试降级;生产容器请设 PRODUCTION=1 使缺权重硬报错)"
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

    # 陷阱（2026-07-11 两轮教训，CI test job 实测抓到）：torch==2.1.0 的
    # torch.autocast.__enter__ 在 device_type="cpu" 时**无条件**调用
    # torch.set_autocast_cpu_dtype(dtype)——不看 enabled，只要 dtype!=
    # torch.bfloat16 就必然 TORCH_CHECK 失败("AutocastCPU only support
    # Bfloat16")。第一轮修复只堵了 precision=="fp16" 这一种取值（把 dtype
    # 换成 bfloat16），却让 precision=="fp32" 的 else 分支从原来的 bfloat16
    # 变成了 float16——以为 enabled=False 时 dtype 是 no-op（Gate2 review 也
    # 这么判断），但 __enter__ 的校验发生在 enabled 判断之前，实测 CPU 上任何
    # dtype=float16 都崩，与 enabled 值无关。正确做法：CPU 上 dtype 恒为
    # bfloat16（不管 precision 是 fp32/fp16/bf16），靠 enabled 门控是否真正
    # 生效；GPU 上维持原语义不变。
    if device_type == "cpu":
        ac_dtype = torch.bfloat16
    else:
        ac_dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    autocast_ctx = torch.autocast(
        device_type=device_type,
        enabled=(precision in ["fp16", "bf16"] and not is_fp16_on_cpu),
        dtype=ac_dtype,
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
