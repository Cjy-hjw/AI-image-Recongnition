"""LoRA checkpoint 合并脚本——训练期 LoRA 结构 → 推理期普通 Conv2d 结构。

背景（见 memory/lora-checkpoint-inference-mismatch.md）：
  train_thyroidxl.py 用 use_lora=True 构建模型并保存 checkpoint，state_dict 里含
  lora_A/lora_B 权重。但 predict.py/evaluate.py 走的 model_runner.build_model()
  构建的是**不感知 LoRA** 的普通模型（use_lora 默认 False），若直接把 LoRA
  checkpoint 喂给它，build_model 现在会因 key 不匹配硬报错（已从 strict=False
  静默丢弃改为硬校验，见 src/inference/model_runner.py）。

本脚本用 LoRAConv2d.merge_weights() 把 lora_A/lora_B 数学合并回原始 Conv2d
weight（out = W + (B·A)*scale，在 fp32、非 autocast 上下文执行，无精度损失，
见 src/models/lora_utils.py），产出的普通 state_dict 可被 build_model() 直接
strict 加载，不需要推理侧感知 LoRA 这个训练期概念（Opus 架构讨论结论）。

用法：
    python scripts/merge_lora_checkpoint.py \
        --in checkpoints/thyroidxl_best.pth \
        --out checkpoints/thyroidxl_merged.pth \
        --config configs/thyroid_config.yaml \
        --lora-r 8 --lora-alpha 16

产出的 --out 是裸 state_dict（与 best_model.pth 同规格，非 {"state_dict":...}
包装），可直接 `predict.py --weights <out>` / `evaluate.py --weights <out>`。
"""
import argparse

import torch
import yaml

from src.models.multi_task_model import MultiTaskThyroidModel
from src.models.lora_utils import disable_lora, LoRAConv2d


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, help="训练产出的 LoRA 结构 checkpoint")
    p.add_argument("--out", required=True, help="合并后的普通 state_dict 输出路径")
    p.add_argument("--config", default="configs/thyroid_config.yaml")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")  # 合并是纯权重代数运算，无需 GPU
    model = MultiTaskThyroidModel(
        backbone_name=config["model"]["backbone"],
        pretrained=False,
        config=config,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(device)

    raw = torch.load(args.in_path, map_location=device, weights_only=True)
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    # strict=True：LoRA 结构模型与 LoRA checkpoint 必须逐 key 完全对齐；
    # 对不齐说明 --lora-r/--lora-alpha 或 backbone 与训练时不一致，直接报错而非猜测。
    model.load_state_dict(state_dict, strict=True)
    n_lora_before = sum(1 for m in model.modules() if isinstance(m, LoRAConv2d))
    print(f"加载完成：{n_lora_before} 个 LoRAConv2d 层待合并")

    # 合并必须在 fp32 执行（lora_A/lora_B 目前始终是 fp32 master weight，本项目
    # 所有 AMP 训练都只在前向 autocast、从不持久化转换参数 dtype，故此处 .float()
    # 在当前管线是 no-op）。保留这行是给未来防御：若某天训练脚本改成持久化转
    # bf16/fp16 存参数（而非纯 autocast），这里会先转回 fp32 再合并，防止合并本身
    # 引入额外精度损失。
    model = model.float()
    disable_lora(model)
    n_lora_after = sum(1 for m in model.modules() if isinstance(m, LoRAConv2d))
    assert n_lora_after == 0, f"合并后仍残留 {n_lora_after} 个 LoRAConv2d，disable_lora 未生效"

    torch.save(model.state_dict(), args.out)
    print(f"合并完成 → {args.out}（普通 state_dict，可直接被 build_model() strict 加载）")


if __name__ == "__main__":
    main()
