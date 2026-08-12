"""LoRA（Low-Rank Adaptation）工具——为 Conv2d (1x1) 注入低秩适配器。

适配目标：
  EfficientNet-B2：cls_backbone.blocks.*.conv_pw / conv_pwl（1x1 pointwise）
  ConvNeXt-Nano：seg_backbone.stages.*.blocks.*.mlp.fc1 / fc2（1x1 Conv2d）

参考：LoRA: Low-Rank Adaptation of Large Language Models (Hu et al. 2022)
"""
import math
from typing import List

import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    """1x1 Conv2d 的 LoRA 包装。

    原始权重 W 冻结，仅 A（r×in）和 B（out×r）矩阵参与梯度。
    前向：out = W(x) + (B·A)(x) * (alpha/r)
    初始化：A ~ Kaiming uniform，B = 0（训练初期 LoRA 贡献为零）。
    """

    def __init__(self, conv: nn.Conv2d, r: int = 8, alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        assert conv.kernel_size == (1, 1), (
            f"LoRAConv2d 仅支持 1x1 conv，当前 kernel={conv.kernel_size}"
        )
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.stride = conv.stride
        self.padding = conv.padding
        self.groups = conv.groups
        self.r = r
        self.scale = alpha / r
        assert self.scale <= 16.0, f"LoRA scale={self.scale} (alpha={alpha}/r={r}) > 16 在 fp16 下有溢出风险"

        # 冻结原始权重
        self.weight = nn.Parameter(conv.weight.data.clone(), requires_grad=False)
        self.bias = (
            nn.Parameter(conv.bias.data.clone(), requires_grad=False)
            if conv.bias is not None
            else None
        )

        # LoRA 矩阵：A (r, in, 1, 1)，B (out, r, 1, 1)
        self.lora_A = nn.Parameter(torch.empty(r, self.in_channels, 1, 1))
        self.lora_B = nn.Parameter(torch.zeros(self.out_channels, r, 1, 1))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = nn.functional.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, groups=self.groups
        )
        lora = nn.functional.conv2d(
            self.dropout(x), self.lora_A, None, self.stride, self.padding, groups=self.groups
        )
        lora = nn.functional.conv2d(lora, self.lora_B, None, stride=1, padding=0)
        return base + lora * self.scale

    def merge_weights(self) -> nn.Conv2d:
        """将 LoRA 合并进 weight，返回等价的普通 Conv2d（推理部署前调用）。"""
        merged = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            1,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=self.bias is not None,
        )
        with torch.no_grad():
            # lora_B: (out, r, 1, 1), lora_A: (r, in, 1, 1)
            lora_w = (
                self.lora_B.reshape(self.out_channels, self.r)
                @ self.lora_A.reshape(self.r, self.in_channels)
            ).unsqueeze(-1).unsqueeze(-1)
            merged.weight.copy_(self.weight + lora_w * self.scale)
            if self.bias is not None:
                merged.bias.copy_(self.bias)
        return merged


def _is_lora_target(name: str, module: nn.Module) -> bool:
    """判断是否为 LoRA 插入目标：EfficientNet-B2 pointwise 或 ConvNeXt MLP fc 层。"""
    if not isinstance(module, nn.Conv2d):
        return False
    if module.kernel_size != (1, 1):
        return False
    # EfficientNet-B2: blocks.*.conv_pw 或 conv_pwl
    if "conv_pw" in name and "act" not in name and "bn" not in name:
        return True
    # ConvNeXt-Nano: stages.*.blocks.*.mlp.fc1 / fc2
    if "mlp.fc1" in name or "mlp.fc2" in name:
        return True
    return False


def apply_lora(model: nn.Module, r: int = 8, alpha: int = 16, lora_dropout: float = 0.1) -> int:
    """将 LoRAConv2d 替换进模型目标层，返回替换数量。

    替换后 get_lora_params(model) 才能正确收集 A/B 矩阵。
    """
    replaced = 0
    # list() 防止迭代过程中修改模型结构
    for name, module in list(model.named_modules()):
        if not _is_lora_target(name, module):
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr = parts[-1]
        setattr(parent, attr, LoRAConv2d(module, r=r, alpha=alpha, dropout=lora_dropout))
        replaced += 1
    return replaced


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """返回所有 LoRA A/B 矩阵参数，供 optimizer 单独分组（不参与 LLRD）。"""
    params = []
    for module in model.modules():
        if isinstance(module, LoRAConv2d):
            params.extend([module.lora_A, module.lora_B])
    return params


def disable_lora(model: nn.Module) -> None:
    """将所有 LoRAConv2d 合并回普通 Conv2d（推理容器部署前调用）。"""
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRAConv2d):
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr = parts[-1]
        setattr(parent, attr, module.merge_weights())
