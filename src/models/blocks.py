"""可复用的 nn 构建块（从 multi_task_model.py 外移，rules.md §6 保持装配枢纽 < 300 行）。

- `ConvBlock`：U-Net 风格 double-conv 解码块。
- `TaskAlignmentAdaptor`：分割→分类 单向软注意力门控。
  **契约**：调用方 MUST 先 `.detach()` `f_seg`，防分类误差梯度回流污染分割主干；
  `gamma` 零初始化残差门控。详见 `src/models/AGENTS.md` §1.1 + Gate 8 隔离审计。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic double convolution block for U-Net style decoder."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class TaskAlignmentAdaptor(nn.Module):
    def __init__(self, seg_in_channels, cls_in_channels):
        super().__init__()
        self.channel_projector = nn.Sequential(
            nn.Conv2d(seg_in_channels, cls_in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cls_in_channels),
            nn.ReLU(inplace=True),
        )
        self.attention_generator = nn.Sequential(
            nn.Conv2d(
                cls_in_channels,
                cls_in_channels,
                kernel_size=3,
                padding=1,
                groups=cls_in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(cls_in_channels),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f_seg_detached, f_cls):
        if f_seg_detached.shape[-2:] != f_cls.shape[-2:]:
            f_seg_detached = F.interpolate(
                f_seg_detached,
                size=f_cls.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        f_seg_detached = f_seg_detached.float()
        f_cls_float = f_cls.float()
        projected = self.channel_projector(f_seg_detached)
        attention_mask = self.attention_generator(projected)
        out = f_cls_float + self.gamma * (f_cls_float * attention_mask)
        return out.to(dtype=f_cls.dtype)
