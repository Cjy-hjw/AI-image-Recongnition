"""可复用的 nn 构建块（从 multi_task_model.py 外移，rules.md §6 保持装配枢纽 < 300 行）。

- `ConvBlock`：U-Net 风格 double-conv 解码块。
- `TaskAlignmentAdaptor`：分割→分类 单向软注意力门控。
  **契约**：调用方 MUST 先 `.detach()` `f_seg`，防分类误差梯度回流污染分割主干；
  `gamma` 零初始化残差门控。详见 `src/models/AGENTS.md` §1.1 + Gate 8 隔离审计。
- `CoordConv2d`：坐标增强卷积，给位置头提供绝对空间参照。
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


class CoordConv2d(nn.Module):
    """卷积前拼接归一化坐标通道（Liu et al. 2018, "An Intriguing Failing of CNNs"）。

    **为什么位置头必须要它**：普通卷积是平移等变的 —— 同一个结节图块出现在图像左侧
    还是右侧，卷积响应完全相同。而「左叶/右叶」恰恰只能由**绝对位置**区分，纯 CNN
    特征 + 全局池化在原理上就丢掉了这个信息。拼两个坐标通道后，卷积核可以学出
    「响应强度随 x 变化」的滤波器，绝对位置就被编码进通道特征，池化后仍保留。

    坐标取 `linspace(-1, 1)`：与 `F.grid_sample`/`affine_grid` 的归一化坐标系一致，
    且对输入分辨率不变（16×16 与 32×32 的最左列都是 -1），换 backbone 或改 img_size
    时头不需要重新学坐标尺度。

    🔴 **必须先过非线性再全局池化，否则坐标通道完全失效**（2026-08-12 实测坐实）。
    卷积是线性算子，可拆成两项：

        GAP(conv([x, c])) = GAP(conv_x(x)) + GAP(conv_c(c))

    第二项里 `c` 是与输入无关的固定坐标网格 → 该项是**常数偏置**；第一项则平移不变。
    两项都区分不了"图案在左还是在右"，于是 `CoordConv → GAP` 的输出对左右图案
    **逐位相等**。加了 ReLU 之后拆分不再成立：坐标偏置改变了哪些位置越过激活阈值，
    位置信息才真正进入池化结果。
    → 用法固定为 `CoordConv2d → (BN) → ReLU → pool`，**不要把池化提到非线性前面**。
    `tests/test_location_head.py::test_coordconv_needs_nonlinearity_before_pooling`
    用反证锁住这条（去掉 ReLU 立刻左右不可分）。

    ⚠️ H 或 W 为 1 时 `linspace(-1, 1, 1)` 返回 `[-1.]`（torch 语义），该维坐标恒为
    -1、无区分度。位置头输入是 16×16，不触发；此处不额外特判，避免为不会发生的
    情况引入分支。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels + 2,  # +2：x/y 两个坐标通道
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        # 每次前向按实际 H/W 生成，不用 register_buffer 缓存：位置头输入分辨率随
        # img_size 变化，缓存反而会在换分辨率时静默用错尺寸（或触发 state_dict 不匹配）。
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        x_ch = xs.view(1, 1, 1, w).expand(b, 1, h, w)
        y_ch = ys.view(1, 1, h, 1).expand(b, 1, h, w)
        return self.conv(torch.cat([x, x_ch, y_ch], dim=1))
