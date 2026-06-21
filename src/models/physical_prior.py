import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiablePhysicalPriorExtractor(nn.Module):
    """
    可微物理先验特征提取层：计算 soft mask 的可微宽高比、回声比和圆度，
    并将其合并为 3 维特征向量，支持全局端到端反向传播。
    """

    def __init__(self, img_size=512, eps=1e-4):
        super().__init__()
        self.img_size = img_size
        self.eps = eps

        # 预先生成空间网格坐标，注册为 buffer 避免不必要的显存重建
        y_coords, x_coords = torch.meshgrid(
            torch.arange(img_size, dtype=torch.float32),
            torch.arange(img_size, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("x_coords", x_coords)
        self.register_buffer("y_coords", y_coords)

    def forward(self, mask, img):
        """
        Args:
            mask: [B, 1, H, W] - 经 Sigmoid 激活的连续 soft 掩码
            img:  [B, 1, H, W] - 输入 of 2D 归一化单通道超声图像
        Returns:
            v_phys: [B, 3] - 提取出的完全可微的物理指标 [AR_diff, ER_diff, C_diff]
        """
        orig_dtype = mask.dtype
        mask = mask.float()
        img = img.float()
        B, C, H, W = mask.shape
        eps = self.eps

        # 类型与设备对齐，强制使用 float32 计算以防 FP16 溢出/下溢
        x_coords = self.x_coords.to(device=mask.device, dtype=torch.float32)
        y_coords = self.y_coords.to(device=mask.device, dtype=torch.float32)

        # ----------------------------------------------------
        # 0. 退化掩码检测与安全替换 (Degenerate Mask Guard)
        # ----------------------------------------------------
        mask_sum = mask.sum(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
        mask_sum_flat = mask_sum[
            ..., 0, 0
        ]  # [B, 1] — explicit indexing, no double squeeze
        is_degenerate = mask_sum_flat <= 10.0  # [B, 1] — bool

        # 将退化样本的 mask 替换为安全 dummy（常数 0.5），阻断 inf 梯度路径
        safe_mask = torch.where(
            is_degenerate.unsqueeze(-1).unsqueeze(-1),  # [B, 1, 1, 1]
            torch.full_like(mask, 0.5),
            mask,
        )

        # 当软像素面积总和大于 100 时物理算子开始激活，低于 100 逐渐回归基准值 1.0
        presence = (
            torch.sigmoid((mask_sum - 100.0) / 20.0).squeeze(-1).squeeze(-1)
        )  # [B, 1]

        # ----------------------------------------------------
        # 1. 计算可微等效宽高比 (Aspect Ratio) — 使用 safe_mask
        # ----------------------------------------------------
        safe_mask_sum = safe_mask.sum(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
        prob_dist = safe_mask / (safe_mask_sum + eps)  # 软归一化概率密度分布

        mu_x = (x_coords.unsqueeze(0).unsqueeze(0) * prob_dist).sum(
            dim=(2, 3)
        )  # [B, 1]
        mu_y = (y_coords.unsqueeze(0).unsqueeze(0) * prob_dist).sum(
            dim=(2, 3)
        )  # [B, 1]

        var_x = (
            (x_coords.unsqueeze(0).unsqueeze(0) - mu_x.unsqueeze(-1).unsqueeze(-1)) ** 2
            * prob_dist
        ).sum(dim=(2, 3))
        var_y = (
            (y_coords.unsqueeze(0).unsqueeze(0) - mu_y.unsqueeze(-1).unsqueeze(-1)) ** 2
            * prob_dist
        ).sum(dim=(2, 3))

        aspect_ratio_raw = torch.sqrt(var_y + eps) / torch.sqrt(var_x + eps)  # [B, 1]

        # ----------------------------------------------------
        # 2. 计算可微回声比例 (Echo Ratio) — 使用 safe_mask
        # ----------------------------------------------------
        # 通过强门控剥离超声黑边
        active_gate = torch.sigmoid(100.0 * (img - 0.05))  # [B, 1, H, W]
        background_mask = active_gate * (1.0 - safe_mask)

        safe_mask_sum_flat = safe_mask_sum[..., 0, 0]  # [B, 1]
        roi_mean = (img * safe_mask).sum(dim=(2, 3)) / (safe_mask_sum_flat + eps)
        bg_mean = (img * background_mask).sum(dim=(2, 3)) / (
            background_mask.sum(dim=(2, 3)) + eps
        )
        echo_ratio_raw = roi_mean / (bg_mean + eps)  # [B, 1]

        # ----------------------------------------------------
        # 3. 计算可微圆度 (Circularity) — 使用 safe_mask
        # ----------------------------------------------------
        area = safe_mask_sum_flat  # [B, 1]

        # 使用 PyTorch 2D Max-Pooling 实现可微数学形态学腐蚀，提取极窄（单像素级）软边缘，对齐评估期 cv2.findContours 结果
        eroded_mask = -F.max_pool2d(-safe_mask, kernel_size=3, stride=1, padding=1)
        boundary = torch.clamp(safe_mask - eroded_mask, min=0.0)
        perimeter = boundary.sum(dim=(2, 3))  # [B, 1]

        # 计算稳定圆度，防范 perimeter -> 0 时的除零爆炸
        circularity_raw = (4.0 * 3.1415926 * area) / (perimeter**2 + 1e-6)

        # ----------------------------------------------------
        # 4. 退化样本特征覆写为基准值 1.0
        # ----------------------------------------------------
        aspect_ratio_raw = torch.where(
            is_degenerate, torch.ones_like(aspect_ratio_raw), aspect_ratio_raw
        )
        echo_ratio_raw = torch.where(
            is_degenerate, torch.ones_like(echo_ratio_raw), echo_ratio_raw
        )
        circularity_raw = torch.where(
            is_degenerate, torch.ones_like(circularity_raw), circularity_raw
        )

        # ----------------------------------------------------
        # 5. 物理特征融合与门控插值
        # ----------------------------------------------------
        presence_flat = presence
        aspect_ratio = presence_flat * aspect_ratio_raw + (1.0 - presence_flat) * 1.0
        echo_ratio = presence_flat * echo_ratio_raw + (1.0 - presence_flat) * 1.0
        circularity = presence_flat * circularity_raw + (1.0 - presence_flat) * 1.0

        # 物理指标范围限制，防范除零导致的数值爆炸与 FP16 溢出
        aspect_ratio = torch.clamp(aspect_ratio, min=0.1, max=10.0)
        echo_ratio = torch.clamp(echo_ratio, min=0.1, max=10.0)
        circularity = torch.clamp(circularity, min=0.0, max=2.0)

        v_phys = torch.cat([aspect_ratio, echo_ratio, circularity], dim=-1)  # [B, 3]
        return v_phys.to(dtype=orig_dtype)
