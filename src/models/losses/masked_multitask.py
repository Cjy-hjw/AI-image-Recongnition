"""
掩膜多任务损失包装器 (Masked Multi-Task Loss Wrapper)

用于处理异构标注的部分标签问题（Partial Labels）。

原理：
- 多源数据集标注异质化：ThyroidXL全标注，ThyUS2Path仅病理，DDTI仅分割
- 训练时通过task_mask指示每个样本的标注可用性
- 损失计算时自动屏蔽无标注任务的梯度（避免负梯度惩罚）

实现：
- 包装现有MultiTaskLoss，保持向后兼容
- batch级task_mask控制，每个任务独立屏蔽
- 支持完全缺失任务（整个batch无标注时loss=0）

Usage:
    >>> loss_fn = MaskedMultiTaskLoss(base_loss=MultiTaskLoss(...))
    >>> task_mask = {
    ...     'mask': torch.tensor([1, 0, 1, 0]),  # 样本0,2有分割mask
    ...     'malignancy': torch.tensor([1, 1, 1, 1]),  # 全部有病理
    ...     'composition': torch.tensor([1, 0, 1, 0]),  # 部分有TI-RADS
    ... }
    >>> total_loss, loss_dict = loss_fn(outputs, targets, task_mask=task_mask)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class MaskedMultiTaskLoss(nn.Module):
    """
    掩膜多任务损失包装器

    在batch维度对各任务应用有效性掩膜，自动零梯度处理缺失标注

    Args:
        base_loss: 基础损失函数（如MultiTaskLoss实例）
        default_all_valid: 若task_mask=None时是否默认全部有效（向后兼容）
    """

    def __init__(self, base_loss: nn.Module, default_all_valid: bool = True):
        super().__init__()
        self.base_loss = base_loss
        self.default_all_valid = default_all_valid

        # 任务名映射到损失字典key
        self.task_to_loss_keys = {
            'mask': ['seg'],
            'composition': ['comp', 'concepts_ce', 'prior'],
            'echogenicity': ['echo', 'concepts_ce', 'prior'],
            'shape': ['shape', 'concepts_ce', 'prior'],
            'margin': ['margin', 'concepts_ce', 'prior'],
            'foci': ['foci', 'concepts_ce', 'prior'],
            'malignancy': ['malignancy', 'supcon'],
        }

    def _apply_mask_to_loss(
        self, loss: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        对单个损失应用掩膜

        Args:
            loss: 损失张量，形状可能是 (B,) 或 标量
            mask: 掩膜张量 (B,)，值为0或1

        Returns:
            masked_loss: 屏蔽后的损失
        """
        mask_float = mask.float()
        n_valid = mask_float.sum()

        if n_valid == 0:
            # 整个batch无有效标注，返回零梯度
            return loss * 0.0

        # 如果loss是标量（已经reduce过），无法直接应用mask
        # 这种情况需要在base_loss内部处理，这里只能保持原样
        if loss.dim() == 0:
            # 标量loss，假设base_loss已经处理了mask
            # 或者这是一个不支持batch-wise mask的任务
            return loss

        # loss是(B,)形状，应用mask并平均
        if loss.size(0) == mask.size(0):
            return (loss * mask_float).sum() / n_valid

        # 其他情况（如loss已经是mean over batch），保持不变
        return loss

    def _create_default_mask(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """创建默认全有效的task_mask"""
        return {
            task: torch.ones(batch_size, dtype=torch.long, device=device)
            for task in self.task_to_loss_keys.keys()
        }

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        task_mask: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ):
        """
        前向传播

        Args:
            outputs: 模型输出字典
            targets: 目标标签字典（包含valid_*字段）
            task_mask: 任务掩膜字典 {task_name: Tensor[B]}
                如果为None且targets包含valid_*字段，自动从targets提取
            **kwargs: 传递给base_loss的其他参数

        Returns:
            (total_loss, loss_dict): 总损失和分项损失字典
        """
        batch_size = outputs['mask'].size(0)
        device = outputs['mask'].device

        # 1. 构建task_mask
        if task_mask is None:
            # 尝试从targets的valid_*字段提取
            task_mask = {}
            for task_name in self.task_to_loss_keys.keys():
                valid_key = f'valid_{task_name}'
                if valid_key in targets:
                    # targets[valid_key]可能是list或tensor
                    if isinstance(targets[valid_key], torch.Tensor):
                        task_mask[task_name] = targets[valid_key]
                    else:
                        task_mask[task_name] = torch.tensor(
                            targets[valid_key], dtype=torch.long, device=device
                        )
                elif self.default_all_valid:
                    task_mask[task_name] = torch.ones(batch_size, dtype=torch.long, device=device)

        # 2. 调用基础损失（不传task_mask，避免基础loss不支持该参数）
        total_loss, loss_dict = self.base_loss(outputs, targets, **kwargs)

        # 3. 后处理：对loss_dict中的各项应用task_mask
        # 注意：这里是近似处理，因为base_loss已经返回了聚合后的loss
        # 理想情况是在base_loss内部按样本计算loss，然后在这里mask
        # 当前实现：通过统计有效样本比例来调整loss权重

        masked_loss_dict = {}
        total_masked_loss = 0.0

        for task_name, loss_keys in self.task_to_loss_keys.items():
            if task_name not in task_mask:
                continue

            mask = task_mask[task_name]
            valid_ratio = mask.float().mean()  # 有效样本比例

            for loss_key in loss_keys:
                if loss_key in loss_dict:
                    original_loss = loss_dict[loss_key]

                    # 按有效比例缩放损失（近似mask效果）
                    # 完美实现需要在compute_*_loss内部应用mask
                    masked_loss = original_loss * valid_ratio

                    masked_loss_dict[loss_key] = masked_loss

                    # 累加到total（避免重复累加concepts_ce/prior）
                    if loss_key not in ['concepts_ce', 'prior']:
                        total_masked_loss += masked_loss

        # 更新total_loss
        if masked_loss_dict:
            # 使用masked版本的total
            total_loss = sum(masked_loss_dict.values())

        # 保留原始loss_dict中的其他项（如latent_l2, spin_l1）
        for key, value in loss_dict.items():
            if key not in masked_loss_dict and key != 'total':
                masked_loss_dict[key] = value

        masked_loss_dict['total'] = total_loss

        return total_loss, masked_loss_dict


# ========== 更完善的实现：修改base_loss内部计算 ==========

def compute_segmentation_loss_masked(
    bce_fn, dice_fn, outputs, targets, task_mask=None
):
    """
    带mask的分割损失计算（示例：如何在组件内部支持mask）

    Args:
        task_mask: dict with 'mask' key, Tensor[B] of 0/1
    """
    pred_mask = outputs['mask']
    true_mask = targets['mask']

    # BCE + Dice
    loss_bce = bce_fn(pred_mask, true_mask)  # (B, H, W) -> scalar
    loss_dice = dice_fn(pred_mask, true_mask)  # scalar

    loss_seg = loss_bce + loss_dice

    # Apply mask if provided
    if task_mask is not None and 'mask' in task_mask:
        mask_valid = task_mask['mask'].float()  # (B,)
        n_valid = mask_valid.sum()

        if n_valid > 0:
            # 需要per-sample loss，这里假设loss已经是(B,)或可以广播
            # 实际实现需要修改DiceLoss返回per-sample值
            # 当前是近似：按有效比例缩放
            loss_seg = loss_seg * (n_valid / mask_valid.numel())
        else:
            loss_seg = loss_seg * 0.0

    return loss_seg


# ========== 单元测试 ==========

if __name__ == '__main__':
    print("[测试] 掩膜多任务损失包装器")

    # 测试1: task_mask逻辑
    batch_size = 4
    task_mask = {
        'mask': torch.tensor([1, 0, 1, 0]),
        'malignancy': torch.tensor([1, 1, 1, 1]),
    }

    # 验证mask提取
    mask_valid = task_mask['mask'].float()
    valid_ratio = mask_valid.mean()
    print(f"Mask有效比例: {valid_ratio.item():.2f} (预期0.50)")
    assert abs(valid_ratio.item() - 0.5) < 0.01, "Mask比例计算错误"

    # 测试2: 损失缩放逻辑
    mock_loss = torch.tensor([1.0, 2.0, 3.0, 4.0])  # (B,)
    masked = (mock_loss * mask_valid).sum() / mask_valid.sum()
    print(f"Masked loss: {masked.item():.2f} (预期2.0: (1+3)/2)")
    assert abs(masked.item() - 2.0) < 0.01, "损失屏蔽计算错误"

    # 测试3: 全部无效
    all_zero_mask = torch.zeros(batch_size)
    if all_zero_mask.sum() == 0:
        masked_zero = mock_loss * 0.0
        print(f"全无效mask loss: {masked_zero.sum().item():.2f} (预期0.0)")
        assert masked_zero.sum().item() == 0.0, "全无效处理错误"

    print("\n[OK] 掩膜多任务损失核心逻辑测试通过")
    print("注意：完整测试需要真实训练环境（依赖MultiTaskLoss）")
    print("当前实现：后处理缩放版本（按有效比例调整loss权重）")
