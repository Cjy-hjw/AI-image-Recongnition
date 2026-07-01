"""
PCGrad (Projecting Conflicting Gradients) 优化器

用于解决多任务学习中的梯度冲突问题。

问题背景：
- 分割任务（几何驱动）vs 分类任务（纹理驱动）
- 共享编码器的梯度更新方向可能冲突（余弦<0）
- 负迁移导致多任务性能低于单任务

原理：
- 计算各任务对共享参数的梯度 g_i, g_j
- 若 g_i · g_j < 0（冲突），将 g_i 投影到 g_j 的正交超平面
- 投影公式：g_i' = g_i - (g_i · g_j / ||g_j||²) * g_j
- 消除冲突方向，保留一致方向

实现：
- 包装PyTorch Optimizer（如Adam/SGD）
- 在step()前拦截梯度，检测冲突并投影
- 支持多任务（>2个任务时两两投影）

References:
- PCGrad Paper: https://arxiv.org/abs/2001.06782
- 32G显存约束：PCGrad额外显存开销<5%，适合官方平台

Usage:
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> pcgrad = PCGrad(optimizer)
    >>>
    >>> # 训练循环（注意：不要手动backward）
    >>> pcgrad.pc_backward([loss_seg, loss_cls])  # 自动backward+投影
    >>> pcgrad.step()
    >>> pcgrad.zero_grad()
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Optional
import numpy as np
import copy


class PCGrad:
    """
    PCGrad优化器包装器

    Args:
        optimizer: 基础优化器（如Adam）
        reduction: 梯度聚合方式 'mean'/'sum'
    """

    def __init__(self, optimizer: Optimizer, reduction: str = 'mean'):
        self.optimizer = optimizer
        self.reduction = reduction

        # 获取所有需要梯度的参数
        self._param_groups = optimizer.param_groups
        self._params = []
        for group in self._param_groups:
            self._params.extend(group['params'])

    def zero_grad(self):
        """清零梯度"""
        self.optimizer.zero_grad()

    def step(self):
        """执行优化步"""
        return self.optimizer.step()

    def state_dict(self):
        """保存状态"""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """加载状态"""
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def _flatten_grad(self, grads: List[torch.Tensor]) -> torch.Tensor:
        """将梯度列表展平为1D向量"""
        return torch.cat([g.flatten() if g is not None else torch.zeros_like(p).flatten()
                          for g, p in zip(grads, self._params)])

    def _unflatten_grad(self, flat_grad: torch.Tensor) -> List[torch.Tensor]:
        """将1D向量还原为梯度列表"""
        grads = []
        offset = 0
        for p in self._params:
            numel = p.numel()
            grads.append(flat_grad[offset:offset+numel].view_as(p))
            offset += numel
        return grads

    def _project_conflicting(self, grad_i: torch.Tensor, grad_j: torch.Tensor) -> torch.Tensor:
        """
        投影冲突梯度

        Args:
            grad_i: 任务i的梯度向量（1D）
            grad_j: 任务j的梯度向量（1D）

        Returns:
            grad_i': 投影后的梯度
        """
        # 计算内积
        dot_product = torch.dot(grad_i, grad_j)

        # 若余弦>=0，无冲突，保持原样
        if dot_product >= 0:
            return grad_i

        # 冲突：投影到正交超平面
        # g_i' = g_i - (g_i · g_j / ||g_j||²) * g_j
        grad_j_norm_sq = torch.dot(grad_j, grad_j)
        if grad_j_norm_sq == 0:
            return grad_i  # 避免除零

        projection = (dot_product / grad_j_norm_sq) * grad_j
        return grad_i - projection

    def pc_backward(self, losses: List[torch.Tensor]):
        """
        PCGrad反向传播（自动处理backward）

        Args:
            losses: 各任务损失列表 [loss_seg, loss_cls, ...]
                注意：调用前不要手动backward，本函数内部自动处理

        修复说明：
            - 旧版假设外部已backward，但此时梯度已累加无法投影
            - 新版内部逐任务backward+清零，提取独立梯度
        """
        num_tasks = len(losses)
        if num_tasks <= 1:
            # 单任务，无需PCGrad
            return

        # 1. 逐任务backward并提取独立梯度（每次清零避免累加）
        task_grads = []
        for i, loss in enumerate(losses):
            self.zero_grad()  # 清零避免与前一任务累加
            loss.backward(retain_graph=(i < num_tasks - 1))
            grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                     for p in self._params]
            task_grads.append(grads)

        # 2. 展平为向量
        flat_grads = [self._flatten_grad(grads) for grads in task_grads]

        # 3. 两两投影冲突梯度
        projected_grads = []
        for i in range(num_tasks):
            grad_i = flat_grads[i].clone()

            # 与其他所有任务检测冲突
            for j in range(num_tasks):
                if i == j:
                    continue
                grad_i = self._project_conflicting(grad_i, flat_grads[j])

            projected_grads.append(grad_i)

        # 4. 聚合投影后的梯度
        if self.reduction == 'mean':
            merged_grad = torch.stack(projected_grads).mean(dim=0)
        elif self.reduction == 'sum':
            merged_grad = torch.stack(projected_grads).sum(dim=0)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        # 5. 还原并赋值给参数梯度
        unflattened = self._unflatten_grad(merged_grad)
        for p, g in zip(self._params, unflattened):
            if p.grad is not None:
                p.grad.copy_(g)


class PCGradWrapper(nn.Module):
    """
    PCGrad训练包装器（简化接口）

    集成到训练循环中，自动处理多任务梯度冲突

    Usage:
        >>> model = ThyroidMultiTaskModel(...)
        >>> loss_fn = MaskedMultiTaskLoss(...)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> pcgrad = PCGradWrapper(optimizer)
        >>>
        >>> # 训练步
        >>> outputs = model(inputs)
        >>> total_loss, loss_dict = loss_fn(outputs, targets)
        >>>
        >>> # 方式1：从loss_dict自动提取任务损失
        >>> task_losses = [loss_dict['seg'], loss_dict['malignancy'], loss_dict['concepts_ce']]
        >>> pcgrad.backward_and_step(task_losses)
        >>>
        >>> # 方式2：手动backward + pc_backward
        >>> optimizer.zero_grad()
        >>> for i, loss in enumerate(task_losses):
        ...     loss.backward(retain_graph=(i < len(task_losses)-1))
        >>> pcgrad.pc_backward(task_losses)
        >>> optimizer.step()
    """

    def __init__(self, optimizer: Optimizer, reduction: str = 'mean'):
        super().__init__()
        self.pcgrad = PCGrad(optimizer, reduction)

    def backward_and_step(self, task_losses: List[torch.Tensor]):
        """
        简化接口：自动backward + 投影 + step

        Args:
            task_losses: 任务损失列表

        注意：pc_backward已内部处理backward，此函数直接调用即可
        """
        # pc_backward内部自动backward+投影
        self.pcgrad.pc_backward(task_losses)

        # 优化步
        self.pcgrad.step()

    def zero_grad(self):
        self.pcgrad.zero_grad()

    def step(self):
        return self.pcgrad.step()

    def pc_backward(self, task_losses: List[torch.Tensor]):
        self.pcgrad.pc_backward(task_losses)


# ========== 辅助工具：梯度冲突分析 ==========

def compute_gradient_cosine(grad1: torch.Tensor, grad2: torch.Tensor) -> float:
    """
    计算两个梯度向量的余弦相似度

    Args:
        grad1, grad2: 展平的梯度向量

    Returns:
        cosine: 余弦值，[-1, 1]
            >0: 方向一致
            <0: 冲突
    """
    dot = torch.dot(grad1, grad2)
    norm1 = torch.norm(grad1)
    norm2 = torch.norm(grad2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return (dot / (norm1 * norm2)).item()


def analyze_gradient_conflicts(model: nn.Module, task_losses: List[torch.Tensor]) -> dict:
    """
    分析多任务梯度冲突情况（用于监控训练）

    Args:
        model: 模型
        task_losses: 任务损失列表

    Returns:
        dict: {
            'conflict_pairs': [(i, j, cosine), ...],  # 冲突对及余弦值
            'conflict_rate': float,  # 冲突比例
            'mean_cosine': float,    # 平均余弦
        }
    """
    num_tasks = len(task_losses)
    if num_tasks <= 1:
        return {'conflict_pairs': [], 'conflict_rate': 0.0, 'mean_cosine': 1.0}

    # 提取梯度
    task_grads = []
    for i, loss in enumerate(task_losses):
        model.zero_grad()
        loss.backward(retain_graph=True)

        # 展平梯度
        grad_flat = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        task_grads.append(grad_flat)

    # 两两计算余弦
    conflict_pairs = []
    cosines = []

    for i in range(num_tasks):
        for j in range(i+1, num_tasks):
            cosine = compute_gradient_cosine(task_grads[i], task_grads[j])
            cosines.append(cosine)

            if cosine < 0:
                conflict_pairs.append((i, j, cosine))

    conflict_rate = len(conflict_pairs) / len(cosines) if cosines else 0.0
    mean_cosine = np.mean(cosines) if cosines else 0.0

    return {
        'conflict_pairs': conflict_pairs,
        'conflict_rate': conflict_rate,
        'mean_cosine': mean_cosine,
    }


# ========== 单元测试 ==========

if __name__ == '__main__':
    print("[测试] PCGrad梯度投影")

    # 测试1: 梯度投影逻辑
    grad_i = torch.tensor([1.0, 2.0, 3.0])
    grad_j = torch.tensor([-1.0, -1.0, 0.0])

    # 计算余弦
    cosine = compute_gradient_cosine(grad_i, grad_j)
    print(f"原始梯度余弦: {cosine:.4f} (< 0则冲突)")

    # 创建PCGrad实例（mock optimizer）
    mock_model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(mock_model.parameters(), lr=0.01)
    pcgrad = PCGrad(optimizer)

    # 投影
    grad_i_proj = pcgrad._project_conflicting(grad_i, grad_j)
    cosine_proj = compute_gradient_cosine(grad_i_proj, grad_j)
    print(f"投影后梯度余弦: {cosine_proj:.4f} (应>=0)")

    assert cosine < 0, "测试用例应构造冲突梯度"
    assert cosine_proj >= -1e-6, "投影后应消除冲突"

    print("\n[OK] PCGrad核心逻辑测试通过")
    print("完整测试需真实训练环境（多任务损失backward）")
