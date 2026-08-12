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
- 14~16G显存约束：PCGrad额外显存开销<5%，适合平台实际约16G/容器(当前分配14G)

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


class PCGrad:
    """
    PCGrad优化器包装器

    ⚠️ 实验性,未接入训练主循环(train.py 当前用 FAMO,不调用本类)。
       - 与 FAMO 互斥(二选一多任务权衡方案,叠加语义未定义);
       - 当前不兼容项目强制的 AMP/GradScaler:pc_backward 手工 backward + grad.copy_
         绕过了 scaler.scale()/unscale_,启用需切 fp32 或额外处理缩放;
       - 激活前必须过 Gate2 复审。

    Args:
        optimizer: 基础优化器（如Adam）
        reduction: 梯度聚合方式 'mean'/'sum'
        shared_parameters: 仅对这些参数(共享编码器)做冲突投影的可选参数列表。
            PCGrad 理论只应对**共享参数**投影(对各任务私有 head 投影是错的)。
            未传时退化为对 optimizer 全部参数投影 —— 此行为不严格正确,
            真正接入时**必须**传入共享编码器参数。
    """

    def __init__(
        self,
        optimizer: Optimizer,
        reduction: str = 'mean',
        shared_parameters: Optional[List[torch.nn.Parameter]] = None,
    ):
        self.optimizer = optimizer
        self.reduction = reduction

        # 获取所有需要梯度的参数
        self._param_groups = optimizer.param_groups
        self._all_params = []
        for group in self._param_groups:
            self._all_params.extend(group['params'])

        # 投影作用域: 仅共享参数(正确);未传则退回全模型参数(不严格正确,见类 docstring)
        if shared_parameters is not None:
            self._params = list(shared_parameters)
        else:
            self._params = self._all_params

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
        >>> loss_fn = MultiTaskLoss(...)
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

# 注: 梯度冲突分析/可视化工具及冒烟测试应在需要时于 tests/ 或独立分析脚本补充
# (原内联 __main__ 及未被调用的 analyze_gradient_conflicts/compute_gradient_cosine 已删除)。
