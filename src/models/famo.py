import torch
import torch.nn as nn


class FAMOScheduler:
    """Fast Adaptive Multi-Objective (FAMO) dynamic loss weighting.

    Optimizes task weight logits directly in loss space O(1) to balance
    multi-task learning without high-dimensional gradient conflicts.
    Compatible with GradScaler and mixed precision training.
    """

    def __init__(self, num_tasks=4, beta=0.05, gamma=0.001):
        self.num_tasks = num_tasks
        self.beta = beta  # Weight update step size
        self.gamma = gamma  # Weight regularization parameter
        self.w_logits = torch.zeros(num_tasks)
        self.prev_loss = torch.zeros(num_tasks)

    def get_weighted_loss(
        self, losses_tensor: torch.Tensor, update_weights: bool = True
    ) -> tuple:
        """Returns (total_weighted_loss, normalized_weights).

        losses_tensor: torch.Tensor of shape [num_tasks]
        """
        device = losses_tensor.device
        self.w_logits = self.w_logits.to(device)
        self.prev_loss = self.prev_loss.to(device)

        # Softmax normalized task weights
        weights = torch.softmax(self.w_logits, dim=0)

        if update_weights and self.prev_loss.sum() > 0:
            # Task improvement rate: (L_{t-1} - L_t) / L_{t-1} with 1e-4 epsilon and gradient clipping
            improvement = (self.prev_loss - losses_tensor.detach()) / (
                self.prev_loss + 1e-4
            )
            improvement = torch.clamp(improvement, min=-2.0, max=2.0)
            avg_improvement = improvement.mean()
            delta = improvement - avg_improvement

            # Dynamically adjust weights to maximize the worst-performing task
            self.w_logits = (
                self.w_logits + self.beta * delta - self.gamma * self.w_logits
            )
            # Clamp logits to prevent softmax overflow under extreme changes
            self.w_logits = torch.clamp(self.w_logits, min=-10.0, max=10.0)
            weights = torch.softmax(self.w_logits, dim=0)
        if update_weights:
            # Store current losses for next iteration
            self.prev_loss.copy_(losses_tensor.detach())

        # Weighted loss summation
        weighted_loss = (losses_tensor * weights).sum()
        return weighted_loss, weights
