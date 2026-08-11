import torch


class FAMOScheduler:
    """Fast Adaptive Multi-Objective (FAMO) dynamic loss weighting.

    Optimizes task weight logits directly in loss space O(1) to balance
    multi-task learning without high-dimensional gradient conflicts.
    Compatible with GradScaler and mixed precision training.
    """

    def __init__(self, num_tasks=5, beta=0.05, gamma=0.001):
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

        if (
            update_weights
            and self.prev_loss.sum() > 0
            and not torch.isnan(losses_tensor).any()
        ):
            # Task improvement rate: (L_{t-1} - L_t) / L_{t-1} with 1e-2 epsilon and gradient clipping.
            # Epsilon widened 1e-4 -> 1e-2 (2026-08-11 R6 / DR §3.1): Dice loss decays toward ~1e-3
            # late in training, where a 1e-4 denominator makes the ratio explode on batch noise.
            improvement = (self.prev_loss - losses_tensor.detach()) / (
                torch.abs(self.prev_loss) + 1e-2
            )
            improvement = torch.clamp(improvement, min=-2.0, max=2.0)
            avg_improvement = improvement.mean()
            # 2026-08-11 R6 sign fix (DR report §2.2): delta = avg - I_k gives MORE weight to
            # slow-improving (under-optimized) tasks — this is the "save the worst task" direction.
            # The previous `improvement - avg_improvement` did the opposite (boosted the strongest
            # task), letting the seg backbone dominate the shared representation and dragging
            # classification AUC down to ~0.51-0.53. sum(avg - I) == 0 keeps logit mean from drifting.
            delta = avg_improvement - improvement

            # Dynamically adjust weights to maximize the worst-performing task
            self.w_logits = (
                self.w_logits + self.beta * delta - self.gamma * self.w_logits
            )
            # Clamp logits to prevent softmax overflow under extreme changes
            self.w_logits = torch.clamp(self.w_logits, min=-10.0, max=10.0)
            weights = torch.softmax(self.w_logits, dim=0)
        if update_weights and not torch.isnan(losses_tensor).any():
            # Store current losses for next iteration
            self.prev_loss.copy_(losses_tensor.detach())

        # Weighted loss summation. weights.detach() makes the scheduling coefficients
        # pure constants w.r.t. autograd: task weights are a hand-written update rule,
        # not learned by backprop. losses_tensor itself is NOT detached, so backbone /
        # task-head gradients still flow normally — only the weight scalars are frozen.
        # (w_logits is a plain tensor with requires_grad=False today, so no gradient is
        # injected currently; detach() makes that guarantee explicit and future-proof.)
        weighted_loss = (losses_tensor * weights.detach()).sum()
        return weighted_loss, weights

    def state_dict(self) -> dict:
        """Serialize dynamic state for checkpoint resume.

        Only the two dynamic tensors need persisting; hyperparameters
        (num_tasks/beta/gamma) are restored from __init__. Stored on CPU so the
        checkpoint is device-agnostic.
        """
        return {
            "w_logits": self.w_logits.detach().cpu().clone(),
            "prev_loss": self.prev_loss.detach().cpu().clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore dynamic state saved by ``state_dict``.

        Asserts task-count consistency so a checkpoint trained with a different
        number of FAMO channels fails loudly instead of broadcasting silently.
        """
        w = state["w_logits"]
        assert w.numel() == self.num_tasks, (
            f"FAMO state mismatch: checkpoint has {w.numel()} tasks, "
            f"scheduler expects {self.num_tasks}"
        )
        self.w_logits = w.clone()
        self.prev_loss = state["prev_loss"].clone()
