import random
import torch


class PCGradWrapper:
    """Wraps a standard PyTorch optimizer with Projecting Conflicting Gradients (PCGrad) surgery.

    Usage:
        optimizer = PCGradWrapper(optim.AdamW(model.parameters(), lr=1e-4))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer._optim, T_max=80)

        # In train loop (no GradScaler needed; autocast forward still works):
        loss_cls = loss_dict["malignancy"] + loss_dict["comp"] + ...
        optimizer.pc_backward([loss_dict["seg"], loss_cls])
        optimizer.step()
    """

    def __init__(self, optimizer):
        self._optim = optimizer
        self.param_groups = optimizer.param_groups  # expose for scheduler compatibility

    def zero_grad(self):
        self._optim.zero_grad()

    def step(self):
        self._optim.step()

    def state_dict(self):
        return self._optim.state_dict()

    def load_state_dict(self, state):
        self._optim.load_state_dict(state)

    def pc_backward(self, losses):
        """Compute per-task gradients, project conflicting ones, merge, then write back.

        losses: list of scalar Tensors, one per task.
        Backward is plain (no GradScaler) — safe on A800 because autocast defaults to BF16,
        which doesn't require loss scaling.
        """
        shared = [p for g in self._optim.param_groups for p in g["params"] if p.requires_grad]
        task_grads = []

        for i, loss in enumerate(losses):
            self._optim.zero_grad()
            loss.backward(retain_graph=(i < len(losses) - 1))
            vec = []
            for p in shared:
                vec.append(
                    p.grad.data.clone().view(-1) if p.grad is not None
                    else torch.zeros(p.numel(), device=p.device)
                )
            task_grads.append(torch.cat(vec))

        # Gradient surgery: project conflicting components
        indices = list(range(len(losses)))
        random.shuffle(indices)
        for i in indices:
            for j in range(len(losses)):
                if i == j:
                    continue
                gj = task_grads[j]
                dot = torch.dot(task_grads[i], gj)
                if dot < 0:
                    task_grads[i] = task_grads[i] - (dot / (gj.dot(gj) + 1e-12)) * gj

        # Write merged gradients back to parameter .grad fields
        merged = sum(task_grads)
        offset = 0
        for p in shared:
            n = p.numel()
            p.grad = merged[offset:offset + n].view_as(p.data).clone()
            offset += n
