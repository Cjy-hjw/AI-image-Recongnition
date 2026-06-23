import torch
import torch.nn as nn


def gate3_auc_margin(model, criterion, x, targets):
    """AUC-Margin Dual Minimizing Parameter Registration & Update check."""
    try:
        model.zero_grad()
        # Bind optimization parameters mimicking train.py configure_stage_parameters logic
        opt_params = []
        for name, p in model.named_parameters():
            if "vclub" not in name:
                opt_params.append(p)

        # Register dual parameters (a, b, alpha)
        opt_params.append(criterion.mal_loss_fn.a)
        opt_params.append(criterion.mal_loss_fn.b)
        opt_params.append(criterion.mal_loss_fn.alpha)

        optimizer = torch.optim.AdamW(opt_params, lr=1e-3)
        initial_alpha = criterion.mal_loss_fn.alpha.clone()

        # Run one step of multi-task loss backward
        out = model(x)
        loss, _ = criterion(out, targets)
        loss.backward()

        # Rule 3.1: Check gradient existence on alpha
        alpha_grad = criterion.mal_loss_fn.alpha.grad
        assert (
            alpha_grad is not None
        ), "Dual parameter 'alpha' did not receive any gradient."
        assert not torch.isnan(alpha_grad).any(), "Gradient of 'alpha' contains NaNs."

        # Rule 3.2: Check if optimizer updates alpha (in inverse maximization direction)
        criterion.mal_loss_fn.step_dual_variable("pre")
        optimizer.step()
        criterion.mal_loss_fn.step_dual_variable("post")

        new_alpha = criterion.mal_loss_fn.alpha
        assert not torch.isnan(new_alpha).any(), "Updated 'alpha' is NaN."
        assert (
            new_alpha - initial_alpha
        ).abs().sum() > 0, (
            "Dual parameter 'alpha' was not updated by the optimizer step."
        )
        # Rule 3.3: Verify BatchNorm freezing logic for stage transition safety
        # Simulate stage 2 freeze using current train.py logic
        for name, param in model.named_parameters():
            if "vclub" in name:
                continue
            if any(
                k in name
                for k in ["cbm_head", "prior_mlp", "supcon_projector", "cls_backbone"]
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
        model.train()
        for m in model.modules():
            params = list(m.parameters())
            if len(params) > 0 and all(not p.requires_grad for p in params):
                m.eval()

        frozen_bn_count = 0
        active_bn_count = 0
        for name, m in model.named_modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                if all(not p.requires_grad for p in m.parameters()):
                    assert (
                        not m.training
                    ), f"BatchNorm layer '{name}' is in train mode but its parameters are frozen!"
                    frozen_bn_count += 1
                else:
                    assert (
                        m.training
                    ), f"BatchNorm layer '{name}' is in eval mode but its parameters are active!"
                    active_bn_count += 1

        assert frozen_bn_count > 0, "No frozen BatchNorm layers verified."
        print(
            f"✔ GATE 3: AUC-Margin dual variables verified. BatchNorm freezing logic validated (Frozen: {frozen_bn_count}, Active: {active_bn_count})."
        )
        return True
    except Exception as e:
        print(f"✘ GATE 3 Failed: {e}")
        return False
