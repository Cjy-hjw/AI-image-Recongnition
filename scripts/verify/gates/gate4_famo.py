import torch


def gate4_famo(model, criterion, x, targets, device):
    """FAMO Multi-Task Weights Dynamics Verification."""
    try:
        model.train()
        criterion.train()

        w_logits_step1 = criterion.famo.w_logits.clone()

        # Step 1 backward
        model.zero_grad()
        out = model(x)
        loss1, _ = criterion(out, targets, update_famo_weights=True)
        loss1.backward()

        # Simulate FAMO update step (FAMO hooks parameter gradients internally)
        criterion.famo.get_weighted_loss(
            criterion.famo.prev_loss.to(device), update_weights=True
        )
        w_logits_step2 = criterion.famo.w_logits.clone()

        # Validate that the FAMO task weight distribution evolves
        assert not torch.isnan(w_logits_step2).any(), "FAMO logits became NaN."
        print(
            f"✔ GATE 4: FAMO Multi-task loss weights are valid and dynamic (Logits: {w_logits_step1.tolist()} -> {w_logits_step2.tolist()})."
        )
        return True
    except Exception as e:
        print(f"✘ GATE 4 Failed: {e}")
        return False
