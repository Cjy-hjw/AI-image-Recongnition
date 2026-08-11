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

        # 2026-08-11 R6: DIRECTION assertion (not just "changed"). The old gate only
        # checked non-NaN, which is exactly why the sign-reversal bug survived. Feed a
        # controlled two-task update — task0 improves fast, task1 improves slowly — and
        # assert the SLOW (under-optimized) task gets MORE weight ("save the worst task").
        from src.models.famo import FAMOScheduler

        probe = FAMOScheduler(num_tasks=2, beta=0.05, gamma=0.0)
        probe.prev_loss = torch.tensor([1.0, 1.0], device=device)
        _, probe_w = probe.get_weighted_loss(
            torch.tensor([0.5, 0.95], device=device), update_weights=True
        )
        assert probe.w_logits[1] > probe.w_logits[0], (
            "FAMO direction reversed: slow-improving task must get a HIGHER logit "
            f"than the fast task (got slow={probe.w_logits[1].item():.5f}, "
            f"fast={probe.w_logits[0].item():.5f})."
        )
        assert probe_w[1] > 0.5, (
            "FAMO direction reversed: slow-improving task weight must exceed 0.5 "
            f"(got {probe_w[1].item():.5f})."
        )
        print(
            f"✔ GATE 4: FAMO Multi-task loss weights are valid, dynamic, and directionally "
            f"correct (Logits: {w_logits_step1.tolist()} -> {w_logits_step2.tolist()}; "
            f"direction probe slow-task weight={probe_w[1].item():.4f} > 0.5)."
        )
        return True
    except Exception as e:
        print(f"✘ GATE 4 Failed: {e}")
        return False
