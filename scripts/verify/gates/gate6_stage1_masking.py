from src.training.stages import configure_stage_parameters


def gate6_stage1_masking(model, criterion, x, targets, config):
    """Stage 1 Gradient Masking Verification."""
    try:
        # Re-configure model for Stage 1 (epoch 1): freeze classification head, unfreeze seg+concepts
        lr_base = config["train"].get("lr", 1e-3)
        weight_decay = config["train"].get("weight_decay", 1e-4)
        configure_stage_parameters(
            model, criterion, epoch=1, lr_base=lr_base, weight_decay=weight_decay
        )

        model.train()
        criterion.train()
        model.zero_grad()

        # Forward pass
        out_s1 = model(x)
        # Compute loss with enable_mal_loss=False (stage 1 equivalence)
        loss_s1, loss_dict_s1 = criterion(
            out_s1, targets, update_famo_weights=True, enable_mal_loss=False
        )

        # Assert 6.1: classification losses are masked to 0.0
        assert (
            loss_dict_s1["malignancy"].item() == 0.0
        ), f"Stage 1 gradient masking FAILED: loss_mal = {loss_dict_s1['malignancy'].item()}, expected 0.0"
        assert (
            loss_dict_s1["supcon"].item() == 0.0
        ), f"Stage 1 gradient masking FAILED: loss_supcon = {loss_dict_s1['supcon'].item()}, expected 0.0"

        # Backward pass on malignancy and supcon ONLY to verify gradient masking
        loss_mal_only = loss_dict_s1["malignancy"] + loss_dict_s1["supcon"]
        loss_mal_only.backward()

        # Assert 6.2: no gradient leaks from masked classification/supcon losses to any parameter
        shared_leaked = False
        for name, p in model.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0.0:
                shared_leaked = True
                print(
                    f"  [STAGE1 LEAKAGE] '{name}' received grad norm = {p.grad.abs().sum().item()}"
                )
        assert (
            not shared_leaked
        ), "Stage 1 gradient masking FAILED: classification/supcon losses leaked gradients to backbone/heads!"
        print(
            "✔ GATE 6: Stage 1 gradient masking verified (loss_mal=0.0, loss_supcon=0.0, zero gradient leakage)."
        )
        return True
    except Exception as e:
        print(f"✘ GATE 6 Failed: {e}")
        return False
