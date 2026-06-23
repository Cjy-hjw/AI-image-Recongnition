import torch

from src.models.multi_task_model import MultiTaskThyroidModel
from src.models.loss import MultiTaskLoss
from src.training.stages import (
    configure_stage_parameters,
    get_stage_number,
)


def gate7_optimizer_lr(backbone, config, device):
    """Rebuilt Optimizer LR Alignment Check."""
    try:
        # Re-initialize a clean model & criterion for optimizer tests to avoid state pollution
        model_lr = MultiTaskThyroidModel(backbone_name=backbone, pretrained=False).to(device)
        criterion_lr = MultiTaskLoss().to(device)

        lr_base = config["train"].get("lr", 1e-3)
        weight_decay = config["train"].get("weight_decay", 1e-4)

        test_epochs = [1, 41, 101]
        expected_stages = [1, 2, 3]

        for epoch_test, expected_stage in zip(test_epochs, expected_stages):
            param_groups, stage_lr, stage_name = configure_stage_parameters(
                model_lr,
                criterion_lr,
                epoch=epoch_test,
                lr_base=lr_base,
                weight_decay=weight_decay,
            )

            # Verify stage number
            assert (
                get_stage_number(epoch_test) == expected_stage
            ), f"Epoch {epoch_test} should be stage {expected_stage}, got {get_stage_number(epoch_test)}"

            # Build optimizer with param_groups and stage_lr
            optimizer_test = torch.optim.AdamW(param_groups, lr=stage_lr)

            # Assert 7.1: every param group has exactly stage_lr
            for pg_idx, pg in enumerate(optimizer_test.param_groups):
                assert pg["lr"] == stage_lr, (
                    f"Epoch {epoch_test} ({stage_name}): param_group[{pg_idx}] lr = {pg['lr']}, "
                    f"expected stage_lr = {stage_lr}"
                )

            print(
                f"✔ GATE 7: Epoch {epoch_test} ({stage_name}): all param groups lr = {stage_lr}"
            )

        print("✔ GATE 7: Optimizer LR alignment verified across all 3 stages.")
        return True
    except Exception as e:
        print(f"✘ GATE 7 Failed: {e}")
        return False
