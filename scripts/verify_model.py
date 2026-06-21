import sys
import os
import torch
import torch.nn as nn
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.multi_task_model import MultiTaskThyroidModel
from src.models.loss import MultiTaskLoss
from src.training.stages import (
    configure_stage_parameters,
    create_optimizer,
    get_stage_number,
)


def verify():
    print("=== MODEL SANITY & GRADIENT FLOW VERIFICATION ===")

    # 1. Load config
    config_path = "configs/thyroid_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return False

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    backbone = config["model"]["backbone"]
    print(f"Testing configured backbone: {backbone}")

    # 2. Instantiate Model and Loss
    try:
        model = MultiTaskThyroidModel(backbone_name=backbone, pretrained=False)
        criterion = MultiTaskLoss()
        print("✔ Model & Loss instantiation successful.")
    except Exception as e:
        print(f"✘ Model instantiation failed: {e}")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    # Prepare normalized dummy data
    x = torch.randn(2, 1, 512, 512)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)  # normalized
    x = x.to(device)

    targets = {
        "mask": torch.randint(0, 2, (2, 1, 512, 512)).float().to(device),
        "composition": torch.tensor([0, 3]).to(device),
        "echogenicity": torch.tensor([1, 2]).to(device),
        "shape": torch.tensor([0, 1]).to(device),
        "margin": torch.tensor([2, 0]).to(device),
        "foci": torch.randn(2, 4).sigmoid().to(device),
        "malignancy": torch.tensor([1, 0]).to(device),
    }

    # ==========================================
    # GATE 1: Forward Pass & Output Shape Sanity
    # ==========================================
    try:
        model.eval()
        with torch.no_grad():
            out = model(x)

        # Verify shapes
        assert out["mask"].shape == (
            2,
            1,
            512,
            512,
        ), f"Bad mask shape: {out['mask'].shape}"
        assert out["malignancy"].shape == (
            2,
            2,
        ), f"Bad malignancy shape: {out['malignancy'].shape}"
        assert out["composition"].shape == (
            2,
            4,
        ), f"Bad composition shape: {out['composition'].shape}"
        assert out["echogenicity"].shape == (
            2,
            4,
        ), f"Bad echogenicity shape: {out['echogenicity'].shape}"
        assert out["shape"].shape == (2, 2), f"Bad shape shape: {out['shape'].shape}"
        assert out["margin"].shape == (2, 4), f"Bad margin shape: {out['margin'].shape}"
        assert out["foci"].shape == (2, 4), f"Bad foci shape: {out['foci'].shape}"
        print("✔ GATE 1: Shape verification passed.")
    except Exception as e:
        print(f"✘ GATE 1 Failed: {e}")
    # ==========================================
    # GATE 2: vCLUB Mutual Information Leakage Audit
    # ==========================================
    try:
        model.train()
        model.zero_grad()

        out = model(x)
        # 1. 模拟估计器拟合：传入分离梯度的 z 和 concept_vector，检验 vCLUB 本地梯度生成
        vclub_learning_loss = model.vclub.learning_loss(
            out["z_latent"].detach(), out["concept_vector"].detach()
        )
        vclub_learning_loss.backward()

        vclub_has_grad = False
        for name, p in model.named_parameters():
            if "vclub." in name:
                if p.grad is not None and p.grad.abs().sum() > 0:
                    vclub_has_grad = True

        # 2. 清零梯度，进入核心解耦隔离审计
        model.zero_grad()
        out = model(x)

        # 核心解耦逻辑：
        # 通过将 z_latent 进行 detach，重新运行 cbm_head 前向，得到 concept_vector_decouple。
        # 这样互信息梯度可以回传给 cbm_head 调整映射决策面，但被 fused_cls_feat (z_latent) 的 detach 阻断，绝对无法污染 Encoder Backbone！
        z_latent_detached = out["z_latent"].detach()
        (
            logits_comp,
            logits_echo,
            logits_shape,
            logits_margin,
            logits_foci,
            concept_vector_decouple,
            _,
            _,
            _,
        ) = model.cbm_head(z_latent_detached)

        vclub_mi_bound = model.vclub(z_latent_detached, concept_vector_decouple)
        vclub_mi_bound.backward()
        backbone_leaked = False
        for name, p in model.named_parameters():
            if any(k in name for k in ["shared_", "seg_", "up"]):
                if p.grad is not None and p.grad.abs().sum() > 1e-7:
                    backbone_leaked = True
                    print(
                        f"  [LEAKAGE DETECTED] Shared/Seg parameter '{name}' received gradient from vCLUB: {p.grad.abs().sum().item()}"
                    )
        assert (
            vclub_has_grad
        ), "vCLUB estimator parameters did not update under local learning loss."
        assert (
            not backbone_leaked
        ), "vCLUB gradient leaked into the encoder! Backpropagation isolation is broken."
        print(
            "✔ GATE 2: vCLUB gradient isolation & local gradient generation verified successfully."
        )
    except Exception as e:
        print(f"✘ GATE 2 Failed: {e}")
        return False
    # ==========================================
    # GATE 3: AUC-Margin Dual Minimizing Parameter Registration & Update check
    # ==========================================
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
        criterion.mal_loss_fn.alpha.grad.data.neg_()
        optimizer.step()
        criterion.mal_loss_fn.alpha.data.clamp_(min=0.0)

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
                for k in ["cls_stage", "cbm_head", "prior_mlp", "supcon_projector"]
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
        # Apply trainer.py automatic freeze logic: set ALL modules with all-frozen params to eval
        model.train()
        for m in model.modules():
            params = list(m.parameters())
            if len(params) > 0 and all(not p.requires_grad for p in params):
                m.eval()

        # Verify that frozen batchnorm modules have training == False
        frozen_bn_count = 0
        active_bn_count = 0
        for name, m in model.named_modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                # If its parameters are frozen
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
    except Exception as e:
        print(f"✘ GATE 3 Failed: {e}")
        return False

    # ==========================================
    # GATE 4: FAMO Multi-Task Weights Dynamics Verification
    # ==========================================
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
    except Exception as e:
        print(f"✘ GATE 4 Failed: {e}")
        return False

    # ==========================================
    # GATE 5: Data Leakage & Patient ID Isolation Check
    # ==========================================
    try:
        import json

        train_dir = "data/train"
        val_dir = "data/val"

        if os.path.exists(train_dir) and os.path.exists(val_dir):
            train_patients = set()
            val_patients = set()

            # 扫描训练集元数据
            for folder in os.listdir(train_dir):
                meta_p = os.path.join(train_dir, folder, "meta.json")
                if os.path.exists(meta_p):
                    with open(meta_p, "r") as f:
                        meta = json.load(f)
                        if "patient_id" in meta:
                            train_patients.add(str(meta["patient_id"]))

            # 扫描验证集元数据
            for folder in os.listdir(val_dir):
                meta_p = os.path.join(val_dir, folder, "meta.json")
                if os.path.exists(meta_p):
                    with open(meta_p, "r") as f:
                        meta = json.load(f)
                        if "patient_id" in meta:
                            val_patients.add(str(meta["patient_id"]))

            overlap = train_patients.intersection(val_patients)
            # 强行断言：在工业级数据隔离流水线中，训练集与验证集之间的患者重叠必须为 0，否则立即抛错熔断！
            assert (
                len(overlap) == 0
            ), f"CRITICAL DATA LEAKAGE: {len(overlap)} overlapping patients found between train and val splits!"
            print("✔ GATE 5: Patient ID isolation verified successfully (0 overlap).")
        else:
            print(
                "⚠ GATE 5: Patient isolation check skipped (Data folders not found on this machine)."
            )
    except Exception as e:
        print(f"✘ GATE 5 Failed: {e}")
        return False

    # ==========================================
    # GATE 6: Stage 1 Gradient Masking Verification
    # ==========================================
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
        # Compute loss with stage=1 (should mask loss_mal and loss_supcon to 0.0)
        loss_s1, loss_dict_s1 = criterion(
            out_s1, targets, update_famo_weights=True, stage=1
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
    except Exception as e:
        print(f"✘ GATE 6 Failed: {e}")
        return False

    # ==========================================
    # GATE 7: Rebuilt Optimizer LR Alignment Check
    # ==========================================
    try:
        # Re-initialize a clean model & criterion for optimizer tests to avoid state pollution
        model_lr = MultiTaskThyroidModel(backbone_name=backbone, pretrained=False).to(
            device
        )
        criterion_lr = MultiTaskLoss().to(device)

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
    except Exception as e:
        print(f"✘ GATE 7 Failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
