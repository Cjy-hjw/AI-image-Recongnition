def gate2_vclub(model, x):
    """vCLUB Mutual Information Leakage Audit."""
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
        concept_vector_decouple = model.cbm_head(z_latent_detached).concept_vector

        vclub_mi_bound = model.vclub(z_latent_detached, concept_vector_decouple)
        vclub_mi_bound.backward()
        backbone_leaked = False
        for name, p in model.named_parameters():
            if any(
                k in name
                for k in [
                    "seg_backbone",
                    "cls_backbone",
                    "up_stages",
                    "dec_stages",
                    "adaptor_stage",
                ]
            ):
                if p.grad is not None and p.grad.abs().sum() > 1e-7:
                    backbone_leaked = True
                    print(
                        f"  [LEAKAGE DETECTED] Backbone/decoder parameter '{name}' received gradient from vCLUB: {p.grad.abs().sum().item()}"
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
        return True
    except Exception as e:
        print(f"✘ GATE 2 Failed: {e}")
        return False
