def vclub_learning_step(model, vclub_optimizer, prev_z_latent, prev_concept_vector):
    """
    vCLUB 独立优化：使用前一批次的 detached 输出，在 autocast 外执行独立 backward。
    """
    if prev_z_latent is None:
        return
    
    vclub_optimizer.zero_grad()
    vclub_learning_loss = model.vclub.learning_loss(
        prev_z_latent, prev_concept_vector
    )
    vclub_learning_loss.backward()
    vclub_optimizer.step()


def compute_vclub_mi_bound(model, outputs):
    """
    vCLUB 互信息极小化约束：全程启用，软正交隔离隐表征与属性概念。
    
    对输入 cbm_head 的 z_latent 分支进行 detach 计算条件概念向量，并 detach 该向量防止其梯度回传。
    计算互信息上界时，第一个 z_latent 参数必须保留计算图和梯度，以将退耦约束导流回 encoder (backbone)。
    """
    z_latent_detached = outputs["z_latent"].detach()
    concept_vector_decouple = model.cbm_head(z_latent_detached).concept_vector.detach()

    # 计算互信息上界时，第一个 z_latent 参数必须保留计算图和梯度
    vclub_mi_bound = model.vclub(outputs["z_latent"], concept_vector_decouple)
    return vclub_mi_bound
