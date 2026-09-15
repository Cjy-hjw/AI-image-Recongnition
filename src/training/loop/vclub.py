def vclub_learning_step(model, vclub_optimizer, prev_z_latent, prev_concept_vector):
    """
    vCLUB 独立优化：使用前一批次的 detached 输出，在 autocast 外执行独立 backward。

    vclub 参数全冻(如 seg_only 模式)时直接跳过:learning_loss 的输入是已
    detach 的 latent,唯一梯度来源就是 vclub 自身参数,全冻后 backward 必抛
    "does not require grad"(2026-09-14 Gate2 P0,端到端实测复现于
    --seg-only 冒烟第 2 个 batch)。跳过 = 单变量原则下 vclub 不更新。
    """
    if prev_z_latent is None:
        return
    if not any(p.requires_grad for p in model.vclub.parameters()):
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

    第一个 z_latent 参数（`outputs["z_latent"]`，保留计算图与梯度）把退耦约束导流回
    encoder (backbone)；第二个概念向量必须 detach 掐断梯度回传。

    ⚠️ 复用 `model()` 前向已算出的 `outputs["concept_vector"]`，**不再二次跑
    `model.cbm_head(...)`**（审查 S2）——二次前向会让 cbm_head 内的 EMA 熵门
    (`ema_entropy_tau`) 每 step 被推进两次（`alpha_gate` 收敛快 2×）且浪费算力。
    数值等价:concept_vector 在 e_prior 融合之前算出，与 z_latent 是否 detach、
    e_prior 是否 None 均无关，故复用值与 `cbm_head(z_latent.detach()).concept_vector`
    逐位相同；再 `.detach()` 即得所需的无梯度退耦向量。
    """
    concept_vector_decouple = outputs["concept_vector"].detach()

    vclub_mi_bound = model.vclub(outputs["z_latent"], concept_vector_decouple)
    return vclub_mi_bound
