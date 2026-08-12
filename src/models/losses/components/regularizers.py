import torch


def compute_regularizers(
    total_loss, outputs, lambda_latent, delta_budget, spin_l1_weight
):
    """Compute latent residual L2 (Lagrangian) constraint and SPIN L1 penalty.

    Returns (total_loss, loss_l2).
    """
    latent_res = outputs["latent_res"]
    loss_l2 = torch.mean(latent_res**2)
    # 2026-08-11 R6-sibling 符号核实(#120 review 误报后补注释):
    # 约束 g(x)=loss_l2-delta_budget <= 0,目标最小化,拉格朗日 L = f + lambda*g。
    # 参数方向: grad_p(L) = lambda * grad_p(loss_l2)。lambda>0 恒成立,loss_l2>=0,
    #   故梯度恒把残差往小推,与 g 的符号无关 —— 约束满足(g<=0)时取值变负但**不翻转方向**,
    #   g 只改变惩罚量级(违反时大力压、满足时弱压=自限性),绝无"把参数往错误方向推"。
    # 乘子方向(train.py:198 同步): lambda <- lambda + eta*g,违反加压/满足松绑,与对偶上升一致。
    # 注意:这里是"固定惩罚 + 下降率手调 lambda",lambda 不参与梯度,非交替对偶变量。
    loss_lagrangian = lambda_latent * (loss_l2 - delta_budget)

    total_loss = total_loss + loss_lagrangian + spin_l1_weight * outputs["spin_l1"]

    return total_loss, loss_l2
