import torch
import torch.nn.functional as F


def symmetric_label_regression_loss(
    logits,
    targets,
    num_classes,
    alpha=0.1,
    beta=1.0,
    eps=1e-4,
    label_smoothing=0.0,
):
    """
    多分类对称标签回归 (SLR) 损失，用于减轻临床主观标注的声学属性噪声。

    Args:
        logits: 模型预测 logits [B, C]
        targets: 整数类别标签 [B]
        num_classes: 类别数
        alpha: CE 损失权重
        beta: RCE 损失权重
        eps: 对数平滑项
        label_smoothing: 标签平滑系数 (0=不平滑, 0.1=10%平滑)
    """
    probs = F.softmax(logits, dim=1)
    # F.cross_entropy 内置 label_smoothing，对整数 targets 安全
    loss_ce = F.cross_entropy(
        logits,
        targets,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    # RCE 分支：平滑 one-hot targets 以匹配 CE 平滑语义
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    if label_smoothing > 0.0:
        smooth = label_smoothing / num_classes
        targets_one_hot = targets_one_hot * (1.0 - label_smoothing) + smooth
    loss_rce = -(probs * torch.log(targets_one_hot + eps)).sum(dim=1)
    return alpha * loss_ce + beta * loss_rce


def symmetric_label_regression_loss_bce(
    logits,
    targets,
    alpha=0.1,
    beta=1.0,
    eps=1e-4,
    label_smoothing=0.0,
):
    """
    二分类 (BCE) 对称标签回归 (SLR) 损失，用于 foci 等属性。

    Args:
        logits: 模型预测 logits [B, 1]
        targets: 二值标签 [B]
        alpha: BCE 损失权重
        beta: RCE 损失权重
        eps: 对数平滑项
        label_smoothing: 标签平滑系数
    """
    targets = targets.float()
    # BCE 标签平滑：将 hard 0/1 向 0.5 收缩
    if label_smoothing > 0.0:
        targets_smooth = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    else:
        targets_smooth = targets
    loss_ce = F.binary_cross_entropy_with_logits(
        logits, targets_smooth, reduction="none"
    )
    probs = torch.sigmoid(logits)
    loss_rce = -probs * torch.log(targets_smooth + eps) - (1.0 - probs) * torch.log(
        1.0 - targets_smooth + eps
    )
    loss_slr = alpha * loss_ce + beta * loss_rce
    return loss_slr.mean(dim=1)
