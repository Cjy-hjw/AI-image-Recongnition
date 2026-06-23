import torch
import torch.nn as nn
import torch.nn.functional as F


class AUCMLossApprox(nn.Module):
    """
    AUC-Margin Loss with Lagrangian Minimax Dual Optimization
    Optimizes the pairwise ranking of positive and negative samples directly.
    """

    def __init__(self, margin=1.0, imratio=0.33):
        super().__init__()
        self.margin = margin
        self.p = imratio

        # Dual optimization multipliers
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2 and logits.shape[1] == 2:
            logits_diff = logits[:, 1] - logits[:, 0]
        else:
            logits_diff = logits.squeeze(-1)

        # 裁剪保护，防止 FP16 下平方运算发生溢出 NaN
        logits_diff = torch.clamp(logits_diff, min=-15.0, max=15.0)

        targets = targets.float()
        y_mapped = torch.where(
            targets > 0.5, torch.ones_like(targets), -torch.ones_like(targets)
        )

        pos_mask = (y_mapped == 1.0).float()
        neg_mask = (y_mapped == -1.0).float()

        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()

        if n_pos < 0.5 or n_neg < 0.5:
            # 保护：单类 batch 时返回 0，保持梯度图连接
            return logits_diff.sum() * 0.0

        # 正确的条件期望：除以各自类别的样本数量
        loss_pos = (1.0 - self.p) * (
            torch.sum(((logits_diff - self.a) ** 2) * pos_mask) / n_pos
        )
        loss_neg = self.p * (
            torch.sum(((logits_diff - self.b) ** 2) * neg_mask) / n_neg
        )

        mean_neg = torch.sum(logits_diff * neg_mask) / n_neg
        mean_pos = torch.sum(logits_diff * pos_mask) / n_pos

        # AIM: Adaptive Imbalance Margin algorithm
        prob_mal = torch.sigmoid(logits_diff)
        entropy_batch = -(
            prob_mal * torch.log2(prob_mal + 1e-5)
            + (1.0 - prob_mal) * torch.log2(1.0 - prob_mal + 1e-5)
        )
        # 必须 detach，防止梯度回传强行让分类概率接近 0.5 (最大熵)
        h_batch = entropy_batch.mean().detach()

        delta_pos = 1.0 / ((n_pos * (1.0 + 0.5 * h_batch)) ** 0.25)
        delta_neg = 1.0 / ((n_neg * (1.0 + 0.5 * h_batch)) ** 0.25)
        margin_adaptive = delta_pos + delta_neg

        # 绝对值保护，保证 alpha 乘子的符号方向正确
        alpha_val = torch.clamp(self.alpha, min=0.0)
        term_dual = 2.0 * alpha_val * (margin_adaptive + mean_neg - mean_pos)
        term_penalty = -self.p * (1.0 - self.p) * (alpha_val**2)

        return loss_pos + loss_neg + term_dual + term_penalty

    def step_dual_variable(self, phase="pre"):
        """
        Encapsulates the optimizer magic required for the Lagrangian multiplier.
        - 'pre' (before optimizer.step): Flips the gradient sign so alpha is maximized.
        - 'post' (after optimizer.step): Clamps alpha to [0, inf) non-negative constraint.
        """
        if phase == "pre":
            if self.alpha.grad is not None:
                self.alpha.grad.data.mul_(-1.0)
        elif phase == "post":
            with torch.no_grad():
                self.alpha.clamp_(min=0.0)
        else:
            raise ValueError("phase must be 'pre' or 'post'")


class SupervisedContrastiveLoss(nn.Module):
    """
    有监督对比学习损失 (SupCon Loss)：聚焦罕见恶性样本的特征空间汇聚。
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [B, 128] - 对比隐向量 z_contrastive
            labels: [B] - 良恶性金标准标签
        """
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        labels = labels.view(-1, 1)
        # 强制阈值化为硬标签以防 MixUp/CutMix 产生浮点数导致 eq 比较异常
        labels_hard = (labels > 0.5).long()
        # 寻找同类样本作为正样本
        mask = torch.eq(labels_hard, labels_hard.T).float().to(device)

        # L2 正则化校验
        features_norm = F.normalize(features, p=2, dim=1)

        # 计算余弦相似度张量并施加温度尺度
        similarity_matrix = (
            torch.matmul(features_norm, features_norm.T) / self.temperature
        )

        # 对角线置零，剔除锚点样本自身的比较
        logits_mask = torch.scatter(
            torch.ones_like(similarity_matrix),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # Log-Sum-Exp 保持数值稳定
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        sum_exp_logits = exp_logits.sum(dim=1, keepdim=True)

        log_prob = logits - torch.log(sum_exp_logits + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        loss = -mean_log_prob_pos
        return loss.mean()
