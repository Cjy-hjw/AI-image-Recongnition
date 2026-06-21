import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalCLUBMean(nn.Module):
    """
    变分对比对数比例上界（vCLUB）估算器
    用于软解耦约束隐残差 z 与物理预测概念 C_pred 的重叠度
    """

    def __init__(self, x_dim=352, y_dim=18, hidden_size=64):
        super(VariationalCLUBMean, self).__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, y_dim),
        )

    def get_mu(self, x_samples):
        return self.p_mu(x_samples.float())

    def forward(self, x_samples, y_samples):
        """
        计算变分互信息上界 I(z; C_pred)
        使用 detach 权重前向，将互信息极小化惩罚限制在主干输入特征层，防止估计器参数暗中作弊
        """
        x_samples = x_samples.float()
        y_samples = y_samples.float()
        h = F.linear(
            x_samples, self.p_mu[0].weight.detach(), self.p_mu[0].bias.detach()
        )
        h = F.relu(h)
        mu = F.linear(h, self.p_mu[2].weight.detach(), self.p_mu[2].bias.detach())

        positive = -0.5 * torch.sum((mu - y_samples) ** 2, dim=-1)
        prediction_1 = mu.unsqueeze(1)  # [B, 1, y_dim]
        y_samples_1 = y_samples.unsqueeze(0)  # [1, B, y_dim]
        negative = -0.5 * torch.mean(
            torch.sum((y_samples_1 - prediction_1) ** 2, dim=-1), dim=1
        )
        return torch.mean(positive - negative)

    def learning_loss(self, x_samples, y_samples):
        """
        变分估计器自身更新损失
        """
        x_samples = x_samples.float()
        y_samples = y_samples.float()
        mu = self.get_mu(x_samples)
        return torch.mean(torch.sum((mu - y_samples) ** 2, dim=-1))


class ConceptEmbeddingCBMHead(nn.Module):
    """
    Concept Embedding Model (CEM) classifier head with Shannon Entropy Gate and physical prior fusion.
    """

    def __init__(self, in_features, num_concepts=18, emb_dim=16, latent_dim=64):
        super(ConceptEmbeddingCBMHead, self).__init__()
        self.num_concepts = num_concepts
        self.emb_dim = emb_dim

        self.proj_plus = nn.Sequential(
            nn.Linear(in_features, num_concepts * emb_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.proj_minus = nn.Sequential(
            nn.Linear(in_features, num_concepts * emb_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.s_layer = nn.Linear(2 * emb_dim, 1)

        self.register_buffer("ema_entropy_tau", torch.tensor(0.15))

        self.latent_mlp = nn.Sequential(
            nn.Linear(in_features, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, 2),
        )

        self.dropout = nn.Dropout(p=0.3)

        # Classifier direct path: 288 (concepts) + 16 (physical prior) = 304
        self.classifier_direct = nn.Sequential(
            nn.Linear(num_concepts * emb_dim + 16, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, x, e_prior=None):
        B = x.size(0)
        c_plus = self.proj_plus(x).view(B, self.num_concepts, self.emb_dim)
        c_minus = self.proj_minus(x).view(B, self.num_concepts, self.emb_dim)

        c_cat = torch.cat([c_plus, c_minus], dim=-1)  # [B, 18, 32]
        concept_logits = self.s_layer(c_cat).squeeze(-1)  # [B, 18]

        logits_comp = concept_logits[:, 0:4]
        logits_echo = concept_logits[:, 4:8]
        logits_shape = concept_logits[:, 8:10]
        logits_margin = concept_logits[:, 10:14]
        logits_foci = concept_logits[:, 14:18]

        prob_comp = F.softmax(logits_comp, dim=1)
        prob_echo = F.softmax(logits_echo, dim=1)
        prob_shape = F.softmax(logits_shape, dim=1)
        prob_margin = F.softmax(logits_margin, dim=1)
        prob_foci = torch.sigmoid(logits_foci)

        concept_vector = torch.cat(
            [prob_comp, prob_echo, prob_shape, prob_margin, prob_foci], dim=1
        )  # [B, 18]

        p = concept_vector.unsqueeze(-1)  # [B, 18, 1]
        c_mixed = p * c_plus + (1.0 - p) * c_minus  # [B, 18, 16]
        c_flat = c_mixed.view(B, -1)  # [B, 288]

        # Shannon Entropy (Grouped and normalized to [0, 1] space to avoid double-counting)
        LN2 = 0.6931471805599453
        log_prob_comp = F.log_softmax(logits_comp, dim=1)
        h_comp = -(prob_comp * log_prob_comp).sum(dim=1) / (2.0 * LN2)

        log_prob_echo = F.log_softmax(logits_echo, dim=1)
        h_echo = -(prob_echo * log_prob_echo).sum(dim=1) / (2.0 * LN2)

        log_prob_shape = F.log_softmax(logits_shape, dim=1)
        h_shape = -(prob_shape * log_prob_shape).sum(dim=1) / (1.0 * LN2)

        log_prob_margin = F.log_softmax(logits_margin, dim=1)
        h_margin = -(prob_margin * log_prob_margin).sum(dim=1) / (2.0 * LN2)

        log_prob_foci = F.logsigmoid(logits_foci)
        log_one_minus_prob_foci = F.logsigmoid(-logits_foci)
        h_foci_each = (
            -(prob_foci * log_prob_foci + (1.0 - prob_foci) * log_one_minus_prob_foci)
            / LN2
        )
        h_foci_norm = h_foci_each.mean(dim=1)

        mean_entropy = (h_comp + h_echo + h_shape + h_margin + h_foci_norm) / 5.0

        # EMA 动态中位数熵门控自适应更新，防止梯度死锁且保障良性泄漏
        if self.training:
            with torch.no_grad():
                batch_median_entropy = torch.median(mean_entropy)
                self.ema_entropy_tau.copy_(
                    0.9 * self.ema_entropy_tau + 0.1 * batch_median_entropy
                )

        alpha_raw = torch.sigmoid(10.0 * (mean_entropy - self.ema_entropy_tau))
        alpha_gate = alpha_raw * 0.95 + 0.05  # [B]

        latent_res = self.latent_mlp(x) * alpha_gate.unsqueeze(-1)  # [B, 2]
        if e_prior is None:
            e_prior = torch.zeros(B, 16, device=x.device)

        c_fused = torch.cat([c_flat, e_prior], dim=-1)  # [B, 304]
        c_fused = self.dropout(c_fused)
        logits_malignancy = self.classifier_direct(c_fused) + latent_res

        # W_norm dummy for compatibility, as loss_ortho will be set to 0 in loss.py
        W_norm = torch.eye(18, device=x.device)

        return (
            logits_comp,
            logits_echo,
            logits_shape,
            logits_margin,
            logits_foci,
            concept_vector,
            logits_malignancy,
            W_norm,
            latent_res,
        )

    def get_l1_penalty(self):
        return torch.sum(torch.abs(self.proj_plus[0].weight)) + torch.sum(
            torch.abs(self.proj_minus[0].weight)
        )
