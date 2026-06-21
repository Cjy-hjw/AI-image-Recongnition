import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.famo import FAMOScheduler
from src.models.losses.segmentation import DiceLoss
from src.models.losses.classification import AUCMLossApprox, SupervisedContrastiveLoss
from src.models.losses.concepts import (
    symmetric_label_regression_loss,
    symmetric_label_regression_loss_bce,
)


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss function.
    Balances segmentation BCE+Dice, TI-RADS concepts (via Symmetric Label Regression), and cost-sensitive malignancy classification
    using Homoscedastic Uncertainty Weighting and ACR TI-RADS prior regularization.
    """

    def __init__(self, num_tasks=5, alpha_prior=0.1, fn_weight=1.5):
        super().__init__()
        # Keep log_vars for checkpoint compatibility but use FAMO for optimization
        self.log_vars = nn.Parameter(torch.tensor([-0.5, 0.0, 0.0]))
        self.alpha_prior = alpha_prior  # Clinical prior regularization factor
        self.mal_loss_fn = AUCMLossApprox(margin=1.0, imratio=0.33)
        self.famo = FAMOScheduler(num_tasks=5, beta=0.05, gamma=0.001)
        self.supcon_loss = SupervisedContrastiveLoss(temperature=0.07)

        self.bce_seg = nn.BCEWithLogitsLoss()
        self.dice_seg = DiceLoss()

        # ACR TI-RADS weights matrix
        tirads_weights = [
            0.0,
            0.0,
            1.0,
            2.0,  # Composition
            0.0,
            1.0,
            2.0,
            3.0,  # Echogenicity
            0.0,
            3.0,  # Shape
            0.0,
            0.0,
            2.0,
            3.0,  # Margin
            0.0,
            1.0,
            2.0,
            3.0,  # Foci
        ]
        self.register_buffer(
            "tirads_matrix", torch.tensor(tirads_weights, dtype=torch.float32)
        )
        self.register_buffer("lambda_latent", torch.tensor(0.01))
        self.delta_budget = 0.1

    def forward(
        self,
        outputs,
        targets,
        slr_weights=None,
        mix_info=None,
        update_famo_weights=True,
        spin_l1_weight=0.0,
        label_smoothing=0.0,
        stage=2,
    ):
        # 1. Segmentation Loss
        loss_bce_seg = self.bce_seg(outputs["mask"], targets["mask"])
        loss_dice_seg = self.dice_seg(outputs["mask"], targets["mask"])
        loss_seg = loss_bce_seg + loss_dice_seg
        # NaN Sentinel: detect which component produces NaN first
        if torch.isnan(loss_seg):
            print(
                f"  [NaN Guard] loss_seg is NaN! bce={loss_bce_seg.item()}, dice={loss_dice_seg.item()}, "
                f"mask range=[{outputs['mask'].min().item():.4f}, {outputs['mask'].max().item():.4f}]",
                flush=True,
            )

        # Determine mixing parameters
        batch_size = outputs["mask"].size(0)
        device = outputs["mask"].device

        if mix_info is not None:
            lam = mix_info["lam"]  # Shape [B] or scalar
            # Ensure lam is a tensor of shape [B]
            if isinstance(lam, float) or lam.dim() == 0:
                lam = torch.full((batch_size,), float(lam), device=device)
            index = mix_info["index"]
            mask_weight = mix_info.get(
                "mask_weight", torch.ones(batch_size, device=device)
            )
        else:
            lam = torch.ones(batch_size, device=device)
            index = torch.arange(batch_size, device=device)
            mask_weight = torch.ones(batch_size, device=device)

        # 2. Concept classification losses (computed sample-wise using SLR to defend against label noise)
        slr_alpha = 0.1
        slr_beta = 1.0

        loss_comp = lam * symmetric_label_regression_loss(
            outputs["composition"],
            targets["composition"],
            num_classes=4,
            alpha=slr_alpha,
            beta=slr_beta,
            label_smoothing=label_smoothing,
        ) + (1.0 - lam) * symmetric_label_regression_loss(
            outputs["composition"],
            targets["composition"][index],
            num_classes=4,
            alpha=slr_alpha,
            beta=slr_beta,
            label_smoothing=label_smoothing,
        )
        loss_echo = lam * symmetric_label_regression_loss(
            outputs["echogenicity"],
            targets["echogenicity"],
            num_classes=4,
            alpha=slr_alpha,
            beta=slr_beta,
            label_smoothing=label_smoothing,
        ) + (1.0 - lam) * symmetric_label_regression_loss(
            outputs["echogenicity"],
            targets["echogenicity"][index],
            num_classes=4,
            alpha=slr_alpha,
            beta=slr_beta,
            label_smoothing=label_smoothing,
        )
        loss_shape = lam * symmetric_label_regression_loss(
            outputs["shape"],
            targets["shape"],
            num_classes=2,
            alpha=slr_alpha,
            beta=slr_beta,
            label_smoothing=label_smoothing,
        ) + (1.0 - lam) * symmetric_label_regression_loss(
            outputs["shape"],
            targets["shape"][index],
            num_classes=2,
            alpha=slr_alpha,
            beta=slr_beta,
            label_smoothing=label_smoothing,
        )
        loss_margin = lam * symmetric_label_regression_loss(
            outputs["margin"],
            targets["margin"],
            num_classes=4,
            alpha=slr_alpha,
            beta=slr_beta,
            label_smoothing=label_smoothing,
        ) + (1.0 - lam) * symmetric_label_regression_loss(
            outputs["margin"],
            targets["margin"][index],
            num_classes=4,
            alpha=slr_alpha,
            beta=slr_beta,
            label_smoothing=label_smoothing,
        )
        loss_foci = lam * symmetric_label_regression_loss_bce(
            outputs["foci"],
            targets["foci"],
            alpha=slr_alpha,
            beta=slr_beta,
            label_smoothing=label_smoothing,
        ) + (1.0 - lam) * symmetric_label_regression_loss_bce(
            outputs["foci"],
            targets["foci"][index],
            alpha=slr_alpha,
            beta=slr_beta,
            label_smoothing=label_smoothing,
        )

        loss_concepts_ce = loss_comp + loss_echo + loss_shape + loss_margin + loss_foci

        # 2b. ACR TI-RADS clinical prior regularization
        prob_comp = F.softmax(outputs["composition"], dim=1)
        prob_echo = F.softmax(outputs["echogenicity"], dim=1)
        prob_shape = F.softmax(outputs["shape"], dim=1)
        prob_margin = F.softmax(outputs["margin"], dim=1)
        prob_foci = torch.sigmoid(outputs["foci"])

        pred_concepts = torch.cat(
            [prob_comp, prob_echo, prob_shape, prob_margin, prob_foci], dim=1
        )

        gt_comp = F.one_hot(targets["composition"], num_classes=4).float()
        gt_echo = F.one_hot(targets["echogenicity"], num_classes=4).float()
        gt_shape = F.one_hot(targets["shape"], num_classes=2).float()
        gt_margin = F.one_hot(targets["margin"], num_classes=4).float()
        gt_foci = targets["foci"].float()
        gt_concepts = torch.cat([gt_comp, gt_echo, gt_shape, gt_margin, gt_foci], dim=1)

        gt_comp_perm = F.one_hot(targets["composition"][index], num_classes=4).float()
        gt_echo_perm = F.one_hot(targets["echogenicity"][index], num_classes=4).float()
        gt_shape_perm = F.one_hot(targets["shape"][index], num_classes=2).float()
        gt_margin_perm = F.one_hot(targets["margin"][index], num_classes=4).float()
        gt_foci_perm = targets["foci"][index].float()
        gt_concepts_perm = torch.cat(
            [gt_comp_perm, gt_echo_perm, gt_shape_perm, gt_margin_perm, gt_foci_perm],
            dim=1,
        )

        # Reconstruct clinical score
        score_pred = torch.mv(pred_concepts, self.tirads_matrix)
        score_gt = lam * torch.mv(gt_concepts, self.tirads_matrix) + (
            1.0 - lam
        ) * torch.mv(gt_concepts_perm, self.tirads_matrix)

        loss_prior = F.huber_loss(score_pred, score_gt, delta=1.0)

        # Decouple concepts CE and prior for independent FAMO channels
        loss_concepts_ce = loss_concepts_ce * mask_weight
        loss_prior = loss_prior * mask_weight

        # 3. AVS-Loss malignancy loss (computed sample-wise)
        mal_logits = outputs["malignancy"]
        mal_targets = targets["malignancy"]

        # 对于对偶变量 minimax (a, b, alpha) 的更新，使用未插值真实标签防止扰动污染
        loss_mal = self.mal_loss_fn(mal_logits, mal_targets)
        loss_mal = loss_mal * mask_weight

        # SupCon有监督对比损失：独立FAMO通道，不混入AUCM损失
        loss_supcon = self.supcon_loss(outputs["z_contrastive"], mal_targets)

        # Apply Supervised Label Rectification weights if provided
        if slr_weights is not None:
            loss_mal = loss_mal * slr_weights

        # Stage 1: 分类分支完全冻结，将分类损失与对比损失乘以 0.0，阻断梯度泄露回 shared backbone
        if stage == 1:
            loss_mal = loss_mal * 0.0
            loss_supcon = loss_supcon * 0.0

        # Reduce all FAMO channels to scalar
        loss_concepts_ce = loss_concepts_ce.mean()
        loss_prior = loss_prior.mean()
        loss_mal = loss_mal.mean()
        # loss_supcon: ensure scalar (SupervisedContrastiveLoss typically returns scalar)
        if loss_supcon.dim() > 0:
            loss_supcon = loss_supcon.mean()

        # Store for external access
        self.loss_concepts_ce = loss_concepts_ce
        self.loss_prior = loss_prior

        # NaN Sentinel: all 5 FAMO channels
        if torch.isnan(loss_concepts_ce):
            print(f"  [NaN Guard] loss_concepts_ce is NaN!", flush=True)
        if torch.isnan(loss_prior):
            print(f"  [NaN Guard] loss_prior is NaN!", flush=True)
        if torch.isnan(loss_mal):
            print(
                f"  [NaN Guard] loss_mal is NaN! mal_logits range=[{mal_logits.min().item():.4f}, {mal_logits.max().item():.4f}]",
                flush=True,
            )
        if torch.isnan(loss_supcon):
            print(f"  [NaN Guard] loss_supcon is NaN!", flush=True)

        # 4. FAMO 5-channel dynamic loss weighting
        #   Ch1: loss_seg (BCE + Dice segmentation)
        #   Ch2: loss_concepts_ce (concept SLR: comp+echo+shape+margin+foci)
        #   Ch3: loss_prior (TI-RADS clinical prior Huber)
        #   Ch4: loss_mal (AUCMLoss malignancy classification)
        #   Ch5: loss_supcon (supervised contrastive on backbone features)
        losses_tensor = torch.stack(
            [loss_seg, loss_concepts_ce, loss_prior, loss_mal, loss_supcon]
        )
        total_loss, weights = self.famo.get_weighted_loss(
            losses_tensor, update_weights=update_famo_weights and self.training
        )
        # NaN Sentinel: FAMO weighted loss
        if torch.isnan(total_loss):
            print(
                f"  [NaN Guard] total_loss is NaN! seg={loss_seg.item():.4f}, "
                f"concepts_ce={loss_concepts_ce.item():.4f}, prior={loss_prior.item():.4f}, "
                f"mal={loss_mal.item():.4f}, supcon={loss_supcon.item():.4f}",
                flush=True,
            )
            # Fallback: bypass FAMO and use simple sum to keep training alive
            total_loss = (
                loss_seg + loss_concepts_ce + loss_prior + loss_mal + loss_supcon
            )

        # 5. CBM 物理空间正交惩罚及隐残差 L2 范数约束 (重构为拉格朗日约束)
        # These regularization terms are NOT scheduled by FAMO — added directly
        loss_ortho = torch.tensor(0.0, device=total_loss.device)
        latent_res = outputs["latent_res"]
        loss_l2 = torch.mean(latent_res**2)
        loss_lagrangian = self.lambda_latent * (loss_l2 - self.delta_budget)

        total_loss = (
            total_loss
            + 0.01 * loss_ortho
            + loss_lagrangian
            + spin_l1_weight * outputs["spin_l1"]
        )

        loss_dict = {
            "total": total_loss,
            "seg": loss_seg,
            "concepts_ce": loss_concepts_ce,
            "prior": loss_prior,
            "comp": loss_comp.mean(),
            "echo": loss_echo.mean(),
            "shape": loss_shape.mean(),
            "margin": loss_margin.mean(),
            "foci": loss_foci.mean(),
            "malignancy": loss_mal,
            "supcon": loss_supcon,
            "ortho": loss_ortho,
            "latent_l2": loss_l2,
            "spin_l1": outputs["spin_l1"],
        }

        return total_loss, loss_dict
