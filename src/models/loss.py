import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Sorensen-Dice loss for binary image segmentation."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # pred shape: [B, 1, H, W], logits
        pred_prob = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        denominator = pred_flat.sum() + target_flat.sum()
        
        dice_coeff = (2. * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice_coeff

class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss function.
    Balances segmentation BCE+Dice, TI-RADS concepts, and cost-sensitive malignancy classification
    using Homoscedastic Uncertainty Weighting and ACR TI-RADS prior regularization.
    """
    def __init__(self, num_tasks=3, alpha_prior=0.1, fn_weight=1.5):
        super().__init__()
        # 3 learnable task uncertainty weights: 0 -> seg, 1 -> concepts, 2 -> malignancy
        self.log_vars = nn.Parameter(torch.tensor([-0.5, 0.0, 0.0]))
        self.alpha_prior = alpha_prior  # Clinical prior regularization factor
        self.fn_weight = fn_weight      # Penalty factor for False Negatives in malignancy
        
        self.bce_seg = nn.BCEWithLogitsLoss()
        self.dice_seg = DiceLoss()
        
        # ACR TI-RADS weights matrix
        tirads_weights = [
            0.0, 0.0, 1.0, 2.0,  # Composition
            0.0, 1.0, 2.0, 3.0,  # Echogenicity
            0.0, 3.0,            # Shape
            0.0, 0.0, 2.0, 3.0,  # Margin
            0.0, 1.0, 2.0, 3.0   # Foci
        ]
        self.register_buffer("tirads_matrix", torch.tensor(tirads_weights, dtype=torch.float32))

    def forward(self, outputs, targets, slr_weights=None, mix_info=None):
        # 1. Segmentation Loss
        loss_bce_seg = self.bce_seg(outputs["mask"], targets["mask"])
        loss_dice_seg = self.dice_seg(outputs["mask"], targets["mask"])
        loss_seg = loss_bce_seg + loss_dice_seg
        
        # Determine mixing parameters
        batch_size = outputs["mask"].size(0)
        device = outputs["mask"].device
        
        if mix_info is not None:
            lam = mix_info["lam"]  # Shape [B] or scalar
            # Ensure lam is a tensor of shape [B]
            if isinstance(lam, float) or lam.dim() == 0:
                lam = torch.full((batch_size,), float(lam), device=device)
            index = mix_info["index"]
            mask_weight = mix_info.get("mask_weight", torch.ones(batch_size, device=device))
        else:
            lam = torch.ones(batch_size, device=device)
            index = torch.arange(batch_size, device=device)
            mask_weight = torch.ones(batch_size, device=device)
            
        # 2. Concept classification losses (computed sample-wise)
        loss_comp = lam * F.cross_entropy(outputs["composition"], targets["composition"], reduction='none') + \
                    (1.0 - lam) * F.cross_entropy(outputs["composition"], targets["composition"][index], reduction='none')
        loss_echo = lam * F.cross_entropy(outputs["echogenicity"], targets["echogenicity"], reduction='none') + \
                    (1.0 - lam) * F.cross_entropy(outputs["echogenicity"], targets["echogenicity"][index], reduction='none')
        loss_shape = lam * F.cross_entropy(outputs["shape"], targets["shape"], reduction='none') + \
                     (1.0 - lam) * F.cross_entropy(outputs["shape"], targets["shape"][index], reduction='none')
        loss_margin = lam * F.cross_entropy(outputs["margin"], targets["margin"], reduction='none') + \
                      (1.0 - lam) * F.cross_entropy(outputs["margin"], targets["margin"][index], reduction='none')
        loss_foci = lam * F.binary_cross_entropy_with_logits(outputs["foci"], targets["foci"], reduction='none').mean(dim=1) + \
                    (1.0 - lam) * F.binary_cross_entropy_with_logits(outputs["foci"], targets["foci"][index], reduction='none').mean(dim=1)
                    
        loss_concepts_ce = loss_comp + loss_echo + loss_shape + loss_margin + loss_foci
        
        # 2b. ACR TI-RADS clinical prior regularization
        prob_comp = F.softmax(outputs["composition"], dim=1)
        prob_echo = F.softmax(outputs["echogenicity"], dim=1)
        prob_shape = F.softmax(outputs["shape"], dim=1)
        prob_margin = F.softmax(outputs["margin"], dim=1)
        prob_foci = torch.sigmoid(outputs["foci"])
        
        pred_concepts = torch.cat([prob_comp, prob_echo, prob_shape, prob_margin, prob_foci], dim=1)
        
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
        gt_concepts_perm = torch.cat([gt_comp_perm, gt_echo_perm, gt_shape_perm, gt_margin_perm, gt_foci_perm], dim=1)
        
        # Reconstruct clinical score
        score_pred = torch.mv(pred_concepts, self.tirads_matrix)
        score_gt = lam * torch.mv(gt_concepts, self.tirads_matrix) + (1.0 - lam) * torch.mv(gt_concepts_perm, self.tirads_matrix)
        
        loss_prior = (score_pred - score_gt) ** 2
        
        loss_concepts = loss_concepts_ce + self.alpha_prior * loss_prior
        
        # Apply mask_weight
        loss_concepts = (loss_concepts * mask_weight)
        
        # 3. Cost-Sensitive NLL Loss for malignancy (computed sample-wise)
        mal_logits = outputs["malignancy"]
        mal_targets = targets["malignancy"]
        log_prob = F.log_softmax(mal_logits, dim=1)
        
        # Original targets loss
        loss_mal_raw_orig = F.nll_loss(log_prob, mal_targets, reduction='none')
        preds_orig = torch.argmax(mal_logits, dim=1)
        fn_mask_orig = (mal_targets == 1) & (preds_orig == 0)
        penalty_weights_orig = torch.ones_like(loss_mal_raw_orig, dtype=torch.float32)
        penalty_weights_orig[fn_mask_orig] = self.fn_weight
        loss_mal_orig = loss_mal_raw_orig * penalty_weights_orig
        
        # Permuted targets loss
        mal_targets_perm = mal_targets[index]
        loss_mal_raw_perm = F.nll_loss(log_prob, mal_targets_perm, reduction='none')
        fn_mask_perm = (mal_targets_perm == 1) & (preds_orig == 0)
        penalty_weights_perm = torch.ones_like(loss_mal_raw_perm, dtype=torch.float32)
        penalty_weights_perm[fn_mask_perm] = self.fn_weight
        loss_mal_perm = loss_mal_raw_perm * penalty_weights_perm
        
        loss_mal = lam * loss_mal_orig + (1.0 - lam) * loss_mal_perm
        loss_mal = (loss_mal * mask_weight)
        # Apply Supervised Label Rectification weights if provided
        if slr_weights is not None:
            loss_concepts = loss_concepts * slr_weights
            loss_mal = loss_mal * slr_weights
            
        loss_concepts = loss_concepts.mean()
        loss_mal = loss_mal.mean()
        
        # 4. Homoscedastic Uncertainty Weighting
        w_seg = torch.exp(-self.log_vars[0])
        w_concept = torch.exp(-self.log_vars[1])
        w_mal = torch.exp(-self.log_vars[2])
        
        total_loss = (
            w_seg * loss_seg + 0.5 * self.log_vars[0] +
            w_concept * loss_concepts + 0.5 * self.log_vars[1] +
            w_mal * loss_mal + 0.5 * self.log_vars[2]
        )
        
        loss_dict = {
            "total": total_loss,
            "seg": loss_seg,
            "comp": loss_comp.mean(),
            "echo": loss_echo.mean(),
            "shape": loss_shape.mean(),
            "margin": loss_margin.mean(),
            "foci": loss_foci.mean(),
            "malignancy": loss_mal
        }
        
        return total_loss, loss_dict
