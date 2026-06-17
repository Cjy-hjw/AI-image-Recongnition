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
    Balances segmentation BCE+Dice, TI-RADS concepts, and cost-sensitive malignancy classification.
    """
    def __init__(self, fn_weight=2.5, w_seg=1.0, w_concepts=0.5, w_mal=1.5):
        super().__init__()
        self.fn_weight = fn_weight # Penalty factor for False Negatives in malignancy
        self.w_seg = w_seg
        self.w_concepts = w_concepts
        self.w_mal = w_mal
        
        self.bce_seg = nn.BCEWithLogitsLoss()
        self.dice_seg = DiceLoss()
        
        self.ce_comp = nn.CrossEntropyLoss()
        self.ce_echo = nn.CrossEntropyLoss()
        self.ce_shape = nn.CrossEntropyLoss()
        self.ce_margin = nn.CrossEntropyLoss()
        self.bce_foci = nn.BCEWithLogitsLoss()
        
    def forward(self, outputs, targets, slr_weights=None, mix_info=None):
        """
        Args:
            outputs: Dictionary of model outputs
            targets: Dictionary of ground truth labels
            slr_weights: Optional sample-wise weights for Supervised Label Rectification
            mix_info: Optional dictionary for MixUp/CutMix mixing info
        """
        # 1. Segmentation Loss
        loss_bce_seg = self.bce_seg(outputs["mask"], targets["mask"])
        loss_dice_seg = self.dice_seg(outputs["mask"], targets["mask"])
        loss_seg = loss_bce_seg + loss_dice_seg
        
        # 2. TI-RADS Concept Losses
        if mix_info is not None:
            index = mix_info["index"]
            lam = mix_info["lam"]
            
            loss_comp = lam * self.ce_comp(outputs["composition"], targets["composition"]) + (1.0 - lam) * self.ce_comp(outputs["composition"], targets["composition"][index])
            loss_echo = lam * self.ce_echo(outputs["echogenicity"], targets["echogenicity"]) + (1.0 - lam) * self.ce_echo(outputs["echogenicity"], targets["echogenicity"][index])
            loss_shape = lam * self.ce_shape(outputs["shape"], targets["shape"]) + (1.0 - lam) * self.ce_shape(outputs["shape"], targets["shape"][index])
            loss_margin = lam * self.ce_margin(outputs["margin"], targets["margin"]) + (1.0 - lam) * self.ce_margin(outputs["margin"], targets["margin"][index])
            loss_foci = lam * self.bce_foci(outputs["foci"], targets["foci"]) + (1.0 - lam) * self.bce_foci(outputs["foci"], targets["foci"][index])
        else:
            loss_comp = self.ce_comp(outputs["composition"], targets["composition"])
            loss_echo = self.ce_echo(outputs["echogenicity"], targets["echogenicity"])
            loss_shape = self.ce_shape(outputs["shape"], targets["shape"])
            loss_margin = self.ce_margin(outputs["margin"], targets["margin"])
            loss_foci = self.bce_foci(outputs["foci"], targets["foci"])
            
        loss_concepts = loss_comp + loss_echo + loss_shape + loss_margin + loss_foci
        
        # 3. Cost-Sensitive Malignancy Loss
        mal_logits = outputs["malignancy"]
        mal_targets = targets["malignancy"]
        
        log_prob = F.log_softmax(mal_logits, dim=1)
        
        if mix_info is not None:
            index = mix_info["index"]
            lam = mix_info["lam"]
            
            # original targets loss
            loss_mal_raw_orig = F.nll_loss(log_prob, mal_targets, reduction='none')
            preds_orig = torch.argmax(mal_logits, dim=1)
            fn_mask_orig = (mal_targets == 1) & (preds_orig == 0)
            penalty_weights_orig = torch.ones_like(loss_mal_raw_orig, dtype=torch.float32)
            penalty_weights_orig[fn_mask_orig] = self.fn_weight
            loss_mal_orig = loss_mal_raw_orig * penalty_weights_orig
            
            # permuted targets loss
            mal_targets_perm = mal_targets[index]
            loss_mal_raw_perm = F.nll_loss(log_prob, mal_targets_perm, reduction='none')
            fn_mask_perm = (mal_targets_perm == 1) & (preds_orig == 0)
            penalty_weights_perm = torch.ones_like(loss_mal_raw_perm, dtype=torch.float32)
            penalty_weights_perm[fn_mask_perm] = self.fn_weight
            loss_mal_perm = loss_mal_raw_perm * penalty_weights_perm
            
            # mix the losses
            loss_mal = lam * loss_mal_orig + (1.0 - lam) * loss_mal_perm
        else:
            loss_mal_raw = F.nll_loss(log_prob, mal_targets, reduction='none')
            preds = torch.argmax(mal_logits, dim=1)
            fn_mask = (mal_targets == 1) & (preds == 0)
            penalty_weights = torch.ones_like(loss_mal_raw, dtype=torch.float32)
            penalty_weights[fn_mask] = self.fn_weight
            loss_mal = loss_mal_raw * penalty_weights
            
        # Apply Supervised Label Rectification weights if provided
        if slr_weights is not None:
            loss_concepts = loss_concepts * slr_weights.mean()
            loss_mal = loss_mal * slr_weights
            
        loss_mal = loss_mal.mean()
        
        # Total combined loss
        total_loss = (self.w_seg * loss_seg) + (self.w_concepts * loss_concepts) + (self.w_mal * loss_mal)
        
        loss_dict = {
            "total": total_loss,
            "seg": loss_seg,
            "comp": loss_comp,
            "echo": loss_echo,
            "shape": loss_shape,
            "margin": loss_margin,
            "foci": loss_foci,
            "malignancy": loss_mal
        }
        
        return total_loss, loss_dict
