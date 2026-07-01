import torch
import torch.nn as nn

from src.models.famo import FAMOScheduler
from src.models.losses.segmentation import DiceLoss
from src.models.losses.classification import AUCMLossApprox, SupervisedContrastiveLoss
from src.models.losses.components import (
    compute_segmentation_loss,
    resolve_mix_params,
    compute_concept_losses,
    compute_malignancy_losses,
    apply_famo_weighting,
    compute_regularizers,
)


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss function.
    Balances segmentation BCE+Dice, TI-RADS concepts (via Symmetric Label Regression), and cost-sensitive malignancy classification
    using Homoscedastic Uncertainty Weighting and ACR TI-RADS prior regularization.
    """

    def __init__(self, num_tasks=5, alpha_prior=0.1, fn_weight=1.5):
        super().__init__()
        self.alpha_prior = alpha_prior  # Clinical prior regularization factor
        self.mal_loss_fn = AUCMLossApprox(margin=1.0, imratio=0.33)
        self.famo = FAMOScheduler(num_tasks=num_tasks, beta=0.05, gamma=0.001)
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
        enable_mal_loss=True,
        task_mask=None,
    ):
        """
        Args:
            task_mask: dict of binary masks (B,) indicating label availability
                {
                    'mask': Tensor[B],       # 1 if has segmentation mask, 0 otherwise
                    'composition': Tensor[B],
                    'echogenicity': Tensor[B],
                    'shape': Tensor[B],
                    'margin': Tensor[B],
                    'foci': Tensor[B],
                    'malignancy': Tensor[B],
                }
                If None, assumes all labels available (backward compat)

                Note: Segmentation already supports valid_mask via targets['valid_mask'],
                      so we pass task_mask['mask'] -> targets['valid_mask']
        """
        # 注入task_mask到targets（利用现有valid_mask支持）
        if task_mask is not None:
            if 'mask' in task_mask and 'valid_mask' not in targets:
                targets = {**targets, 'valid_mask': task_mask['mask'].float()}

        # 1. Segmentation Loss (自动应用valid_mask)
        loss_seg = compute_segmentation_loss(
            self.bce_seg, self.dice_seg, outputs, targets
        )

        # 2. Concept losses
        lam, index, mask_weight = resolve_mix_params(outputs, mix_info)
        concept_losses = compute_concept_losses(
            outputs,
            targets,
            lam,
            index,
            mask_weight,
            label_smoothing,
            self.tirads_matrix,
        )

        # Store for external access
        self.loss_concepts_ce = concept_losses["concepts_ce"].mean()
        self.loss_prior = concept_losses["prior"].mean()

        # 3. Malignancy losses
        loss_mal, loss_supcon = compute_malignancy_losses(
            self.mal_loss_fn,
            self.supcon_loss,
            outputs,
            targets,
            mask_weight,
            slr_weights,
        )

        if not enable_mal_loss:
            loss_mal = loss_mal * 0.0
            loss_supcon = loss_supcon * 0.0

        # 4. FAMO 5-channel dynamic loss weighting
        total_loss = apply_famo_weighting(
            self.famo,
            loss_seg,
            self.loss_concepts_ce,
            self.loss_prior,
            loss_mal,
            loss_supcon,
            update_famo_weights,
            self.training,
        )

        # 5. Regularization
        total_loss, loss_l2 = compute_regularizers(
            total_loss,
            outputs,
            self.lambda_latent,
            self.delta_budget,
            spin_l1_weight,
        )

        loss_dict = {
            "total": total_loss,
            "seg": loss_seg,
            "concepts_ce": self.loss_concepts_ce,
            "prior": self.loss_prior,
            "comp": concept_losses["comp"].mean(),
            "echo": concept_losses["echo"].mean(),
            "shape": concept_losses["shape"].mean(),
            "margin": concept_losses["margin"].mean(),
            "foci": concept_losses["foci"].mean(),
            "malignancy": loss_mal,
            "supcon": loss_supcon,
            "latent_l2": loss_l2,
            "spin_l1": outputs["spin_l1"],
        }

        return total_loss, loss_dict
