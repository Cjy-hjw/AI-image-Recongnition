"""MultiTaskLoss computation components — re-exports."""

from src.models.losses.components.segmentation import compute_segmentation_loss
from src.models.losses.components.concepts import (
    resolve_mix_params,
    compute_concept_losses,
)
from src.models.losses.components.malignancy import compute_malignancy_losses
from src.models.losses.components.famo_mix import apply_famo_weighting
from src.models.losses.components.regularizers import compute_regularizers
# Phase2 三头（阶段 2.1/2.2/2.3）。尚未接入 `MultiTaskLoss.forward`——接线在阶段 2.4，
# 这里先补齐 re-export，使三者与既有 5 个组件的导入路径一致。
from src.models.losses.components.location import compute_location_loss
from src.models.losses.components.extension import (
    class_balanced_weights,
    compute_extension_loss,
    log_class_prior,
)
from src.models.losses.components.clinical_t import (
    compute_clinical_t_loss,
    inverse_frequency_weights,
)

__all__ = [
    "compute_segmentation_loss",
    "resolve_mix_params",
    "compute_concept_losses",
    "compute_malignancy_losses",
    "apply_famo_weighting",
    "compute_regularizers",
    "compute_location_loss",
    "compute_extension_loss",
    "class_balanced_weights",
    "log_class_prior",
    "compute_clinical_t_loss",
    "inverse_frequency_weights",
]
