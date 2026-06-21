"""
Backward-compatible shim. All loss classes and functions have been moved to
src/models/losses/ subpackage. This file re-exports everything so that
existing imports like `from src.models.loss import MultiTaskLoss` continue
to work without modification.
"""

from src.models.losses.segmentation import DiceLoss
from src.models.losses.classification import (
    AUCMLossApprox,
    AdaptiveVSAndAsymmetricLoss,
    SupervisedContrastiveLoss,
)
from src.models.losses.concepts import (
    symmetric_label_regression_loss,
    symmetric_label_regression_loss_bce,
)
from src.models.losses.multitask import MultiTaskLoss

__all__ = [
    "DiceLoss",
    "AUCMLossApprox",
    "AdaptiveVSAndAsymmetricLoss",
    "SupervisedContrastiveLoss",
    "symmetric_label_regression_loss",
    "symmetric_label_regression_loss_bce",
    "MultiTaskLoss",
]
