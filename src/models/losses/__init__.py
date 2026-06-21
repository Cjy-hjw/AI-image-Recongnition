"""
Loss function module decomposition.
Exports all loss classes and functions for backward compatibility.
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
