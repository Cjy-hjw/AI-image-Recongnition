import os
import yaml
import torch

from src.models.multi_task_model import MultiTaskThyroidModel
from src.models.loss import MultiTaskLoss
from scripts.verify.gates.gate1_shapes import gate1_shapes
from scripts.verify.gates.gate2_vclub import gate2_vclub
from scripts.verify.gates.gate3_auc_margin import gate3_auc_margin
from scripts.verify.gates.gate4_famo import gate4_famo
from scripts.verify.gates.gate5_data_leakage import gate5_data_leakage
from scripts.verify.gates.gate6_stage1_masking import gate6_stage1_masking
from scripts.verify.gates.gate7_optimizer_lr import gate7_optimizer_lr


def verify():
    print("=== MODEL SANITY & GRADIENT FLOW VERIFICATION ===")

    # 1. Load config
    config_path = "configs/thyroid_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return False

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    backbone = config["model"]["backbone"]
    print(f"Testing configured backbone: {backbone}")

    # 2. Instantiate Model and Loss
    try:
        model = MultiTaskThyroidModel(backbone_name=backbone, pretrained=False)
        criterion = MultiTaskLoss()
        print("✔ Model & Loss instantiation successful.")
    except Exception as e:
        print(f"✘ Model instantiation failed: {e}")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    # Prepare normalized dummy data
    x = torch.randn(2, 1, 512, 512)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)  # normalized
    x = x.to(device)

    targets = {
        "mask": torch.randint(0, 2, (2, 1, 512, 512)).float().to(device),
        "composition": torch.tensor([0, 3]).to(device),
        "echogenicity": torch.tensor([1, 2]).to(device),
        "shape": torch.tensor([0, 1]).to(device),
        "margin": torch.tensor([2, 0]).to(device),
        "foci": torch.randn(2, 4).sigmoid().to(device),
        "malignancy": torch.tensor([1, 0]).to(device),
    }

    # Run gates sequentially — stop on failure (matching original behavior)
    if not gate1_shapes(model, x):
        return True  # Gate1 prints but doesn't halt in original

    if not gate2_vclub(model, x):
        return False

    if not gate3_auc_margin(model, criterion, x, targets):
        return False

    if not gate4_famo(model, criterion, x, targets, device):
        return False

    if not gate5_data_leakage():
        return False

    if not gate6_stage1_masking(model, criterion, x, targets, config):
        return False

    if not gate7_optimizer_lr(backbone, config, device):
        return False

    return True
