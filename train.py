import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

from src.dataset.loader import ThyroidDataset
from src.dataset.augmentation import get_train_transforms, apply_mixup_or_cutmix
from src.models.multi_task_model import MultiTaskThyroidModel
from src.models.loss import MultiTaskLoss

def train_epoch(model, loader, optimizer, criterion, scaler, device, config, running_losses_history):
    model.train()
    running_loss = 0.0
    seg_loss = 0.0
    mal_loss = 0.0
    
    # SLR parameters
    slr_beta = config["train"].get("slr_beta", 0.99)
    slr_threshold = config["train"].get("slr_threshold", 2.0)
    
    for batch_idx, (images, masks, labels) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Prepare targets dictionary
        targets = {
            "mask": masks,
            "composition": labels["composition"].to(device),
            "echogenicity": labels["echogenicity"].to(device),
            "shape": labels["shape"].to(device),
            "margin": labels["margin"].to(device),
            "foci": labels["foci"].to(device),
            "malignancy": labels["malignancy"].to(device)
        }
        
        # Apply MixUp / CutMix data augmentation at batch level
        images, masks, targets, mix_info = apply_mixup_or_cutmix(images, masks, targets, alpha=0.2, prob=0.3)
        targets["mask"] = masks
        
        optimizer.zero_grad()
        
        # Mixed Precision Forward pass
        with torch.cuda.amp.autocast(enabled=config["train"].get("amp", True)):
            outputs = model(images)
            
            # SLR computation
            # For SLR, we compute loss per-sample to identify noise without gradients
            if slr_beta < 1.0:
                with torch.no_grad():
                    mal_logits = outputs["malignancy"]
                    mal_targets = targets["malignancy"]
                    log_prob = F.log_softmax(mal_logits, dim=1)
                    
                    if mix_info is not None:
                        index = mix_info["index"]
                        lam = mix_info["lam"]
                        loss_mal_raw_orig = F.nll_loss(log_prob, mal_targets, reduction='none')
                        loss_mal_raw_perm = F.nll_loss(log_prob, mal_targets[index], reduction='none')
                        sample_losses = lam * loss_mal_raw_orig + (1.0 - lam) * loss_mal_raw_perm
                    else:
                        sample_losses = F.nll_loss(log_prob, mal_targets, reduction='none')
                        
                for sl in sample_losses.cpu().numpy():
                    running_losses_history.append(sl)
                    
                if len(running_losses_history) > 1000:
                    running_losses_history[:] = running_losses_history[-1000:]
                    
                mean_loss = np.mean(running_losses_history)
                std_loss = np.std(running_losses_history) + 1e-6
                
                # Check for anomalies (potential label noise) and set weights
                slr_weights = torch.ones_like(sample_losses)
                anomaly_mask = sample_losses > (mean_loss + slr_threshold * std_loss)
                slr_weights[anomaly_mask] = 1.0 - slr_beta
                
                loss, loss_dict = criterion(outputs, targets, slr_weights=slr_weights, mix_info=mix_info)
            else:
                loss, loss_dict = criterion(outputs, targets, mix_info=mix_info)
                
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        seg_loss += loss_dict["seg"].item()
        mal_loss += loss_dict["malignancy"].item()
        
    avg_loss = running_loss / len(loader)
    return avg_loss, seg_loss / len(loader), mal_loss / len(loader)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    seg_dice = 0.0
    mal_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, masks, labels in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            targets = {
                "mask": masks,
                "composition": labels["composition"].to(device),
                "echogenicity": labels["echogenicity"].to(device),
                "shape": labels["shape"].to(device),
                "margin": labels["margin"].to(device),
                "foci": labels["foci"].to(device),
                "malignancy": labels["malignancy"].to(device)
            }
            
            outputs = model(images)
            loss, loss_dict = criterion(outputs, targets)
            val_loss += loss.item()
            
            # Calculate Dice for validation
            pred_mask = torch.sigmoid(outputs["mask"]) > 0.5
            intersection = (pred_mask * masks).sum()
            union = pred_mask.sum() + masks.sum()
            dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
            seg_dice += dice.item()
            
            # Calculate malignancy accuracy
            pred_mal = torch.argmax(outputs["malignancy"], dim=1)
            mal_correct += (pred_mal == targets["malignancy"]).sum().item()
            total_samples += images.size(0)
            
    avg_loss = val_loss / len(loader)
    avg_dice = seg_dice / len(loader)
    avg_acc = mal_correct / total_samples
    return avg_loss, avg_dice, avg_acc

def train_authenticity_classifier(train_dataset, val_dataset, device, save_path):
    from src.gatekeeper.authenticity import AuthenticityClassifier, extract_authenticity_features
    import torch.nn as nn
    
    print("Training Gatekeeper Authenticity Classifier MLP...")
    auth_model = AuthenticityClassifier().to(device)
    optimizer = optim.Adam(auth_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Extract features for training
    train_feats = []
    train_labels = []
    for idx in range(len(train_dataset)):
        img_tensor, _, labels = train_dataset[idx]
        img_np_255 = (img_tensor.squeeze(0).numpy() * 255.0).astype(np.uint8)
        feat = extract_authenticity_features(img_np_255)
        train_feats.append(feat)
        train_labels.append(labels["authenticity"])
        
    train_feats = torch.tensor(np.array(train_feats), dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
    
    # Extract features for validation
    val_feats = []
    val_labels = []
    for idx in range(len(val_dataset)):
        img_tensor, _, labels = val_dataset[idx]
        img_np_255 = (img_tensor.squeeze(0).numpy() * 255.0).astype(np.uint8)
        feat = extract_authenticity_features(img_np_255)
        val_feats.append(feat)
        val_labels.append(labels["authenticity"])
        
    val_feats = torch.tensor(np.array(val_feats), dtype=torch.float32).to(device)
    val_labels = torch.tensor(val_labels, dtype=torch.long).to(device)
    
    # Train
    auth_model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = auth_model(train_feats)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        
    # Validate
    auth_model.eval()
    with torch.no_grad():
        val_outputs = auth_model(val_feats)
        val_preds = torch.argmax(val_outputs, dim=1)
        val_acc = (val_preds == val_labels).float().mean().item()
        print(f"--> Authenticity MLP Val Accuracy: {val_acc*100:.2f}%")
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(auth_model.state_dict(), save_path)
    print(f"--> Saved Authenticity Classifier weights to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/thyroid_config.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config["environment"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Load dataset loaders
    train_dataset = ThyroidDataset(
        data_dir=config["data"]["train_path"],
        img_size=config["data"]["img_size"],
        transform=get_train_transforms()
    )
    val_dataset = ThyroidDataset(
        data_dir=config["data"]["val_path"],
        img_size=config["data"]["img_size"]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["environment"]["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["environment"]["num_workers"]
    )
    
    print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    
    # Train gatekeeper authenticity MLP before main training
    auth_weights_path = "checkpoints/authenticity_model.pth"
    train_authenticity_classifier(train_dataset, val_dataset, device, auth_weights_path)
    
    # Initialize Multi-task model
    model = MultiTaskThyroidModel(
        backbone_name=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"]
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"]
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["train"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=config["train"]["amp"])
    criterion = MultiTaskLoss()
    
    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)
    
    print("Starting training loop...")
    running_losses_history = []
    for epoch in range(1, config["train"]["epochs"] + 1):
        train_loss, train_seg, train_mal = train_epoch(model, train_loader, optimizer, criterion, scaler, device, config, running_losses_history)
        val_loss, val_dice, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch:02d}/{config['train']['epochs']:02d} | "
              f"Train Loss: {train_loss:.4f} (Seg: {train_seg:.3f}, Mal: {train_mal:.3f}) | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.3f} | Val Mal Acc: {val_acc:.3f}")
              
        # Checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("--> Saved new best checkpoint to checkpoints/best_model.pth")
            
    print("Training finished successfully.")

if __name__ == "__main__":
    main()
