import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class ThyroidDataset(Dataset):
    """
    Thyroid Ultrasound Dataset for Multi-task Segmentation, Detection, and classification.
    Loads images, masks, and clinical labels from structured directories.
    """
    def __init__(self, data_dir, img_size=256, transform=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform
        
        self.samples = []
        if os.path.exists(data_dir):
            for sample_dir in sorted(os.listdir(data_dir)):
                sample_path = os.path.join(data_dir, sample_dir)
                if os.path.isdir(sample_path):
                    img_path = os.path.join(sample_path, "image.png")
                    mask_path = os.path.join(sample_path, "mask.png")
                    meta_path = os.path.join(sample_path, "meta.json")
                    
                    if os.path.exists(img_path) and os.path.exists(meta_path):
                        self.samples.append({
                            "img_path": img_path,
                            "mask_path": mask_path if os.path.exists(mask_path) else None,
                            "meta_path": meta_path
                        })
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load grayscale image
        img = cv2.imread(sample["img_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at {sample['img_path']}")
            
        # Resize
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))
            
        # Load mask
        if sample["mask_path"] is not None:
            mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                if mask.shape[0] != self.img_size or mask.shape[1] != self.img_size:
                    mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            
        # Normalize mask to binary
        mask = (mask > 127).astype(np.float32)
        
        # Load metadata
        with open(sample["meta_path"], "r") as f:
            meta = json.load(f)
            
        # Extract labels
        labels = {
            "composition": int(meta.get("label_composition", 0)),
            "echogenicity": int(meta.get("label_echogenicity", 0)),
            "shape": int(meta.get("label_shape", 0)),
            "margin": int(meta.get("label_margin", 0)),
            "foci": torch.tensor(meta.get("label_foci", [0, 0, 0, 0]), dtype=torch.float32),
            "malignancy": int(meta.get("label_malignancy", 0)),
            "authenticity": int(meta.get("label_authenticity", 0)),
            "is_duplicate": int(meta.get("is_duplicate", 0)),
            "is_tampered": int(meta.get("is_tampered", 0)),
            "patient_id": meta.get("patient_id", "UNKNOWN")
        }
        
        # Apply data augmentations if specified
        if self.transform is not None:
            img, mask = self.transform(img, mask)
            
        # Normalize image to [0, 1] float32 tensor
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return img_tensor, mask_tensor, labels

def get_patient_groups(dataset):
    """Utility to return patient IDs for patient-grouped split validation."""
    return [s["meta_path"] for s in dataset.samples] # can read patient_id directly
