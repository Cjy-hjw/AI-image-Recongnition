import os
import csv
import json
import shutil
import cv2
import numpy as np

from scripts.data.pseudo_labels import extract_features_from_mask_and_image


def _split_samples(samples, split_ratio=0.8, seed=42):
    """纯随机图像级划分(无患者分组)。

    TN3K 原始数据无患者级元数据 —— 不再臆造 PAT_{//2} 伪分组。
    每张图像即为独立身份，train/val 按 split_ratio 随机互斥切分，固定 seed 可复现。

    Returns:
        (train_samples, val_samples): list of (img_name, malignancy) tuples.
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(samples))
    split_idx = int(len(samples) * split_ratio)
    train = [samples[i] for i in indices[:split_idx]]
    val = [samples[i] for i in indices[split_idx:]]
    return train, val


def format_split(csv_path, img_dir, mask_dir, dest_dir, split_ratio=0.8, is_trainval=True):
    if not os.path.exists(csv_path):
        print(f"CSV path not found: {csv_path}")
        return
        
    os.makedirs(dest_dir, exist_ok=True)
    
    # Read CSV
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                samples.append((row[0], int(row[1])))
                
    # 纯随机图像级划分(TN3K 无患者元数据,每张图即独立身份)。
    np.random.seed(42)

    if is_trainval:
        train_samples, val_samples = _split_samples(samples, split_ratio, seed=42)
        splits = {
            "data/train": train_samples,
            "data/val": val_samples,
        }
    else:
        splits = {
            dest_dir: samples
        }
    total_copied = 0
    
    for split_dest, split_data in splits.items():
        print(f"Formatting {len(split_data)} samples to {split_dest}...")
        os.makedirs(split_dest, exist_ok=True)
        
        for idx, (img_name, malignancy) in enumerate(split_data):
            img_path = os.path.join(img_dir, img_name)
            
            # Mask filename can be either .jpg or .png (TN3K uses .jpg)
            mask_name = img_name
            mask_path = os.path.join(mask_dir, mask_name)
            if not os.path.exists(mask_path):
                mask_name = img_name.replace(".jpg", ".png")
                mask_path = os.path.join(mask_dir, mask_name)
            
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                continue
                
            # Create subfolder for sample
            sample_folder_name = f"sample_{img_name.split('.')[0]}"
            sample_dest_dir = os.path.join(split_dest, sample_folder_name)
            os.makedirs(sample_dest_dir, exist_ok=True)
            
            # Extract features from mask
            shape, foci, foci_coords, echo, comp, margin = extract_features_from_mask_and_image(img_path, mask_path)
                
            # 每张图像为独立身份(TN3K 无患者元数据)。
            image_id = img_name.split('.')[0]
            patient_id = image_id
            
            # Construct standard metadata JSON
            meta = {
                "image_id": img_name.split('.')[0],
                "patient_id": patient_id,
                "label_composition": int(comp),
                "label_echogenicity": int(echo),
                "label_shape": int(shape),
                "label_margin": int(margin),
                "label_foci": foci,
                "label_malignancy": int(malignancy),
                "label_authenticity": 0,  # Real human ultrasound
                "is_duplicate": 0,
                "is_tampered": 0,
                "tamper_bbox": None,
                "foci_coords": foci_coords
            }
            
            # Copy files
            shutil.copy(img_path, os.path.join(sample_dest_dir, "image.png"))
            shutil.copy(mask_path, os.path.join(sample_dest_dir, "mask.png"))
            
            with open(os.path.join(sample_dest_dir, "meta.json"), "w") as f_json:
                json.dump(meta, f_json, indent=2)
                
            total_copied += 1
            
    print(f"Formatting complete. Reorganized {total_copied} samples successfully.")

