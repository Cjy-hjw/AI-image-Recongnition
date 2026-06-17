import os
import csv
import json
import shutil
import cv2
import numpy as np

def extract_features_from_mask_and_image(img_path, mask_path):
    """
    Extracts pseudo-ground-truth features from image and mask.
    - Shape (taller-than-wide): Computed from mask bounding box height/width ratio.
    - Foci (micro-calcifications): Detected as bright intensity spikes inside the mask.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or mask is None:
        return 0, [0, 0, 0, 0], []
        
    shape_label = 0
    foci_label = [0, 0, 0, 0]
    foci_coords = []
    
    # 1. Bounding box and shape analysis
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get largest contour
        areas = [cv2.contourArea(c) for c in contours]
        target_contour = contours[np.argmax(areas)]
        x, y, w, h = cv2.boundingRect(target_contour)
        
        # Taller-than-wide (shape = 1) if height > width
        if h > w:
            shape_label = 1
            
        # 2. Foci detection: search for high-intensity peaks (calcifications) inside mask
        nodule_region = img[mask > 127]
        if len(nodule_region) > 0:
            # Threshold for micro-calcifications (sharp white pixels)
            peak_val = np.percentile(nodule_region, 99.5)
            if peak_val > 220:
                foci_label[3] = 1 # Punctate echogenic foci
                
                # Locate coordinates
                ys, xs = np.where((img > 220) & (mask > 127))
                for py, px in zip(ys[:5], xs[:5]): # limit to 5 coordinates
                    foci_coords.append([int(px), int(py)])
                    
    return shape_label, foci_label, foci_coords

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
                
    # Sort or shuffle (seed-based) to maintain consistency
    np.random.seed(42)
    np.random.shuffle(samples)
    
    if is_trainval:
        split_idx = int(len(samples) * split_ratio)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        splits = {
            "data/train": train_samples,
            "data/val": val_samples
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
            shape, foci, foci_coords = extract_features_from_mask_and_image(img_path, mask_path)
            
            # Formulate composition, echogenicity, margin pseudo-labels correlated with malignancy
            if malignancy == 1:
                comp = 3  # Solid (malignant nodules are mostly solid)
                echo = 2  # Hypoechoic
                margin = np.random.choice([2, 3])  # irregular or extrathyroidal extension
            else:
                comp = np.random.choice([0, 1, 2]) # mixed/spongiform
                echo = np.random.choice([0, 1])    # anechoic/hyperechoic
                margin = np.random.choice([0, 1])  # smooth/ill-defined
                
            # Assign simulated patient ID based on image number prefix (e.g. 0130 -> PAT_01)
            img_id_num = int(img_name.split('.')[0])
            patient_id = f"PAT_{img_id_num // 2:03d}"
            
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

def main():
    # 1. Format trainval split
    format_split(
        csv_path="data/TN3K/label4trainval.csv",
        img_dir="data/TN3K/trainval-image",
        mask_dir="data/TN3K/trainval-mask",
        dest_dir="data/train",
        split_ratio=0.85,  # 85% train, 15% val
        is_trainval=True
    )
    
    # 2. Format test split to holdout
    format_split(
        csv_path="data/TN3K/label4test.csv",
        img_dir="data/TN3K/test-image",
        mask_dir="data/TN3K/test-mask",
        dest_dir="data/holdout",
        is_trainval=False
    )

if __name__ == "__main__":
    main()
