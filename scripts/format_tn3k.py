import os
import csv
import json
import shutil
import cv2
import numpy as np

def extract_features_from_mask_and_image(img_path, mask_path):
    """Extract image-based TI-RADS pseudo-labels from ultrasound image and segmentation mask.

    Returns (shape_label, foci_label, foci_coords, echo_label, comp_label, margin_label).
    All labels are derived from physical image features — NOT correlated with malignancy.
    """
    img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    shape_label  = 0
    foci_label   = [0, 0, 0, 0]
    foci_coords  = []
    echo_label   = 1
    comp_label   = 3
    margin_label = 0

    if img is None or mask is None:
        return shape_label, foci_label, foci_coords, echo_label, comp_label, margin_label

    bin_mask = (mask > 127).astype(np.uint8)
    if bin_mask.sum() < 50:
        return shape_label, foci_label, foci_coords, echo_label, comp_label, margin_label

    ys, xs = np.where(bin_mask)

    # 1. SHAPE: bounding-box aspect ratio
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    shape_label = 1 if h > w else 0  # 1=taller-than-wide, 0=wider-than-tall

    # 2. FOCI: white top-hat on nodule ROI
    roi    = img * bin_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
    tophat = cv2.subtract(roi, opened)
    bright_px = (tophat > 40) & (bin_mask == 1)
    if bright_px.sum() > 10:
        foci_label[3] = 1  # punctate echogenic foci (micro-calcification)
        for y, x in zip(*np.where(bright_px)):
            foci_coords.append([int(x), int(y)])

    # 3. ECHOGENICITY: nodule vs. surrounding parenchyma
    dilated  = cv2.dilate(bin_mask, np.ones((31, 31), np.uint8), iterations=1)
    surround = (dilated - bin_mask).astype(bool)
    mu_nod   = float(img[bin_mask == 1].mean())
    sur_vals = img[surround]
    if sur_vals.size == 0:
        echo_label = 2
    else:
        mu_par = float(sur_vals.mean())
        mu_mus = float(np.percentile(sur_vals, 20))  # dark band ≈ muscle
        if mu_nod < 15:
            echo_label = 0   # anechoic
        elif mu_nod >= mu_par:
            echo_label = 1   # hyperechoic / isoechoic
        elif mu_nod <= mu_mus:
            echo_label = 3   # very hypoechoic
        else:
            echo_label = 2   # hypoechoic

    # 4. COMPOSITION: cystic pixel fraction
    nod_pixels  = img[bin_mask == 1].astype(float)
    cystic_ratio = float((nod_pixels < 40).sum()) / nod_pixels.size
    if cystic_ratio > 0.90:
        comp_label = 0   # cystic
    elif cystic_ratio > 0.50:
        comp_label = 2   # mixed
    elif float(nod_pixels.std()) > 35:
        comp_label = 1   # spongiform (heterogeneous solid)
    else:
        comp_label = 3   # solid

    # 5. MARGIN: convexity defect ratio + border touch
    H, W = img.shape
    border_touch = (ys.min() < 5 or ys.max() > H - 5 or xs.min() < 5 or xs.max() > W - 5)
    if border_touch:
        margin_label = 3   # extrathyroidal extension
    else:
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cnt       = max(contours, key=cv2.contourArea)
            hull      = cv2.convexHull(cnt)
            cnt_area  = cv2.contourArea(cnt)
            hull_area = cv2.contourArea(hull)
            if cnt_area > 0:
                ratio = hull_area / cnt_area
                if ratio < 1.05:
                    margin_label = 0   # smooth
                elif ratio < 1.20:
                    margin_label = 1   # ill-defined
                else:
                    margin_label = 2   # lobulated / spiculated

    return shape_label, foci_label, foci_coords, echo_label, comp_label, margin_label

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
                
    # 基于 Patient ID 互斥隔离分组划分 (Grouped Train-Val Split)
    np.random.seed(42)
    
    if is_trainval:
        # 1. 归纳每个样本对应的 Patient ID
        patient_to_samples = {}
        for img_name, malignancy in samples:
            img_id_num = int(img_name.split('.')[0])
            patient_id = f"PAT_{img_id_num // 2:03d}"
            if patient_id not in patient_to_samples:
                patient_to_samples[patient_id] = []
            patient_to_samples[patient_id].append((img_name, malignancy))
            
        # 2. 对患者 ID 集合进行随机 Shuffle
        unique_patients = list(patient_to_samples.keys())
        np.random.shuffle(unique_patients)
        
        # 3. 按比例划分患者
        split_idx = int(len(unique_patients) * split_ratio)
        train_patients = unique_patients[:split_idx]
        val_patients = unique_patients[split_idx:]
        
        # 4. 根据患者划分组装样本
        train_samples = []
        for p in train_patients:
            train_samples.extend(patient_to_samples[p])
            
        val_samples = []
        for p in val_patients:
            val_samples.extend(patient_to_samples[p])
            
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
            shape, foci, foci_coords, echo, comp, margin = extract_features_from_mask_and_image(img_path, mask_path)
                
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
