import os
import cv2
import json
import random
import numpy as np

def generate_ultrasound_noise(size, speckle_std=25):
    """Generates ultrasound-like speckle noise on a dark background."""
    # Base background (typical B-mode ultrasound is dark/black at boundaries)
    bg = np.zeros((size, size), dtype=np.uint8)
    
    # Draw a mock thyroid gland boundary (curved gray region in the center)
    cv2.ellipse(bg, (size//2, size//2), (size//2 - 20, size//3), 0, 0, 360, 40, -1)
    
    # Apply Gaussian blur to smooth the anatomical structures
    bg = cv2.GaussianBlur(bg, (21, 21), 0)
    
    # Add multiplicative speckle noise: I = S * N
    noise = np.random.normal(1.0, speckle_std / 255.0, (size, size))
    img = bg.astype(float) * noise
    
    # Add some additive Gaussian noise
    gaussian_noise = np.random.normal(0, 5, (size, size))
    img += gaussian_noise
    
    # Clip and convert to uint8
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def create_sample(path, index, patient_id, is_phantom=False, is_gan=False, is_tampered=False, is_duplicate=False, duplicate_source=None):
    size = 256
    
    # Generate background ultrasound image
    img = generate_ultrasound_noise(size)
    mask = np.zeros((size, size), dtype=np.uint8)
    
    # Decide if there is a nodule (malignant or benign)
    has_nodule = random.random() > 0.15  # 85% chance of nodule
    
    # Clinical labels
    comp = random.randint(0, 3)  # cystic, spongiform, mixed, solid
    echo = random.randint(0, 3)  # anechoic, hyperechoic/isoechoic, hypoechoic, very hypoechoic
    shape = random.randint(0, 1) # wider-than-tall, taller-than-wide
    margin = random.randint(0, 3) # smooth, ill-defined, lobulated/irregular, extrathyroidal extension
    foci = [random.randint(0, 1) for _ in range(4)] # none/comet-tail, large coarse, peripheral/rim, punctate
    malignancy = 1 if (comp >= 2 and echo >= 2 or sum(foci) >= 2) else 0
    
    foci_coords = []
    
    if has_nodule:
        # Nodule parameters
        center_x = random.randint(size//3, 2*size//3)
        center_y = random.randint(size//3, 2*size//3)
        
        # Shape: if shape label is 1 (taller-than-wide), height > width (in ultrasound, axis Y > axis X)
        if shape == 1:
            axes_x = random.randint(15, 30)
            axes_y = random.randint(35, 50)
        else:
            axes_x = random.randint(35, 50)
            axes_y = random.randint(15, 30)
            
        # Draw the nodule on mask
        cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 255, -1)
        
        # Draw the nodule on image: typically hypoechoic (darker) or hyperechoic (brighter)
        # Nodule interior intensity modifier
        nodule_color = 20 if echo == 3 else (40 if echo == 2 else 90)
        nodule_draw = np.zeros((size, size), dtype=np.uint8)
        cv2.ellipse(nodule_draw, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 255, -1)
        
        # Blending nodule with original image
        nodule_mask = nodule_draw > 0
        img[nodule_mask] = np.clip(img[nodule_mask].astype(float) * (nodule_color / 60.0), 0, 255).astype(np.uint8)
        
        # Add micro-calcifications (punctate echogenic foci)
        if foci[3] == 1:
            for _ in range(random.randint(2, 5)):
                # Pick random points inside the nodule ellipse
                # Simple approximation: near the center
                offset_x = random.randint(-axes_x+5, axes_x-5)
                offset_y = random.randint(-axes_y+5, axes_y-5)
                px, py = center_x + offset_x, center_y + offset_y
                if px > 0 and px < size and py > 0 and py < size:
                    cv2.circle(img, (px, py), 1, 255, -1)
                    foci_coords.append([px, py])
                    
    # Handle duplicate case
    if is_duplicate and duplicate_source is not None:
        # Make a copy of the source image with minor pixel variations or identical
        img = cv2.imread(duplicate_source, cv2.IMREAD_GRAYSCALE)
        # Maybe copy mask as well
        source_mask_path = duplicate_source.replace("image.png", "mask.png")
        mask = cv2.imread(source_mask_path, cv2.IMREAD_GRAYSCALE)
        # Inherit metadata
        source_meta_path = duplicate_source.replace("image.png", "meta.json")
        with open(source_meta_path, 'r') as f:
            meta = json.load(f)
        meta['is_duplicate'] = 1
        meta['patient_id'] = patient_id # maybe different patient (identity theft duplicate)
    else:
        # Setup metadata
        authenticity = 1 if is_phantom else (2 if is_gan else 0)
        
        # Apply physical phantom properties (over-smooth boundaries, regular noise)
        if is_phantom:
            # Over-smooth the image to simulate phantom
            img = cv2.GaussianBlur(img, (7, 7), 0)
            
        # Apply GAN properties (high frequency checkerboard artifacts)
        if is_gan:
            # Create a small high frequency grid artifact
            for x in range(0, size, 4):
                img[:, x] = np.clip(img[:, x].astype(int) + 5, 0, 255).astype(np.uint8)
                
        # Handle tampering (splicing or copy-move)
        tamper_bbox = None
        if is_tampered:
            # Copy a chunk of image and paste it elsewhere
            w, h = 40, 40
            # Source patch
            sx, sy = random.randint(20, size-60), random.randint(20, size-60)
            patch = img[sy:sy+h, sx:sx+w].copy()
            # Destination patch
            dx, dy = random.randint(20, size-60), random.randint(20, size-60)
            img[dy:dy+h, dx:dx+w] = patch
            # Flag bbox
            tamper_bbox = [dx, dy, dx+w, dy+h]
            # Also edit mask if relevant
            patch_mask = mask[sy:sy+h, sx:sx+w].copy()
            mask[dy:dy+h, dx:dx+w] = patch_mask

        meta = {
            "image_id": f"sample_{index}",
            "patient_id": patient_id,
            "label_composition": comp if has_nodule else 0,
            "label_echogenicity": echo if has_nodule else 1,
            "label_shape": shape if has_nodule else 0,
            "label_margin": margin if has_nodule else 0,
            "label_foci": foci if has_nodule else [0, 0, 0, 0],
            "label_malignancy": malignancy if has_nodule else 0,
            "label_authenticity": authenticity,
            "is_duplicate": 0,
            "is_tampered": 1 if is_tampered else 0,
            "tamper_bbox": tamper_bbox,
            "foci_coords": foci_coords
        }
        
    # Write files
    os.makedirs(os.path.join(path, f"sample_{index}"), exist_ok=True)
    img_filepath = os.path.join(path, f"sample_{index}", "image.png")
    mask_filepath = os.path.join(path, f"sample_{index}", "mask.png")
    meta_filepath = os.path.join(path, f"sample_{index}", "meta.json")
    
    cv2.imwrite(img_filepath, img)
    cv2.imwrite(mask_filepath, mask)
    with open(meta_filepath, 'w') as f:
        json.dump(meta, f, indent=2)
        
    return img_filepath

def main():
    print("Generating dummy ultrasound dataset...")
    random.seed(42)
    np.random.seed(42)
    
    splits = {
        "data/train": 40,
        "data/val": 10,
        "data/holdout": 10
    }
    
    total_idx = 0
    generated_images = []
    
    for split_path, count in splits.items():
        os.makedirs(split_path, exist_ok=True)
        print(f"Creating {count} samples in {split_path}...")
        
        for i in range(count):
            # 5 patients in train, 2 in val, 2 in holdout to test patient grouping
            if "train" in split_path:
                patient_id = f"PAT_{random.randint(1, 8)}"
            elif "val" in split_path:
                patient_id = f"PAT_{random.randint(9, 12)}"
            else:
                patient_id = f"PAT_{random.randint(13, 16)}"
                
            # Gatekeeper distributions:
            # 5% phantoms, 5% GAN, 10% tampered, 10% duplicates
            is_phantom = random.random() < 0.05
            is_gan = random.random() < 0.05 and not is_phantom
            is_tampered = random.random() < 0.10 and not (is_phantom or is_gan)
            is_duplicate = random.random() < 0.10 and len(generated_images) > 0 and not (is_phantom or is_gan or is_tampered)
            
            dup_source = None
            if is_duplicate:
                dup_source = random.choice(generated_images)
                
            img_path = create_sample(
                split_path, 
                total_idx, 
                patient_id, 
                is_phantom=is_phantom, 
                is_gan=is_gan, 
                is_tampered=is_tampered, 
                is_duplicate=is_duplicate, 
                duplicate_source=dup_source
            )
            generated_images.append(img_path)
            total_idx += 1
            
    print(f"Dataset generated successfully! Total samples: {total_idx}")

if __name__ == "__main__":
    main()
