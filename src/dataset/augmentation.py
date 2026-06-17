import random
import numpy as np
import cv2
import torch

class RandomSpeckleNoise:
    """Multiplicative speckle noise injection to simulate ultrasound gain."""
    def __init__(self, prob=0.5, std_range=(0.05, 0.20)):
        self.prob = prob
        self.std_range = std_range
        
    def __call__(self, img, mask):
        if random.random() < self.prob:
            std = random.uniform(*self.std_range)
            # Speckle noise: I_new = I * (1 + noise)
            noise = np.random.normal(0, std, img.shape)
            img_aug = img.astype(float) * (1.0 + noise)
            img = np.clip(img_aug, 0, 255).astype(np.uint8)
        return img, mask

class RandomElasticDeformation:
    """
    Elastic deformation to simulate probe pressure variation.
    Uses cv2.remap for rapid, self-contained deformation on 2D images.
    """
    def __init__(self, prob=0.5, alpha=200, sigma=20):
        self.prob = prob
        self.alpha = alpha
        self.sigma = sigma
        
    def __call__(self, img, mask):
        if random.random() < self.prob:
            shape = img.shape
            
            # Generate random displacement fields
            dx = np.random.rand(*shape) * 2 - 1
            dy = np.random.rand(*shape) * 2 - 1
            
            # Smooth displacement fields using Gaussian filter
            dx = cv2.GaussianBlur(dx, (0, 0), self.sigma) * self.alpha
            dy = cv2.GaussianBlur(dy, (0, 0), self.sigma) * self.alpha
            
            # Create map coordinates
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)
            
            # Remap image and mask
            img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            
        return img, mask

class RandomAffine:
    """Random rotation, translation, and scale changes."""
    def __init__(self, prob=0.5, degrees=15, translate=0.1, scale=(0.9, 1.1)):
        self.prob = prob
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        
    def __call__(self, img, mask):
        if random.random() < self.prob:
            h, w = img.shape
            angle = random.uniform(-self.degrees, self.degrees)
            tx = random.uniform(-self.translate, self.translate) * w
            ty = random.uniform(-self.translate, self.translate) * h
            s = random.uniform(*self.scale)
            
            # Compute transformation matrix
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, s)
            M[0, 2] += tx
            M[1, 2] += ty
            
            # Apply affine transformation
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            
        return img, mask

class ComposeTransforms:
    """Sequentially chains multiple augmentations."""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

def get_train_transforms():
    return ComposeTransforms([
        RandomAffine(prob=0.5, degrees=15, translate=0.1, scale=(0.95, 1.05)),
        RandomElasticDeformation(prob=0.3, alpha=150, sigma=15),
        RandomSpeckleNoise(prob=0.5, std_range=(0.05, 0.15))
    ])

def apply_mixup_or_cutmix(images, masks, labels, alpha=0.2, prob=0.5):
    """
    Applies MixUp or CutMix to a batch of tensors.
    Inputs:
        images: tensor [B, C, H, W]
        masks: tensor [B, C, H, W]
        labels: dict of label tensors
    """
    if random.random() > prob or images.size(0) < 2:
        return images, masks, labels, None
        
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    
    # Sample mixing weight lambda (lam) from Beta distribution
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    
    # Decide between MixUp and CutMix
    is_cutmix = random.random() > 0.5
    
    mixed_images = images.clone()
    mixed_masks = masks.clone()
    
    if is_cutmix:
        # CutMix logic
        h, w = images.size(2), images.size(3)
        # Bounding box coordinates
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Radius of cut box
        r = np.sqrt(1.0 - lam)
        cut_w = np.clip(int(w * r), 1, w)
        cut_h = np.clip(int(h * r), 1, h)
        
        # Box bounds
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Paste patch from permuted batch
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        mixed_masks[:, :, y1:y2, x1:x2] = masks[index, :, y1:y2, x1:x2]
        
        # Adjust lambda based on actual area cut
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (h * w))
    else:
        # MixUp logic
        mixed_images = lam * images + (1.0 - lam) * images[index]
        mixed_masks = lam * masks + (1.0 - lam) * masks[index]
        
    mix_info = {
        "index": index,
        "lam": lam
    }
    
    return mixed_images, mixed_masks, labels, mix_info
