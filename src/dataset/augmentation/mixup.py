import random
import numpy as np
import torch


def apply_mixup_or_cutmix(images, masks, labels, alpha=0.2, prob=0.5):
    """
    Applies Dominant-Region CutMix (DR-CutMix) or standard MixUp to a batch of tensors.
    """
    if random.random() > prob or images.size(0) < 2:
        return images, masks, labels, None
        
    batch_size = images.size(0)
    device = images.device
    index = torch.randperm(batch_size, device=device)
    
    # Sample mixing weight lambda (lam) from Beta distribution
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    
    # Decide between MixUp and CutMix
    is_cutmix = random.random() > 0.5
    
    mixed_images = images.clone()
    mixed_masks = masks.clone()
    
    # Initialize sample-wise lams and mask_weight
    lams = torch.full((batch_size,), float(lam), device=device)
    mask_weight = torch.ones(batch_size, device=device)
    
    if is_cutmix:
        h, w = images.size(2), images.size(3)
        # Bounding box coordinates
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Radius of cut box
        r = np.sqrt(1.0 - lam)
        cut_w = np.clip(int(w * r), 1, w)
        cut_h = np.clip(int(h * r), 1, h)
        
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Paste patch from permuted batch (Donor)
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        mixed_masks[:, :, y1:y2, x1:x2] = masks[index, :, y1:y2, x1:x2]
        
        # Dominant-Region CutMix (DR-CutMix) label inheritance logic
        for i in range(batch_size):
            donor_nodule = masks[index[i], :, y1:y2, x1:x2].sum().item()
            total_nodule = mixed_masks[i].sum().item()
            
            if total_nodule > 0:
                R = donor_nodule / (total_nodule + 1e-6)
                if R >= 0.65:
                    lams[i] = 0.0      # inherits 100% of donor labels
                    mask_weight[i] = 1.0
                elif R <= 0.35:
                    lams[i] = 1.0      # inherits 100% of recipient labels
                    mask_weight[i] = 1.0
                else:
                    lams[i] = 0.0      # masked out for classification losses
                    mask_weight[i] = 0.0
            else:
                lams[i] = 0.0      # no nodule present
                mask_weight[i] = 0.0   # completely mask out classification losses
    else:
        # Standard MixUp logic
        mixed_images = lam * images + (1.0 - lam) * images[index]
        mixed_masks = lam * masks + (1.0 - lam) * masks[index]
        
    mix_info = {
        "index": index,
        "lam": lams,
        "mask_weight": mask_weight
    }
    
    return mixed_images, mixed_masks, labels, mix_info
