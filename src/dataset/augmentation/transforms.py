import random
import numpy as np
import cv2


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


class DepthAttenuation:
    """Simulates ultrasound signal attenuation over depth (vertical decay)."""
    def __init__(self, prob=0.3, attenuation_range=(0.1, 0.4)):
        self.prob = prob
        self.attenuation_range = attenuation_range
        
    def __call__(self, img, mask):
        if random.random() < self.prob:
            h, w = img.shape
            mu = random.uniform(*self.attenuation_range)
            y = np.linspace(0, 1, h).reshape(h, 1)
            decay = np.exp(-mu * y)
            img = (img.astype(float) * decay).astype(np.uint8)
        return img, mask


class GaussianShadow:
    """Simulates acoustic shadowing below highly attenuating structures like calcifications."""
    def __init__(self, prob=0.3):
        self.prob = prob
        
    def __call__(self, img, mask):
        if random.random() < self.prob and mask.sum() > 0:
            h, w = img.shape
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                idx = random.randint(0, len(xs) - 1)
                cx, cy = xs[idx], ys[idx]
                shadow_width = random.randint(10, 30)
                x_coords = np.arange(w)
                shadow_profile = np.exp(-((x_coords - cx) ** 2) / (2 * (shadow_width / 2) ** 2))
                shadow_factor = random.uniform(0.3, 0.7)
                
                # Apply vertical shadow from cy to bottom
                shadow_mask = 1.0 - shadow_factor * shadow_profile
                for y in range(cy, h):
                    img[y, :] = np.clip(img[y, :].astype(float) * shadow_mask, 0, 255).astype(np.uint8)
        return img, mask


class HazeArtifact:
    """Simulates near-field clutter/haze noise typical in ultrasound scanning."""
    def __init__(self, prob=0.2):
        self.prob = prob
        
    def __call__(self, img, mask):
        if random.random() < self.prob:
            h, w = img.shape
            nh, nw = h // 8, w // 8
            noise = np.random.normal(0, 15, (nh, nw))
            noise_resized = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
            y_gradient = np.linspace(1.0, 0.0, h).reshape(h, 1)
            img = np.clip(img.astype(float) + noise_resized * y_gradient, 0, 255).astype(np.uint8)
        return img, mask


class SpeckleReduction:
    """Simulates speckle noise reduction filtering (e.g. bilateral filtering)."""
    def __init__(self, prob=0.2):
        self.prob = prob
        
    def __call__(self, img, mask):
        if random.random() < self.prob:
            img = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)
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
        RandomSpeckleNoise(prob=0.5, std_range=(0.05, 0.15)),
        DepthAttenuation(prob=0.3),
        GaussianShadow(prob=0.3),
        HazeArtifact(prob=0.2),
        SpeckleReduction(prob=0.2)
    ])
