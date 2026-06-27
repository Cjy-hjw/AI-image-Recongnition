import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from skimage.feature import graycomatrix, graycoprops

def compute_ela(img_np, quality=90):
    """
    Computes Error Level Analysis (JPEG re-compression error).
    Synthesized (GAN/Diffusion) images show different reconstruction error distributions.
    """
    # Compress image as JPEG
    _, encoded = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, quality])
    decompressed = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    
    # Calculate absolute difference
    ela_map = cv2.absdiff(img_np, decompressed)
    return ela_map

def extract_glcm_features(img_np):
    """
    Extracts Gray-Level Co-occurrence Matrix (GLCM) texture descriptors.
    Helps distinguish biological tissue from physical phantom materials (PVA, agar).
    """
    # Downsample gray levels to 16 for computational efficiency
    img_coarse = (img_np // 16).astype(np.uint8)
    
    # Compute GLCM
    glcm = graycomatrix(img_coarse, distances=[1, 2], angles=[0, np.pi/4], levels=16, symmetric=True, normed=True)
    
    # Extract properties
    contrast = graycoprops(glcm, 'contrast').mean()
    # 零方差(如理想 phantom 平滑图)时 GLCM correlation 除零→NaN,会沿特征向量
    # 传播污染标准化与 CrossEntropy,导致 authenticity 训练整批 NaN。此处守卫。
    correlation = np.nan_to_num(graycoprops(glcm, 'correlation').mean())
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    
    return [contrast, correlation, energy, homogeneity]

def extract_frequency_features(img_np):
    """Extracts high-frequency statistics using Laplacian filtering."""
    laplacian = cv2.Laplacian(img_np, cv2.CV_64F)
    mean = np.mean(laplacian)
    std = np.std(laplacian)
    return [mean, std]

def extract_authenticity_features(img_np):
    """Aggregates texture, frequency, and ELA features into a single vector."""
    glcm_feat = extract_glcm_features(img_np)
    freq_feat = extract_frequency_features(img_np)
    
    ela = compute_ela(img_np)
    ela_mean = np.mean(ela)
    ela_std = np.std(ela)
    
    features = glcm_feat + freq_feat + [ela_mean, ela_std]
    return np.array(features, dtype=np.float32)

class AuthenticityClassifier(nn.Module):
    """
    A simple, robust MLP classifier that categorizes images as:
    0: Real Human Scan
    1: Physical Phantom
    2: Synthetic (GAN/Diffusion)
    """
    def __init__(self, input_dim=8, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

class AuthenticityModule:
    """High-level class wrapping feature extraction and prediction."""
    def __init__(self):
        self.model = AuthenticityClassifier()
        self.model.eval()
        # 训练集特征标准化参数(随权重一起加载);None 时表示旧格式权重,跳过标准化
        self.feat_mean = None
        self.feat_std = None

    def load_weights(self, path):
        if os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu')
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                self.model.load_state_dict(ckpt["state_dict"])
                fm, fs = ckpt.get("feat_mean"), ckpt.get("feat_std")
                self.feat_mean = None if fm is None else np.asarray(fm, dtype=np.float32)
                self.feat_std = None if fs is None else np.asarray(fs, dtype=np.float32)
            else:
                # 兼容旧的纯 state_dict 格式(无标准化参数)
                self.model.load_state_dict(ckpt)

    def predict(self, img_np):
        features = extract_authenticity_features(img_np)
        # 应用与训练一致的特征标准化(否则训练/推理分布不一致,分类失效)
        if self.feat_mean is not None and self.feat_std is not None:
            features = (features - self.feat_mean) / (self.feat_std + 1e-6)
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(feat_tensor)
            prob = torch.softmax(logits, dim=1).numpy()[0]
            pred = np.argmax(prob)
        return int(pred), prob
