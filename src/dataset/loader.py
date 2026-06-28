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

    source_label_mask: 可选 dict,提供**源级**标签有效性默认值(键为 composition/
        echogenicity/shape/margin/foci/malignancy/mask,值 0/1)。用于多源混训时
        声明"本数据源整体缺哪类标签"(如 TN3K 缺 malignancy),免去逐样本写
        label_mask。优先级:样本 meta.json 的 label_mask > source_label_mask > 默认 1。
    """
    def __init__(self, data_dir, img_size=256, transform=None, source_label_mask=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform
        self.source_label_mask = source_label_mask or {}
        
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

        # Label validity mask: 1 = label available, 0 = label absent (skip gradient)
        # 优先级:样本 meta.json 的 label_mask > 源级 source_label_mask > 默认 1。
        # 源级默认让多源混训可直接声明"本源整体缺某类标签",无需逐样本标注。
        lm = meta.get("label_mask", {})
        slm = self.source_label_mask
        concept_tasks = ("composition", "echogenicity", "shape", "margin", "foci", "malignancy")
        for task in concept_tasks:
            labels[f"valid_{task}"] = int(lm.get(task, slm.get(task, 1)))
        # Segmentation mask validity
        labels["valid_mask"] = int(lm.get("mask", slm.get("mask", 1)))
        
        # Apply data augmentations if specified
        if self.transform is not None:
            img, mask = self.transform(img, mask)
            
        # Normalize image to [0, 1] float32 tensor
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        
        return img_tensor, mask_tensor, labels


class MultiSourceThyroidDataset(Dataset):
    """多源甲状腺数据集骨架:拼接多个 `ThyroidDataset`,各源带自己的源级标签有效性。

    为开赛后"多公开数据集混训 + 部分标签(SSCBM)"预留,**当前不接真数据**。
    每个 source 配置: {name, path, label_mask(源级缺标签声明), ratio(加权采样权重,可选)}。
    缺标签样本经 ThyroidDataset 的 source_label_mask 标 valid_*=0,下游 masked loss
    自动零梯度——与现有单源路径完全兼容,无需改损失。

    设计取舍:沿用 ThyroidDataset 的目录契约(image.png/mask.png/meta.json),不假设
    官方数据形态;真实接入时只需把各公开集转成该目录结构即可。
    """
    def __init__(self, sources, img_size=256, transform=None):
        self.sources = sources
        self.datasets = []
        self.source_names = []
        self.index_map = []  # 全局 idx → (ds_pos, local_idx)
        self.sample_weights = []  # 供 WeightedRandomSampler(按源 ratio 均摊到样本)
        for src in sources:
            ds = ThyroidDataset(
                src["path"], img_size=img_size, transform=transform,
                source_label_mask=src.get("label_mask"),
            )
            ds_pos = len(self.datasets)
            self.datasets.append(ds)
            self.source_names.append(src.get("name", f"source{ds_pos}"))
            n = len(ds)
            ratio = float(src.get("ratio", 1.0))
            per_sample_w = (ratio / n) if n > 0 else 0.0
            for local_idx in range(n):
                self.index_map.append((ds_pos, local_idx))
                self.sample_weights.append(per_sample_w)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ds_pos, local_idx = self.index_map[idx]
        return self.datasets[ds_pos][local_idx]

    def get_sample_weights(self):
        """返回与样本对齐的采样权重(源 ratio 均摊),供 WeightedRandomSampler 实现按源配比。"""
        return list(self.sample_weights)
