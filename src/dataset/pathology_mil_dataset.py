"""ThyUS2Path 病理数据的 MIL(多示例学习)bag 封装。

把 PathologyDataset 的 flat 单帧列表按患者聚合成「包(bag)」:一个患者 = 一个
训练/评估单位,其所有超声帧堆叠为 [N,1,H,W],共享一个患者级 histo_label。

复用 pathology_dataset 已验证**无泄漏**的患者级切分:
    build_pathology_samples(扫描+标签广播) + patient_grouped_split(患者分层划分)。
本模块只新增 bag 封装层,不改动上述函数,故 test_pathology_dataset.py 不受影响。

评估铁律(见 memory/patient-leakage-and-probe-trap.md):MIL 天然是患者级——一个
bag 一个预测,AUC 在患者上算,从根上杜绝「同患者帧跨 train/val」的泄漏陷阱。
"""

from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def build_patient_bags(samples: List[Dict]) -> List[Dict]:
    """把 flat 单帧 samples 按 patient_key 聚合成 bag 列表。

    复用 patient_grouped_split 内部同款 defaultdict 分组(pathology_dataset.py:162-165)。
    Returns: list of dict {patient_key, label(患者级), img_paths: List[str]}。
    同患者所有帧 label 一致,取其一;顺序按 patient_key 排序,保证可复现。
    """
    by_patient: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        by_patient[s["patient_key"]].append(s)

    bags: List[Dict] = []
    for pkey in sorted(by_patient.keys()):
        frames = by_patient[pkey]
        bags.append({
            "patient_key": pkey,
            "label": int(frames[0]["label"]),
            "img_paths": [f["img_path"] for f in frames],
        })
    return bags


class PathologyMILDataset(Dataset):
    """MIL bag 数据集。每个样本 = 一个患者的全部帧。

    __getitem__ 返回 (bag[N,1,H,W]f32∈[0,1], label:int, patient_key:str)。
    N 因患者而异 → DataLoader 须用 batch_size=1 或自定义 collate(见训练脚本)。

    max_frames_per_bag:单患者帧数上界(防逐帧过 backbone 的显存/耗时 OOM;官方容器
        显存 32G、2026-08-12 nvidia-smi 实测)。
        超出则**随机采样**(可复现 seed),不足则全取。
    """

    def __init__(
        self,
        bags: List[Dict],
        img_size: int = 512,
        transform=None,
        max_frames_per_bag: int = 32,
        seed: int = 42,
        random_frames: bool = True,
    ):
        """random_frames=True(训练):超出上界时随机采样帧(数据增强)。
        random_frames=False(评估):固定取前 N 帧,保证评估确定性。"""
        self.bags = bags
        self.img_size = img_size
        self.transform = transform
        self.max_frames_per_bag = max_frames_per_bag
        self.random_frames = random_frames
        self._rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.bags)

    def _load_frame(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"图像读取失败: {path}")
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))
        return img

    def __getitem__(self, idx):
        bag = self.bags[idx]
        paths = bag["img_paths"]
        # 超上界则采样，训练随机、评估固定前 N 帧（确定性）。
        if self.max_frames_per_bag and len(paths) > self.max_frames_per_bag:
            if self.random_frames:
                sel = self._rng.choice(len(paths), self.max_frames_per_bag, replace=False)
                paths = [paths[i] for i in sorted(sel)]
            else:
                paths = paths[: self.max_frames_per_bag]

        frames = []
        for p in paths:
            img = self._load_frame(p)
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            if self.transform is not None:
                img, mask = self.transform(img, mask)
            frames.append(torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0)

        bag_tensor = torch.stack(frames, dim=0)  # [N, 1, H, W]
        return bag_tensor, int(bag["label"]), bag["patient_key"]

    def patient_class_counts(self) -> Dict[int, int]:
        """**患者级**类别计数 {0: n_benign, 1: n_malignant}(非帧级),供 CE class-weight。

        用患者级而非帧级,避免帧多的患者在类权重里被重复计数。
        """
        c = defaultdict(int)
        for bag in self.bags:
            c[int(bag["label"])] += 1
        return dict(c)
