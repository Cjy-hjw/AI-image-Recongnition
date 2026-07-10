"""
多源数据集适配器层 - 统一异构标注格式

用于对接ThyroidXL、ThyUS2Path等公开数据集的原生格式，转换为项目统一接口。

设计原则：
1. 保持原始数据不变（只读访问，不转换文件）
2. 运行时动态适配，无需预处理
3. 自动生成task_mask（标注可用性掩膜）
4. 与现有ThyroidDataset/MultiSourceThyroidDataset兼容

架构：
    AbstractAdapter (抽象基类)
        ├─ ThyroidXLAdapter (HuggingFace格式)
        ├─ ThyUS2PathAdapter (Figshare JPG格式)
        └─ StandardAdapter (项目标准格式：image.png/mask.png/meta.json)
"""

# TODO(官方数据后): 按 rules.md §6 拆分为 adapters/{base,thyroidxl,thyus2path}.py + adapted_dataset.py
# 当前 adapter 为空壳骨架(_load_metadata/get_labels 未真实实现),等官方 schema 落地后按真实格式重写再拆分。

import json
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset as TorchDataset


class ThyroidDatasetAdapter(ABC):
    """
    抽象适配器基类

    所有子类必须实现统一接口，返回标准化数据格式
    """

    def __init__(self, data_root: str):
        """
        Args:
            data_root: 数据集根目录
        """
        self.data_root = Path(data_root)
        self._samples = None  # 延迟加载

    @abstractmethod
    def _load_samples(self) -> list:
        """
        扫描数据集，返回样本列表

        Returns:
            list of dict: 每个样本的元数据（路径、ID等）
        """
        pass

    @abstractmethod
    def get_image(self, sample_meta: dict) -> np.ndarray:
        """
        加载图像

        Args:
            sample_meta: _load_samples返回的单个样本元数据

        Returns:
            np.ndarray: 图像数组，uint8 [H, W] 或 [H, W, C]
        """
        pass

    @abstractmethod
    def get_mask(self, sample_meta: dict) -> Optional[np.ndarray]:
        """
        加载分割掩膜

        Args:
            sample_meta: 样本元数据

        Returns:
            np.ndarray or None: 掩膜数组 [H, W]，float32，值域[0,1]
                                若无掩膜返回None
        """
        pass

    @abstractmethod
    def get_labels(self, sample_meta: dict) -> Dict[str, Any]:
        """
        获取标注标签

        Args:
            sample_meta: 样本元数据

        Returns:
            dict: 标签字典，包含：
                - composition: int (0-3)
                - echogenicity: int (0-3)
                - shape: int (0-1)
                - margin: int (0-3)
                - foci: list[int] 长度4
                - malignancy: int (0-1)
                - patient_id: str
        """
        pass

    @abstractmethod
    def get_task_mask(self, sample_meta: dict) -> Dict[str, int]:
        """
        生成任务掩膜（标注可用性）

        Args:
            sample_meta: 样本元数据

        Returns:
            dict: {task_name: availability}
                - 'mask': 1 if 有分割掩膜 else 0
                - 'composition': 1 if 有成分标注 else 0
                - 'echogenicity': 1 if 有回声标注 else 0
                - 'shape': 1 if 有形状标注 else 0
                - 'margin': 1 if 有边缘标注 else 0
                - 'foci': 1 if 有钙化标注 else 0
                - 'malignancy': 1 if 有良恶性标注 else 0
        """
        pass

    @property
    def samples(self) -> list:
        """延迟加载样本列表"""
        if self._samples is None:
            self._samples = self._load_samples()
        return self._samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        """
        统一获取接口

        Returns:
            (image, mask, labels_with_task_mask)
        """
        sample_meta = self.samples[idx]

        image = self.get_image(sample_meta)
        mask = self.get_mask(sample_meta)
        labels = self.get_labels(sample_meta)
        task_mask = self.get_task_mask(sample_meta)

        # 合并labels和task_mask
        labels.update({f'valid_{k}': v for k, v in task_mask.items()})

        return image, mask, labels


# ========== 具体适配器实现 ==========

class ThyroidXLAdapter(ThyroidDatasetAdapter):
    """
    ThyroidXL数据集适配器（真实 HuggingFace 格式）

    数据集结构（两个 split 标注来源完全不同，不可混用同一份反查表）：
    ThyroidXL/
        train/
            images/       {patient_id}_{hash}_{idx}.png
            masks/        对应分割mask
            labels_orig/  {同stem}.json，每图独立标注：{"label": "actinh"(恶性)/"lanhtinh"(良性), bbox, polygon}
        test/
            images/
            masks/
            test_annotations.json  ← 唯一标注文件，仅覆盖 test split

    test_annotations.json 顶层结构：
    {"info": {patient_id: {age, gender, conclusion, nodule_1: {TIRADS, Conclusion, ...}, images:[...]}}}

    标注映射：
    - test: nodule_1.Conclusion（字符串）含"Malignant" → 1，否则 → 0
    - train: label=="actinh" → Conclusion="Malignant"；label=="lanhtinh" → Conclusion="Benign"
      （合成同形状的 nodule dict，使下游 get_labels/get_task_mask 无需区分 split）
    - 细粒度概念（composition/echogenicity/shape/margin/foci）ThyroidXL无此标注，task_mask全0
    """

    def __init__(self, data_root: str, annotations_file: str = "test/test_annotations.json"):
        super().__init__(data_root)
        self.annotations_file = self.data_root / annotations_file
        self.annotations = self._load_annotations()
        self.train_labels = self._load_train_labels()

    def _load_annotations(self) -> Dict[str, tuple]:
        """加载 test split 标注，建立 image_filename -> (patient_id, nodule_record) 反查表。"""
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"标注文件不存在: {self.annotations_file}")

        with open(self.annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        lookup: Dict[str, tuple] = {}
        for pid, pdata in data.get("info", {}).items():
            nodule = pdata.get("nodule_1") or {}
            for img_name in pdata.get("images", []):
                lookup[img_name] = (pid, nodule)
        return lookup

    def _load_train_labels(self) -> Dict[str, dict]:
        """加载 train/labels_orig/*.json（每图独立标注），建立 image_stem -> nodule_record。"""
        labels_dir = self.data_root / "train" / "labels_orig"
        result: Dict[str, dict] = {}
        if not labels_dir.exists():
            return result
        for label_file in labels_dir.glob("*.json"):
            with open(label_file, 'r', encoding='utf-8') as f:
                rec = json.load(f)
            label = rec.get("label", "")
            if label == "actinh":
                conclusion = "Malignant"
            elif label == "lanhtinh":
                conclusion = "Benign"
            else:
                # 未知 label 值：不声明该样本有效，避免静默产出空 Conclusion
                # 混入训练集（宁可少声明有效，不可编造映射，见 rules.md §5）
                continue
            result[label_file.stem] = {"Conclusion": conclusion}
        return result

    def _load_samples(self) -> list:
        """扫描 train/ 和 test/ 两个 images 目录，分别用各自的标注来源过滤。"""
        samples = []

        test_images_dir = self.data_root / 'test' / 'images'
        if test_images_dir.exists():
            for img_file in sorted(test_images_dir.glob('*.png')):
                if img_file.name not in self.annotations:
                    continue
                pid, nodule = self.annotations[img_file.name]
                samples.append({
                    'image_id': img_file.stem,
                    'image_path': img_file,
                    'mask_path': self.data_root / 'test' / 'masks' / img_file.name,
                    'patient_id': pid,
                    'nodule': nodule,
                })

        train_images_dir = self.data_root / 'train' / 'images'
        if train_images_dir.exists():
            for img_file in sorted(train_images_dir.glob('*.png')):
                nodule = self.train_labels.get(img_file.stem)
                if nodule is None:
                    continue
                pid = img_file.stem.split('_')[0]
                samples.append({
                    'image_id': img_file.stem,
                    'image_path': img_file,
                    'mask_path': self.data_root / 'train' / 'masks' / img_file.name,
                    'patient_id': pid,
                    'nodule': nodule,
                })
        return samples

    def get_image(self, sample_meta: dict) -> np.ndarray:
        img = cv2.imread(str(sample_meta['image_path']), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"图像加载失败: {sample_meta['image_path']}")
        return img

    def get_mask(self, sample_meta: dict) -> Optional[np.ndarray]:
        mask_path = sample_meta['mask_path']
        if not mask_path.exists():
            return None
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        return (mask > 127).astype(np.float32)

    def get_labels(self, sample_meta: dict) -> Dict[str, Any]:
        nodule = sample_meta['nodule']
        conclusion = nodule.get('Conclusion') or ''
        malignancy = 1 if 'Malignant' in conclusion or 'malignant' in conclusion else 0

        return {
            'composition': 0,   # ThyroidXL 无细粒度概念标注，占位
            'echogenicity': 0,
            'shape': 0,
            'margin': 0,
            'foci': [0, 0, 0, 0],
            'malignancy': malignancy,
            'patient_id': sample_meta['patient_id'],
            'authenticity': 1,
            'is_duplicate': 0,
            'is_tampered': 0,
        }

    def get_task_mask(self, sample_meta: dict) -> Dict[str, int]:
        nodule = sample_meta['nodule']
        has_conclusion = 1 if nodule.get('Conclusion') else 0
        return {
            'mask': 1 if sample_meta['mask_path'].exists() else 0,
            'composition': 0,   # 无细粒度标注
            'echogenicity': 0,
            'shape': 0,
            'margin': 0,
            'foci': 0,
            'malignancy': has_conclusion,
        }


class ThyUS2PathAdapter(ThyroidDatasetAdapter):
    """
    ThyUS2Path数据集适配器（Figshare原生JPG格式）

    数据集结构（假设）：
    ThyUS2Path/
        images/
            case001_view1.jpg
            case001_view2.jpg
            ...
        metadata.csv  # 或 annotations.json
            patient_id, image_file, pathology, view_type

    特点：
    - **原生JPG格式**（与官方一致，无需JPEG增强）
    - 无分割掩膜（仅图像级分类）
    - 多切面关联（同一患者多张图）
    """

    def __init__(self, data_root: str, metadata_file: str = "metadata.csv"):
        super().__init__(data_root)
        self.metadata_file = self.data_root / metadata_file
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, dict]:
        """加载元数据（简化，实际需pandas读CSV）"""
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"元数据文件不存在: {self.metadata_file}")

        # TODO: 实际实现用pandas读取CSV
        # 这里仅为示例结构
        return {}

    def _load_samples(self) -> list:
        images_dir = self.data_root / 'images'
        if not images_dir.exists():
            raise FileNotFoundError(f"图像目录不存在: {images_dir}")

        samples = []
        for img_file in sorted(images_dir.glob('*.jpg')):
            samples.append({
                'image_id': img_file.stem,
                'image_path': img_file,
            })

        return samples

    def get_image(self, sample_meta: dict) -> np.ndarray:
        # 注意：原生JPG，**不应再应用JPEG增强**
        img = cv2.imread(str(sample_meta['image_path']), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"图像加载失败: {sample_meta['image_path']}")
        return img

    def get_mask(self, sample_meta: dict) -> Optional[np.ndarray]:
        # ThyUS2Path无分割掩膜
        return None

    def get_labels(self, sample_meta: dict) -> Dict[str, Any]:
        # TODO: 从metadata读取病理标签
        return {
            'composition': 0,
            'echogenicity': 0,
            'shape': 0,
            'margin': 0,
            'foci': [0, 0, 0, 0],
            'malignancy': 0,  # 从metadata映射
            'patient_id': sample_meta['image_id'].split('_')[0],
            'authenticity': 1,
            'is_duplicate': 0,
            'is_tampered': 0,
        }

    def get_task_mask(self, sample_meta: dict) -> Dict[str, int]:
        # ThyUS2Path 理论上仅有图像级病理标签,但 _load_metadata 尚未实现(返回 {}),
        # get_labels 的 malignancy 仍是硬编码 dummy 0,并非从 metadata 映射的真标签。
        # 官方 schema 未定,标注 valid=0 避免假标签污染,数据到位后补真实映射
        # (metadata.csv 读取 + malignancy 映射实现后,此处 malignancy 改回 1)。
        return {
            'mask': 0,
            'composition': 0,
            'echogenicity': 0,
            'shape': 0,
            'margin': 0,
            'foci': 0,
            'malignancy': 0,
        }


# ========== 包装为PyTorch Dataset ==========


class AdaptedThyroidDataset(TorchDataset):
    """
    将Adapter包装为PyTorch Dataset

    与现有ThyroidDataset接口兼容，可直接替换使用
    """

    def __init__(self, adapter: ThyroidDatasetAdapter, img_size: int = 512, transform=None):
        self.adapter = adapter
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.adapter)

    def __getitem__(self, idx):
        img, mask, labels = self.adapter[idx]

        # Resize
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))

        if mask is not None:
            if mask.shape[0] != self.img_size or mask.shape[1] != self.img_size:
                mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        # Apply augmentations
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        # To tensor
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # 转换foci为tensor
        if isinstance(labels['foci'], list):
            labels['foci'] = torch.tensor(labels['foci'], dtype=torch.float32)

        return img_tensor, mask_tensor, labels

# 注: 冒烟测试应在 tests/ 补充(官方数据到位、adapter 按真实 schema 重写后)。
