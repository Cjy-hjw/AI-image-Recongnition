"""
JPEG压缩增强模块 - 域随机化（Domain Randomization）

用于抵抗官方数据JPG有损压缩导致的域偏移（25-26%性能崩塌）。
通过在训练时动态注入DCT量化伪影，强制网络对8×8块量化噪声产生抗性。

原理：
- 官方数据采用有损JPG压缩（Q7答疑明确）
- 公开数据多为PNG/DICOM无损格式
- JPG量化会抹除高频纹理（微小钙化、边缘毛刺）并引入方块效应、蚊式噪声
- 直接迁移会导致特异性大幅下降（调研报告实证：25-26%准确率跌幅）

缓解策略：
- 训练时动态JPEG重压缩，Q ~ U(30, 95)
- 覆盖极低质量到接近无损，强制网络学习鲁棒特征
- 集成到数据管道的transform阶段（旋转/缩放后、归一化前）
"""

import random
import numpy as np
import cv2
from typing import Tuple


class JPEGCompressionAugment:
    """
    动态JPEG压缩增强算子

    模拟官方有损JPG的DCT量化伪影，包括：
    - 高频系数截断（微小钙化细节丢失）
    - 8×8块边界的方块效应（Blocking Artifacts）
    - 边缘的吉布斯振荡/蚊式噪声（Mosquito Noise）

    Args:
        prob: 应用概率（默认0.7，训练时高频应用以充分暴露）
        quality_range: JPEG质量因子范围，tuple (min_q, max_q)
            - 默认(30, 95)：30覆盖极度退化，95接近无损
            - Q越低，压缩越重，伪影越明显

    Usage:
        >>> aug = JPEGCompressionAugment(prob=0.7, quality_range=(30, 95))
        >>> img_aug, mask = aug(img, mask)  # mask不变，仅图像压缩
    """

    def __init__(self, prob: float = 0.7, quality_range: Tuple[int, int] = (30, 95)):
        self.prob = prob
        self.quality_range = quality_range

        # 参数验证
        assert 0 <= prob <= 1, f"prob必须在[0,1]，当前{prob}"
        assert 1 <= quality_range[0] <= quality_range[1] <= 100, \
            f"quality_range必须在[1,100]且min<=max，当前{quality_range}"

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用JPEG压缩增强

        Args:
            img: 输入图像，uint8 [H, W] 或 [H, W, C]
            mask: 分割掩膜，保持不变（JPEG不影响标注）

        Returns:
            (img_aug, mask): 压缩后图像和原始mask
        """
        if random.random() < self.prob:
            # cv2.imencode('.jpg', ...) 对非 uint8 输入会静默失败(返回 False/空),
            # 且 JPEG 语义本就针对 8bit 像素,故断言防止归一化后的 float 图误入。
            assert img.dtype == np.uint8, \
                f"JPEG增强要求uint8输入(归一化前),当前{img.dtype}"

            # 随机采样质量因子
            quality = random.randint(*self.quality_range)

            # JPEG编码-解码循环（模拟有损压缩）
            # cv2.imencode的JPEG实现基于libjpeg，与官方处理一致
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', img, encode_param)

            # 解码回numpy数组
            img_decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)

            # 确保shape一致（极端情况下编解码可能改变）
            if img_decoded.shape != img.shape:
                img_decoded = cv2.resize(img_decoded, (img.shape[1], img.shape[0]))

            return img_decoded, mask

        return img, mask


class AdaptiveJPEGAugment:
    """
    自适应JPEG压缩增强（进阶版）

    根据训练阶段动态调整压缩强度：
    - Early stage (epoch 1-30): 温和压缩，避免过早破坏特征学习
    - Mid stage (epoch 31-80): 全范围随机，充分暴露域偏移
    - Late stage (epoch 81+): 聚焦官方可能的质量区间，精细调优

    Args:
        stage: 训练阶段 'early'/'mid'/'late'
        prob: 应用概率
    """

    def __init__(self, stage: str = 'mid', prob: float = 0.7):
        self.stage = stage
        self.prob = prob

        # 阶段质量范围映射
        self.stage_quality_map = {
            'early': (60, 95),   # 温和压缩
            'mid': (30, 95),     # 全范围
            'late': (40, 80),    # 聚焦官方可能区间（根据初赛反馈调整）
        }

        assert stage in self.stage_quality_map, \
            f"stage必须是{list(self.stage_quality_map.keys())}之一"

        self.quality_range = self.stage_quality_map[stage]

        # 组合优于复制: 复用 JPEGCompressionAugment 的编解码逻辑,仅 quality_range 来源不同
        # (此处由 stage 决定),避免两份 __call__ 逐行重复。
        self._compressor = JPEGCompressionAugment(
            prob=prob, quality_range=self.quality_range
        )

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._compressor(img, mask)


# ========== 集成到现有transform pipeline ==========

def get_train_transforms_with_jpeg(stage: str = 'mid', jpeg_prob: float = 0.7):
    """
    获取训练时transform pipeline（包含JPEG增强）

    插入位置：几何变换后、归一化前
    - RandomAffine/Elastic等先执行（保证标注对齐）
    - JPEG压缩在最后（模拟采集后存储的退化）

    Args:
        stage: 训练阶段 'early'/'mid'/'late'
        jpeg_prob: JPEG增强应用概率

    Returns:
        ComposeTransforms实例
    """
    from src.dataset.augmentation.transforms import (
        ComposeTransforms, RandomAffine, RandomElasticDeformation,
        RandomSpeckleNoise, DepthAttenuation, GaussianShadow,
        HazeArtifact, SpeckleReduction
    )

    return ComposeTransforms([
        # 几何变换（必须在JPEG前，避免插值二次退化）
        RandomAffine(prob=0.5, degrees=15, translate=0.1, scale=(0.95, 1.05)),
        RandomElasticDeformation(prob=0.3, alpha=150, sigma=15),

        # 超声物理伪影（模拟采集过程）
        RandomSpeckleNoise(prob=0.5, std_range=(0.05, 0.15)),
        DepthAttenuation(prob=0.3),
        GaussianShadow(prob=0.3),
        HazeArtifact(prob=0.2),
        SpeckleReduction(prob=0.2),

        # JPEG压缩（最后执行，模拟存储退化）
        AdaptiveJPEGAugment(stage=stage, prob=jpeg_prob),
    ])

# 注: JPEG 编解码不变性冒烟测试应在 tests/ 补充(输入须 uint8,见 __call__ 断言)。
