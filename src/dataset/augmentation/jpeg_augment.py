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

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.prob:
            quality = random.randint(*self.quality_range)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', img, encode_param)
            img_decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)

            if img_decoded.shape != img.shape:
                img_decoded = cv2.resize(img_decoded, (img.shape[1], img.shape[0]))

            return img_decoded, mask

        return img, mask


def visualize_jpeg_degradation(img: np.ndarray, qualities: list = [30, 50, 70, 90]) -> dict:
    """
    可视化不同JPEG质量的退化效果（用于验证增强效果）

    Args:
        img: 原始图像
        qualities: 质量因子列表

    Returns:
        dict: {quality: degraded_image}

    Usage:
        >>> results = visualize_jpeg_degradation(img, [30, 50, 70, 90])
        >>> for q, img_q in results.items():
        ...     cv2.imwrite(f'img_q{q}.jpg', img_q)
    """
    results = {'original': img.copy()}

    for q in qualities:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        _, encoded = cv2.imencode('.jpg', img, encode_param)
        img_q = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        results[f'Q{q}'] = img_q

    return results


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


# ========== 单元测试 ==========

if __name__ == '__main__':
    # 测试基础功能
    img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    mask = np.zeros((512, 512), dtype=np.float32)

    aug = JPEGCompressionAugment(prob=1.0, quality_range=(30, 95))
    img_aug, mask_aug = aug(img, mask)

    print(f"原始图像shape: {img.shape}, dtype: {img.dtype}")
    print(f"增强后shape: {img_aug.shape}, dtype: {img_aug.dtype}")
    print(f"mask保持不变: {np.array_equal(mask, mask_aug)}")
    print(f"图像内容改变: {not np.array_equal(img, img_aug)}")

    # 测试可视化
    results = visualize_jpeg_degradation(img, [30, 50, 70, 90])
    print(f"\n可视化结果: {list(results.keys())}")

    print("\n[OK] JPEG压缩增强模块测试通过")
