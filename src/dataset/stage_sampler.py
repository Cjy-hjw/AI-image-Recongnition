"""
比例动态采样器 - 阶段感知的数据源采样策略

用于解决多源混训中的样本规模悬殊问题（99例 vs 17,412帧）。
通过动态调整各数据源的采样概率，防止网络对小样本高质量库遗忘。

问题背景：
- Stanford AIMI: 17,412帧（视频）
- ThyroidXL: 11,635张（静态）
- ThyUS2Path: 8,508张（JPG）
- DDTI: 仅99例（高质量标注但样本极少）

若直接均匀采样，小样本集每epoch仅被访问1-2次，网络会快速遗忘。

策略：
- Stage 1 (Epoch 1-50): 70%大样本集，30%小样本集
  目的：稳定编码器低层几何表达
- Stage 2 (Epoch 51+): 50%大样本集，50%小样本集
  目的：防止小样本遗忘，平衡各源贡献

实现：
- 继承torch.utils.data.Sampler
- 根据当前epoch动态返回采样indices
- 支持按源权重加权采样（结合MultiSourceThyroidDataset）
"""

import math
import numpy as np
from typing import List, Dict, Optional
from torch.utils.data import Sampler


class StageAwareSampler(Sampler):
    """
    阶段感知采样器

    根据训练epoch动态调整各数据源的采样比例

    Args:
        dataset: MultiSourceThyroidDataset实例
        stage1_epochs: Stage 1的epoch数（默认50）
        stage1_ratios: Stage 1各源采样比例，dict {source_name: ratio}
        stage2_ratios: Stage 2各源采样比例
        current_epoch: 当前epoch（外部传入，每epoch更新）
        shuffle: 是否在每个epoch内打乱（默认True）

    Usage:
        >>> dataset = MultiSourceThyroidDataset([...])
        >>> sampler = StageAwareSampler(
        ...     dataset,
        ...     stage1_ratios={'ThyroidXL': 0.7, 'DDTI': 0.3},
        ...     stage2_ratios={'ThyroidXL': 0.5, 'DDTI': 0.5}
        ... )
        >>> # 训练循环中
        >>> for epoch in range(epochs):
        ...     sampler.set_epoch(epoch)
        ...     for batch in DataLoader(dataset, sampler=sampler):
        ...         ...
    """

    def __init__(
        self,
        dataset,
        stage1_epochs: int = 50,
        stage1_ratios: Optional[Dict[str, float]] = None,
        stage2_ratios: Optional[Dict[str, float]] = None,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.stage1_epochs = stage1_epochs
        self.shuffle = shuffle
        self.current_epoch = 0

        # 默认比例：Stage1偏向大样本，Stage2平衡
        self.stage1_ratios = stage1_ratios or {'large': 0.7, 'small': 0.3}
        self.stage2_ratios = stage2_ratios or {'large': 0.5, 'small': 0.5}

        # 构建源索引映射
        self._build_source_indices()

    def _build_source_indices(self):
        """构建每个数据源的样本索引列表"""
        # 假设dataset是MultiSourceThyroidDataset
        # 其有source_names和index_map属性
        if not hasattr(self.dataset, 'source_names'):
            raise ValueError("dataset必须是MultiSourceThyroidDataset")

        self.source_indices = {}
        for idx, (ds_pos, local_idx) in enumerate(self.dataset.index_map):
            source_name = self.dataset.source_names[ds_pos]
            if source_name not in self.source_indices:
                self.source_indices[source_name] = []
            self.source_indices[source_name].append(idx)

        # 打印统计
        print(f"[StageAwareSampler] 数据源统计:")
        for name, indices in self.source_indices.items():
            print(f"  {name}: {len(indices)} 样本")

    def set_epoch(self, epoch: int):
        """更新当前epoch（每个epoch开始时调用）"""
        self.current_epoch = epoch

    def _get_current_ratios(self) -> Dict[str, float]:
        """根据当前epoch返回采样比例"""
        if self.current_epoch < self.stage1_epochs:
            return self.stage1_ratios
        else:
            return self.stage2_ratios

    def __iter__(self):
        """生成采样索引"""
        ratios = self._get_current_ratios()

        # 计算每个源应采样的数量
        total_samples = len(self.dataset)
        source_samples = {}

        for source_name, indices in self.source_indices.items():
            # 根据ratio计算该源在本epoch的采样数
            ratio = ratios.get(source_name, 1.0 / len(self.source_indices))
            n_samples = int(total_samples * ratio)
            source_samples[source_name] = n_samples

        # 归一化（确保总和=total_samples）
        actual_total = sum(source_samples.values())
        if actual_total != total_samples:
            # 调整最大源的样本数以匹配
            largest_source = max(source_samples, key=source_samples.get)
            source_samples[largest_source] += (total_samples - actual_total)

        # 从各源采样
        sampled_indices = []
        for source_name, n_samples in source_samples.items():
            indices = self.source_indices[source_name]

            if n_samples >= len(indices):
                # 过采样（重复采样）
                repeats = n_samples // len(indices)
                remainder = n_samples % len(indices)

                sampled = indices * repeats
                if remainder > 0:
                    sampled.extend(np.random.choice(indices, remainder, replace=False).tolist())
            else:
                # 欠采样（随机选取子集）
                sampled = np.random.choice(indices, n_samples, replace=False).tolist()

            sampled_indices.extend(sampled)

        # 打乱整个采样序列
        if self.shuffle:
            np.random.shuffle(sampled_indices)

        return iter(sampled_indices)

    def __len__(self):
        return len(self.dataset)


class ProportionalStageSampler(Sampler):
    """
    简化版比例采样器（按源配置的ratio直接加权）

    与MultiSourceThyroidDataset的sample_weights配合使用

    Args:
        sample_weights: 每个样本的采样权重（从dataset.get_sample_weights()获取）
        num_samples: 每个epoch采样的总样本数（默认等于数据集大小）
        replacement: 是否有放回采样（默认True，允许重复）

    Usage:
        >>> dataset = MultiSourceThyroidDataset([
        ...     {'name': 'ThyroidXL', 'path': '...', 'ratio': 0.7},
        ...     {'name': 'DDTI', 'path': '...', 'ratio': 0.3},
        ... ])
        >>> weights = dataset.get_sample_weights()
        >>> sampler = ProportionalStageSampler(weights, len(dataset))
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=32)
    """

    def __init__(self, sample_weights: List[float], num_samples: int, replacement: bool = True):
        self.sample_weights = np.array(sample_weights, dtype=np.float64)
        self.num_samples = num_samples
        self.replacement = replacement

        # 归一化权重
        self.sample_weights /= self.sample_weights.sum()

    def __iter__(self):
        # 按权重采样
        indices = np.random.choice(
            len(self.sample_weights),
            size=self.num_samples,
            replace=self.replacement,
            p=self.sample_weights
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


# ========== 辅助工具 ==========

def compute_source_statistics(dataset, sampler: StageAwareSampler, num_iterations: int = 1000):
    """
    统计采样器在多个epoch中的实际采样分布

    Args:
        dataset: 数据集
        sampler: 采样器实例
        num_iterations: 模拟epoch数

    Returns:
        dict: {epoch: {source_name: count}}
    """
    stats = {}

    for epoch in range(num_iterations):
        sampler.set_epoch(epoch)
        indices = list(iter(sampler))

        # 统计各源样本数
        source_counts = {name: 0 for name in sampler.source_indices.keys()}
        for idx in indices:
            ds_pos, _ = dataset.index_map[idx]
            source_name = dataset.source_names[ds_pos]
            source_counts[source_name] += 1

        stats[epoch] = source_counts

    return stats


def visualize_sampling_distribution(stats: dict, stage1_epochs: int = 50):
    """
    可视化采样分布变化（打印统计）

    Args:
        stats: compute_source_statistics返回的统计结果
        stage1_epochs: Stage 1的epoch数
    """
    print("\n[采样分布统计]")
    print(f"Stage 1 (Epoch 1-{stage1_epochs}):")

    # Stage 1平均
    stage1_stats = {k: v for k, v in stats.items() if k < stage1_epochs}
    if stage1_stats:
        source_names = list(next(iter(stage1_stats.values())).keys())
        for name in source_names:
            avg_count = np.mean([epoch_stat[name] for epoch_stat in stage1_stats.values()])
            print(f"  {name}: 平均 {avg_count:.1f} 样本/epoch")

    # Stage 2平均
    stage2_stats = {k: v for k, v in stats.items() if k >= stage1_epochs}
    if stage2_stats:
        print(f"\nStage 2 (Epoch {stage1_epochs+1}+):")
        for name in source_names:
            avg_count = np.mean([epoch_stat[name] for epoch_stat in stage2_stats.values()])
            print(f"  {name}: 平均 {avg_count:.1f} 样本/epoch")


# ========== 单元测试 ==========

if __name__ == '__main__':
    # Mock数据集（模拟MultiSourceThyroidDataset）
    class MockDataset:
        def __init__(self):
            self.source_names = ['ThyroidXL', 'DDTI']
            # ThyroidXL: 10000样本，DDTI: 100样本
            self.index_map = []
            for i in range(10000):
                self.index_map.append((0, i))  # ds_pos=0 -> ThyroidXL
            for i in range(100):
                self.index_map.append((1, i))  # ds_pos=1 -> DDTI

        def __len__(self):
            return len(self.index_map)

    dataset = MockDataset()

    # 创建采样器
    sampler = StageAwareSampler(
        dataset,
        stage1_epochs=50,
        stage1_ratios={'ThyroidXL': 0.7, 'DDTI': 0.3},
        stage2_ratios={'ThyroidXL': 0.5, 'DDTI': 0.5},
    )

    # 模拟100个epoch的采样
    print("\n[测试] 模拟100个epoch...")
    stats = compute_source_statistics(dataset, sampler, num_iterations=100)

    # 可视化
    visualize_sampling_distribution(stats, stage1_epochs=50)

    # 验证
    stage1_ddti_avg = np.mean([stats[e]['DDTI'] for e in range(50)])
    stage2_ddti_avg = np.mean([stats[e]['DDTI'] for e in range(50, 100)])

    print(f"\n[验证]")
    print(f"Stage 1 DDTI占比: {stage1_ddti_avg / len(dataset) * 100:.1f}% (预期~30%)")
    print(f"Stage 2 DDTI占比: {stage2_ddti_avg / len(dataset) * 100:.1f}% (预期~50%)")

    assert 25 <= stage1_ddti_avg / len(dataset) * 100 <= 35, "Stage1比例不符合预期"
    assert 45 <= stage2_ddti_avg / len(dataset) * 100 <= 55, "Stage2比例不符合预期"

    print("\n[OK] 比例动态采样器测试通过")
