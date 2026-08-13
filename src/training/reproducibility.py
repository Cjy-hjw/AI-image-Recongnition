"""可复现性四件套的共享助手 —— 供 `scripts/train_official.py` 使用。

`train.py`(legacy 入口)有一份自己的实测过的 `_seed_worker` + 内联四件套(2026-08-11
崩塌复盘后补),那份**不动**(交付关键路径,已验证)。本模块是官方三头训练脚本的
规范实现,四件套的语义与 `train.py` 逐条对齐,注释里保留"为什么缺一不可"。

四件套(缺一都不可复现,③ 是最容易漏的):
  ① 全局 RNG(random/numpy/torch/cuda)
  ② sampler / DataLoader 各用**独立** generator(不共用一个对象)
  ③ DataLoader worker 子进程的 numpy/random 播种(worker_init_fn)
  ④ cudnn benchmark/deterministic 取舍

血泪教训(train.py:67-104):`environment.seed` 曾从未被任何代码读取,同命令跑两次
结果不同 —— 后果不是"不优雅"而是**训练对拍/消融不可信**,崩塌复盘时正因单 epoch
不可复现、组间差异被噪声吞掉,才放弃了分离 lr / 物理耦合各自贡献的归因消融。
"""
import random
from typing import Tuple

import numpy as np
import torch


def seed_worker(worker_id: int) -> None:
    """DataLoader worker 子进程的 RNG 播种(四件套之③)。

    主进程的 `torch.manual_seed` **管不住 worker 子进程里 numpy/random 的 RNG**,
    而随机增强跑在 worker 里、用的正是 `random`/`numpy` —— 不设这个 hook,只播主
    进程的种子,训练照样不可复现。PyTorch 已为每个 worker 播好 torch RNG
    (base_seed + worker_id),这里只补 numpy / random 两路。

    与 `train.py::_seed_worker` 字节等价(有意重复:那份属 legacy 入口不动,见模块
    docstring;两处若漂移各自的测试会红)。
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def configure_reproducibility(
    seed: int, deterministic: bool = False
) -> Tuple[torch.Generator, torch.Generator]:
    """播全局 RNG(①)+ 建两个独立 generator(②)+ 定 cudnn 取舍(④)。

    Args:
        seed: 主种子。
        deterministic: True → 关 cudnn.benchmark、开 cudnn.deterministic(牺牲速度换
            逐位复现,做归因消融/对拍时才需要);False(默认)→ benchmark 优先,seed 已
            保证数据顺序/增强/权重初始化可复现,cudnn 非确定性只影响卷积算法的浮点尾数。

    Returns:
        (sampler_generator, loader_generator) —— **两个独立对象**,不共用一个。
        sampler 的 generator 喂 `torch.multinomial`(决定抽哪些样本),loader 的
        generator 派生各 worker 的 base_seed;共用会把两条流交错在同一 RNG 状态上,
        改 `num_workers` 就能连带改变采样序列(脆的隐式耦合),故分开。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed + 1)  # +1:与 sampler 错开,避免两流同相位

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    return sampler_generator, loader_generator
