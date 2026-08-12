"""数据集加载/适配包(患者级分组、多源适配、官方数据加载、阶段采样)。

空包 __init__:dataset 下含 .py 且被 `src.training`/脚本绝对导入作为包使用
(python -m 根目录导入,迫追溯包回溯)。各模块直接 `from src.dataset.xxx import`,
与同级包一致。
"""