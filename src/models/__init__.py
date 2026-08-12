"""模型架构包(multi-task 主干 + 增量三头 + 各损失头)。

空包 __init__:models 含 .py 且被 `src.inference.model_runner`/`src.training` 绝对导入
作为包使用(python -m 根目录导入,无 setup.py,须显式标包)。各模块直接
`from src.models.xxx import`,与同级包一致。
"""