"""gatekeeper 真实性/规范性审计包(赛道七已豁免计分,防御性工程)。

补 __init__.py:src/gatekeeper 含 .py 且被 `src.*.gatekeeper.*` 绝对导入当作包使用
(python -m 根目录导入,无 setup.py 安装,须显式标包回溯 namespace)。保持空包,
各模块直接 `from src.gatekeeper.xxx import`,与同级 service/inference 包风格一致。
"""