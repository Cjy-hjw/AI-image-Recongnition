"""PostToolUse Hook: 高风险文件改动 → 注入 Gate2 审查提醒。

触发：Edit/Write 后，文件在 src/models/ 或 src/training/ 下。
效果：输出提醒文字到 Claude 上下文，让 Claude 自行判断是否立即派 reviewer 子agent。
"""
import json, os, sys

HIGH_RISK_PREFIXES = (
    os.path.join("src", "models"),
    os.path.join("src", "training"),
)

try:
    data = json.load(sys.stdin)
except (json.JSONDecodeError, EOFError):
    sys.exit(0)

file_path = data.get("tool_input", {}).get("file_path", "")
if not file_path:
    sys.exit(0)

norm = os.path.normpath(file_path)
# 只关心 src/models/ 和 src/training/ 下的 .py 文件
if not norm.endswith(".py"):
    sys.exit(0)
if not any(norm.__contains__(p) for p in (
    os.path.join("src", "models"),
    os.path.join("src", "training"),
)):
    sys.exit(0)

fname = os.path.basename(file_path)
print(
    f"[GATE2-REMINDER] 修改了高风险文件 `{fname}`。\n"
    "rules.md §2 要求: 涉及梯度路由/优化器绑定/损失数学管道的改动须派 reviewer 子agent (Gate2)。\n"
    "若本次改动属于以上范畴，请在继续前完成 Gate2 审查。",
    flush=True,
)
