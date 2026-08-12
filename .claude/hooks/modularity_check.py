"""PostToolUse Hook: .py 文件超过 300 行 → 提醒拆分（rules.md §6 模块化红线）。

触发：Edit/Write 后，文件是 src/ 下的 .py（脚本/测试目录行数上限较宽松，不检查）。
效果：只提醒，不阻断——是否该拆、怎么拆需要人工判断（单一职责边界不是行数
能自动画出来的），阻断式检查在这里容易产生"为了不超行数而硬拆"的反效果。
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate2_paths import to_repo_relative

LINE_LIMIT = 300

try:
    data = json.load(sys.stdin)
except (json.JSONDecodeError, EOFError):
    sys.exit(0)

file_path = data.get("tool_input", {}).get("file_path", "")
if not file_path or not file_path.endswith(".py"):
    sys.exit(0)

# tool_input.file_path 是绝对路径，原来直接 normpath 后 startswith("src"+sep)
# 对绝对路径恒为 False——这条 hook 从未真正触发过（2026-07-10 Gate2 审查发现）。
# 转仓库相对路径后再判断前缀。
norm = to_repo_relative(file_path)
if not norm.startswith("src" + os.sep):
    sys.exit(0)
if not os.path.isfile(file_path):
    sys.exit(0)

try:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        n_lines = sum(1 for _ in f)
except OSError:
    sys.exit(0)

if n_lines > LINE_LIMIT:
    print(
        f"[MODULARITY] {os.path.basename(file_path)} 已达 {n_lines} 行"
        f"（rules.md §6 提示阈值 {LINE_LIMIT} 行）。\n"
        "🔴 行数只是**提示信号**，不是拆分判据 —— 判据是**职责**（是否承担 3+ 个功能域）。\n"
        "拆之前先过 §6 前置三问，任一为「是」则**不拆**、在模块 docstring 里记录例外与理由：\n"
        "  ① 拆分会降低可分发性/可执行性吗？（多文件撞云上传限额、多一层 import 失败点）\n"
        "  ② 拆出来的部分有独立的认知边界吗？没有 = 把同一件事切两半，零收益。\n"
        "  ③ 需要为守行数压缩/删除注释与实测细节吗？—— 违反 §5，绝对禁止。\n"
        "确实是 3+ 功能域且三问全否，才按 losses/components 已有模式再拆一层。",
        flush=True,
    )
