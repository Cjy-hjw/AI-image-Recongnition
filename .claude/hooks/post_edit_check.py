"""PostToolUse Hook: Python 文件自动语法检查 + 行数告警。

触发条件: Write 或 Edit 工具执行后。
作用:
  1. 对 .py 文件运行 py_compile（语法检查），错误时注入提示。
  2. 超300行时输出告警（rules.md §6），提示考虑拆分模块。
  3. 正常时静默（不输出噪声，仅出问题才提示）。

本文件仅对当前项目生效（.claude/settings.json 配置）。
"""
import json
import os
import subprocess
import sys


def main():
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_input = data.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    # 只处理 .py 文件
    if not file_path or not file_path.endswith(".py"):
        sys.exit(0)

    # 文件是否实际存在（Write 可能刚创建，Edit 应该存在）
    if not os.path.isfile(file_path):
        sys.exit(0)

    # 1. 语法检查
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", file_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = result.stderr.strip()
        print(f"[Hook] ⚠️ 语法错误 → {file_path}\n{err}", flush=True)
        sys.exit(0)  # 非0 exit会被 Claude 感知为错误，此处仍exit 0让AI自行决定

    # 2. 行数检查（仅语法正确时才有意义）
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            n = sum(1 for _ in f)
        fname = os.path.basename(file_path)
        if n > 300:
            print(
                f"[Hook] ⚠️ {fname} 已 {n} 行（rules.md §6 上限 300 行），"
                "若包含 3+ 功能域请考虑拆分模块。",
                flush=True,
            )
        # 正常情况静默（不输出，避免噪声）
    except OSError:
        pass


if __name__ == "__main__":
    main()
