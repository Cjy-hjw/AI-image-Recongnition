"""PostToolUse Hook: git push 后提醒核对 CI headSha（rules.md §2 "CI 只验证已 push 的 commit"）。

触发：Bash 工具执行后，命令匹配 `git push`。
效果：提醒用 `gh run list --branch <branch> --limit 1` 核对 headSha 与本轮
commit 一致、conclusion=success——只能提醒去核对，headSha 是否真的匹配需要
读取 gh 输出做判断，这一步没法被 hook 机械替代。不区分 push 是否成功也提醒
（push 失败时这条提醒只是无害冗余，不值得为此依赖不确定的 tool_response 字段名）。
"""
import json, re, sys

try:
    data = json.load(sys.stdin)
except (json.JSONDecodeError, EOFError):
    sys.exit(0)

cmd = data.get("tool_input", {}).get("command", "")
if not cmd or not re.search(r"\bgit\s+push\b", cmd):
    sys.exit(0)

print(
    "[CI-REMINDER] 已尝试 push。声称「CI 通过」前必须核对："
    "`gh run list --branch <当前分支> --limit 1`（走代理），"
    "确认 headSha 正是本轮 commit 且 conclusion=success。"
    "本地 pytest 全绿不能替代 CI 核对。",
    flush=True,
)
