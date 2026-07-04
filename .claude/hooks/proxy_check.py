"""PreToolUse Hook: git push/fetch/gh 命令前检查代理是否已设置。

触发：Bash 工具执行前，命令匹配 git push/fetch/clone 或 gh。
效果：未设代理时输出警告，提醒加上代理环境变量，防止卡住或超时。
"""
import json, os, sys, re

try:
    data = json.load(sys.stdin)
except (json.JSONDecodeError, EOFError):
    sys.exit(0)

cmd = data.get("tool_input", {}).get("command", "")
if not cmd:
    sys.exit(0)

# 只检查外网访问命令
PATTERNS = [
    r"\bgit\s+push\b",
    r"\bgit\s+fetch\b",
    r"\bgit\s+clone\b",
    r"\bgh\s+",
    r"\bcurl\b.*github",
    r"\bcurl\b.*huggingface",
]
if not any(re.search(p, cmd) for p in PATTERNS):
    sys.exit(0)

proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
if not proxy:
    print(
        "[PROXY-WARN] 访问外网命令但未检测到 HTTP_PROXY。\n"
        "rules.md §5 要求走代理，否则会卡住/超时。\n"
        "Bash 命令请加前缀: HTTP_PROXY=http://127.0.0.1:1203 HTTPS_PROXY=http://127.0.0.1:1203\n"
        "PowerShell 请先: $env:http_proxy='http://127.0.0.1:1203'",
        flush=True,
    )
