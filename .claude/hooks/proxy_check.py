"""PreToolUse Hook: git push/fetch/gh 命令前检查代理是否已设置（提醒式，非阻断）。

触发：Bash 工具执行前，命令**文本**匹配 git push/fetch/clone 或 gh。
效果：未设代理时输出警告，提醒加上代理环境变量，防止卡住或超时。

**2026-07-10 曾短暂升级为阻断式（exit 2），实测立刻自证反例**：正则只能匹配
命令**字符串**里的子串，分不清是真的在调用 `git push` 还是这个词组只是出现在
一条 echo/注释/测试脚本文本里（本改动的验证过程中，含"echo '...git push...'"
的测试命令被自己拦下）。阻断式下，这类假阳性代价从"多余提醒"变成"合法命令
被拦下"，得不偿失，遂回退为提醒式——想做真阻断，需要真正解析 shell 语法
判断这是不是一条即将执行的命令（而非引号/echo内的文本），目前不值得为此增加
复杂度。
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
        "[PROXY-WARN] 命令文本含外网访问相关字样但未检测到 HTTP_PROXY。\n"
        "rules.md §5 要求走代理，否则会卡住/超时。若命令确实要访问外网，请加前缀：\n"
        "HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897\n"
        "PowerShell 请先: $env:http_proxy='http://127.0.0.1:7897'\n"
        "（若这是误报——命令里只是文本提到 git push，不是真的在调用——可忽略）",
        flush=True,
    )
