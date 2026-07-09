"""PreToolUse Hook: git commit 前强制检查 Gate2 审查标记（阻断式,非提醒）。

触发：Bash 工具执行前，命令匹配 `git commit`。
逻辑：
  1. 取 `git diff --cached --name-only` 中已暂存的文件列表。
  2. 若其中含高风险路径(src/models/、src/training/)，检查
     `.claude/.gate2_pass.json` 是否存在、未过期(1小时内)、且覆盖了
     本次暂存的所有高风险文件。
  3. 不满足 → exit 2 阻断 commit，stderr 给出修复指引(Claude Code 会
     把 exit 2 的 stderr 喂回模型上下文，逼迫先做 Gate2)��
  4. 满足 → exit 0 放行。

标记文件写入方式(Gate2 审查完成后手动执行)：
    python .claude/hooks/mark_gate2_pass.py <file1> <file2> ...
"""
import json, os, subprocess, sys, re, time

HIGH_RISK_PREFIXES = (
    os.path.join("src", "models"),
    os.path.join("src", "training"),
)
MARKER_PATH = os.path.join(".claude", ".gate2_pass.json")
MARKER_TTL_SECONDS = 3600  # 1小时内的标记才有效，防止用旧标记混过新改动

try:
    data = json.load(sys.stdin)
except (json.JSONDecodeError, EOFError):
    sys.exit(0)

cmd = data.get("tool_input", {}).get("command", "")
if not cmd or not re.search(r"\bgit\s+commit\b", cmd):
    sys.exit(0)

try:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True, text=True, timeout=10,
    )
    staged = [f.strip() for f in result.stdout.splitlines() if f.strip()]
except Exception:
    sys.exit(0)  # git 命令本身失败，不是本 hook 的职责，放行

high_risk_staged = [
    f for f in staged
    if f.endswith(".py") and any(os.path.normpath(f).startswith(p) for p in HIGH_RISK_PREFIXES)
]
if not high_risk_staged:
    sys.exit(0)

# ── 检查 Gate2 标记 ──
if not os.path.exists(MARKER_PATH):
    print(
        f"[GATE2-BLOCK] commit 涉及高风险文件但无 Gate2 审查标记：\n  "
        + "\n  ".join(high_risk_staged)
        + "\n\nrules.md §2 强制：涉及梯度路由/优化器绑定/损失数学的改动必须先跑 Gate2 "
        "(reviewer 子agent 审查 QA_VP.md §1/§2 清单)。\n"
        "审查通过后执行: python .claude/hooks/mark_gate2_pass.py " + " ".join(high_risk_staged),
        file=sys.stderr,
    )
    sys.exit(2)

try:
    with open(MARKER_PATH, "r", encoding="utf-8") as f:
        marker = json.load(f)
except (json.JSONDecodeError, OSError):
    print("[GATE2-BLOCK] Gate2 标记文件损坏，请重新审查并标记。", file=sys.stderr)
    sys.exit(2)

age = time.time() - marker.get("timestamp", 0)
if age > MARKER_TTL_SECONDS:
    print(
        f"[GATE2-BLOCK] Gate2 标记已过期({int(age)}s > {MARKER_TTL_SECONDS}s)，"
        "请重新审查并标记。",
        file=sys.stderr,
    )
    sys.exit(2)

marked_files = set(marker.get("files", []))
missing = [f for f in high_risk_staged if f not in marked_files]
if missing:
    print(
        "[GATE2-BLOCK] 以下高风险文件未被 Gate2 标记覆盖：\n  "
        + "\n  ".join(missing)
        + "\n请审查后执行: python .claude/hooks/mark_gate2_pass.py " + " ".join(missing),
        file=sys.stderr,
    )
    sys.exit(2)

sys.exit(0)
