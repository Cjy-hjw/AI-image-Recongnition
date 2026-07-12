"""PreToolUse Hook: git commit 前强制检查 Gate2 审查标记（阻断式,非提醒）。

触发：Bash 工具执行前，命令匹配 `git commit`。
逻辑：
  1. 取 `git diff --cached --name-only` 中已暂存的文件列表。
  2. 若其中含高风险路径(见 gate2_paths.is_high_risk，覆盖 src/models/、
     src/training/、src/inference/ 及 scripts/ 下 train_*/pretrain_*/merge_lora*/
     retrain_* 命名的脚本)，检查 `.claude/.gate2_pass.json` 是否存在、未过期
     (1小时内)、且覆盖了本次暂存的所有高风险文件。
  3. 不满足 → exit 2 阻断 commit，stderr 给出修复指引(Claude Code 会
     把 exit 2 的 stderr 喂回模型上下文，逼迫先做 Gate2)。
  4. 满足 → exit 0 放行，并从 .gate2_pending.json 里移除本次已覆盖的文件
     （只清覆盖部分，不影响其余尚未 add、仍待审查的高风险文件的计数）。

标记文件写入方式(Gate2 审查完成后手动执行)：
    python .claude/hooks/mark_gate2_pass.py <file1> <file2> ...

2026-07-12 新增：staged 里若含 gate2_paths.CRITICAL_INFRA_EXACT_FILES（容器
唯一入口/构建定义这类改错即整体失败的文件），marker 的 review_mode 必须是
"deep"，"self"（自审）不放行——见 gate2_paths.py 顶部该常量的教训记录。
"""
import json, os, subprocess, sys, re, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate2_paths import PASS_PATH, is_high_risk, read_pending, requires_deep_review, to_repo_relative, write_pending

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

high_risk_staged = [f for f in staged if is_high_risk(f)]
if not high_risk_staged:
    sys.exit(0)

# ── 检查 Gate2 标记 ──
if not os.path.exists(PASS_PATH):
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
    with open(PASS_PATH, "r", encoding="utf-8") as f:
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

# staged 已是仓库相对路径（git diff 输出），marked_files 统一转相对路径再比对——
# 二者格式不一致正是本次 Gate2 审查揪出的核心 bug（详见 gate2_paths.py 顶部注释）。
marked_files = {to_repo_relative(f) for f in marker.get("files", [])}
missing = [f for f in high_risk_staged if to_repo_relative(f) not in marked_files]
if missing:
    print(
        "[GATE2-BLOCK] 以下高风险文件未被 Gate2 标记覆盖：\n  "
        + "\n  ".join(missing)
        + "\n请审查后执行: python .claude/hooks/mark_gate2_pass.py " + " ".join(missing),
        file=sys.stderr,
    )
    sys.exit(2)

# 关键基础设施文件（entrypoint.sh/Dockerfile/_src_tree_hash.sh）不接受 self
# 审查——必须真走 `/code-review` skill + reviewer 子agent 后用 --review-mode
# deep 标记。
critical_staged = [f for f in high_risk_staged if requires_deep_review(f)]
if critical_staged and marker.get("review_mode") != "deep":
    print(
        "[GATE2-BLOCK] 以下文件是容器唯一入口/构建定义类关键基础设施，不接受"
        "\"self\"自审，必须真走 /code-review skill + reviewer 子agent 后标记"
        " --review-mode deep：\n  "
        + "\n  ".join(critical_staged)
        + "\n审查通过后执行: python .claude/hooks/mark_gate2_pass.py --review-mode deep "
        + " ".join(critical_staged),
        file=sys.stderr,
    )
    sys.exit(2)

# 本次 commit 已合规通过 Gate2：只清掉被本次 staged 覆盖的待审记录，不动其余
# 尚未 add 的高风险文件（2026-07-10 审查发现：旧版无条件删除整个 pending.json，
# 会把未提交、仍待审查的其它文件的记录一并静默清空，绕开 gate2_escalate_block.py
# 的堆积拦截）。
remaining_pending = [p for p in read_pending() if to_repo_relative(p) not in marked_files]
write_pending(remaining_pending)

sys.exit(0)
