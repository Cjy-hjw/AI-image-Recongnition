"""PreToolUse Hook: 高风险改动堆积过多时拦截新的高风险 Edit/Write（阻断式）。

触发：Edit/Write 前，目标文件命中 gate2_paths.is_high_risk。
逻辑：
  1. 读 `.claude/.gate2_pending.json`（gate2_reminder.py 维护的待审清单）。
  2. 若待审文件数已达阈值，且本次要改的文件是清单外的**新**高风险文件
     → exit 2 阻断，要求先做 Gate2 审查 + mark_gate2_pass.py 清账。
  3. 待审清单内已有的文件（迭代同一批改动、按审查意见修正）不拦截——
     只挡"在还没审查的旧账上继续堆新账"，不挡"修正已知在案的文件"。
  4. 未超阈值，或目标文件已在清单内 → exit 0 放行。

目的：防止"改了好几个高风险文件才一次性批量审查"（2026-07-10 教训）——
逼迫审查节奏跟上改动节奏，而不是全部堆到最后才做一次 Gate2。
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate2_paths import ESCALATE_THRESHOLD, is_high_risk, read_pending, to_repo_relative

try:
    data = json.load(sys.stdin)
except (json.JSONDecodeError, EOFError):
    sys.exit(0)

file_path = data.get("tool_input", {}).get("file_path", "")
if not file_path or not is_high_risk(file_path):
    sys.exit(0)

pending = read_pending()
if not pending:
    sys.exit(0)

norm = to_repo_relative(file_path)
if len(pending) < ESCALATE_THRESHOLD or norm in pending:
    sys.exit(0)

quoted = " ".join(f'"{p}"' for p in pending)
print(
    f"[GATE2-BLOCK] 已有 {len(pending)} 个高风险文件未经 Gate2 审查："
    f"{', '.join(os.path.basename(p) for p in pending)}\n"
    f"在清账前不能再改动新的高风险文件（`{os.path.basename(file_path)}`）。\n"
    "请先派 reviewer 子agent 完成 Gate2 审查，再执行：\n"
    f"  python .claude/hooks/mark_gate2_pass.py {quoted}",
    file=sys.stderr,
)
sys.exit(2)
