"""PostToolUse Hook: 高风险文件改动 → 记录待审清单 + 注入 Gate2 审查提醒。

触发：Edit/Write 后，文件命中 gate2_paths.is_high_risk（覆盖 src/models/、
src/training/、src/inference/、scripts/ 下 train_*/pretrain_*/merge_lora*/
retrain_* 脚本）。

效果：
  1. 把该文件加入 `.claude/.gate2_pending.json` 待审清单（跨工具调用持久化，
     直到 mark_gate2_pass.py 标记或 git commit 成功才清空）。
  2. 输出提醒文字到 Claude 上下文，并报告当前待审文件数——数量越多提醒越强硬。
  3. 待审文件数达到阈值时，后续对高风险文件的 Edit/Write 会被
     gate2_escalate_block.py（PreToolUse）直接拦截，逼迫先做 Gate2 而不能无限
     堆积"改完再说"的欠账（2026-07-10 教训：一次性改了4个高风险文件才做一次
     批量审查，中间全靠自觉）。
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate2_paths import ESCALATE_THRESHOLD, is_high_risk, read_pending, to_repo_relative, write_pending

try:
    data = json.load(sys.stdin)
except (json.JSONDecodeError, EOFError):
    sys.exit(0)

file_path = data.get("tool_input", {}).get("file_path", "")
if not file_path or not is_high_risk(file_path):
    sys.exit(0)

# 转仓库相对路径再存盘——否则和 gate2_enforce.py 用 git diff（相对路径）的比对
# 永远错位，见 gate2_paths.py 顶部注释。
norm = to_repo_relative(file_path)

pending = read_pending()
if norm not in pending:
    pending.append(norm)
    write_pending(pending)

fname = os.path.basename(file_path)
n = len(pending)
# 建议命令逐个加引号：仓库路径本身含空格（"AI image Recongnition"），裸拼接
# 会被 shell 拆成错误的位置参数。
quoted = " ".join(f'"{p}"' for p in pending)
if n >= ESCALATE_THRESHOLD:
    print(
        f"[GATE2-REMINDER] 修改了高风险文件 `{fname}`（累计 {n} 个未审查高风险文件待审！）。\n"
        f"待审清单: {', '.join(os.path.basename(p) for p in pending)}\n"
        "已达阈值——继续改动其他高风险文件前会被拦截，请先完成 Gate2 审查并执行 "
        f"`python .claude/hooks/mark_gate2_pass.py {quoted}`",
        flush=True,
    )
else:
    print(
        f"[GATE2-REMINDER] 修改了高风险文件 `{fname}`（当前 {n}/{ESCALATE_THRESHOLD} 个未审查待审）。\n"
        "rules.md §2 要求: 涉及梯度路由/优化器绑定/损失数学管道的改动须派 reviewer 子agent (Gate2)。\n"
        "若本次改动属于以上范畴，建议尽快完成 Gate2 审查，不要无限堆积待审文件。",
        flush=True,
    )
