"""Gate2 审查通过后执行，标记指定文件已过 Gate2 审查。

用法: python .claude/hooks/mark_gate2_pass.py [--review-mode self|deep] <file1> <file2> ...

写入 .claude/.gate2_pass.json：{"timestamp": <now>, "files": [...], "review_mode": "self"|"deep"}
gate2_enforce.py（PreToolUse hook）在 git commit 前读此文件校验。
标记 1 小时后过期，防止用旧标记混过新一轮改动。

--review-mode（2026-07-12 新增）：
  self（默认）— 主会话自审，沿用旧行为。
  deep — 真走了 `/code-review` skill + reviewer 子agent。
  gate2_paths.CRITICAL_INFRA_EXACT_FILES（entrypoint.sh/Dockerfile/_src_tree_hash.sh）
  这类改错即整体交付失败的文件，gate2_enforce.py 强制要求 deep，self 不放行。

同时把这些文件从 .claude/.gate2_pending.json（gate2_reminder.py 维护的待审
清单）里移除——审查通过后不该再算"待审"，否则 gate2_escalate_block.py 会
继续误挡后续改动。
"""
import argparse, json, sys, time, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate2_paths import PASS_PATH, read_pending, to_repo_relative, write_pending

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--review-mode", choices=["self", "deep"], default="self")
parser.add_argument("files", nargs="+")
args = parser.parse_args()

# 无论调用方传相对还是绝对路径，统一转仓库相对路径再存盘——与 gate2_enforce.py
# 用 git diff（相对路径）的比对才不会错位，见 gate2_paths.py 顶部注释。
files = [to_repo_relative(f) for f in args.files]
os.makedirs(os.path.dirname(PASS_PATH), exist_ok=True)

with open(PASS_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {"timestamp": time.time(), "files": files, "review_mode": args.review_mode},
        f, ensure_ascii=False, indent=2,
    )

pending = read_pending()
if pending:
    remaining = [p for p in pending if p not in files]
    write_pending(remaining)

print(f"[GATE2-MARK] 已标记 {len(files)} 个文件通过 Gate2（review_mode={args.review_mode}，1小时内有效）：")
for fp in files:
    print(f"  {fp}")
