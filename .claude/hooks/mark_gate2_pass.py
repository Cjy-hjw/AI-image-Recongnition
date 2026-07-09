"""Gate2 审查通过后执行，标记指定文件已过 Gate2 审查。

用法: python .claude/hooks/mark_gate2_pass.py <file1> <file2> ...

写入 .claude/.gate2_pass.json：{"timestamp": <now>, "files": [...]}
gate2_enforce.py（PreToolUse hook）在 git commit 前读此文件校验。
标记 1 小时后过期，防止用旧标记混过新一轮改动。
"""
import json, sys, time, os

if len(sys.argv) < 2:
    print("用法: python mark_gate2_pass.py <file1> <file2> ...")
    sys.exit(1)

files = sys.argv[1:]
marker_path = os.path.join(".claude", ".gate2_pass.json")
os.makedirs(os.path.dirname(marker_path), exist_ok=True)

with open(marker_path, "w", encoding="utf-8") as f:
    json.dump({"timestamp": time.time(), "files": files}, f, ensure_ascii=False, indent=2)

print(f"[GATE2-MARK] 已标记 {len(files)} 个文件通过 Gate2（1小时内有效）：")
for fp in files:
    print(f"  {fp}")
