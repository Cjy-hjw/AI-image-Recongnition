"""PostToolUse Hook: .py 文件改动后语法检查 + 提示对应测试文件。

触发：Edit/Write 后，文件是 .py。
效果：
  1. py_compile 语法检查（出错时报告）。
  2. 自动定位对应的 tests/test_xxx.py，提示可运行的精准测试命令。
     不自动跑全量 tests（41s 太慢），只提示精准命令让 Claude 按需决定。
"""
import json, os, subprocess, sys

try:
    data = json.load(sys.stdin)
except (json.JSONDecodeError, EOFError):
    sys.exit(0)

file_path = data.get("tool_input", {}).get("file_path", "")
if not file_path or not file_path.endswith(".py"):
    sys.exit(0)
if not os.path.isfile(file_path):
    sys.exit(0)

# 1. 语法检查
result = subprocess.run(
    [sys.executable, "-m", "py_compile", file_path],
    capture_output=True, text=True,
)
if result.returncode != 0:
    print(f"[SYNTAX-ERROR] {os.path.basename(file_path)}\n{result.stderr.strip()}", flush=True)
    sys.exit(0)

# 2. 定位对应测试文件
fname = os.path.basename(file_path)           # e.g. pathology_dataset.py
module = fname.replace(".py", "")             # e.g. pathology_dataset
test_candidates = [
    os.path.join("tests", f"test_{module}.py"),   # test_pathology_dataset.py
]
# 特殊映射：高频文件名 → 对应测试
SPECIAL_MAP = {
    "multi_task_model": "test_phase_refactor.py",
    "cbm_head": "test_cbm_head.py",
    "physical_prior": "test_physical_prior.py",
    "multitask": "test_masked_loss.py",
    "loader": "test_multisource.py",
    "video_io": "test_video.py",
    "video_aggregate": "test_video.py",
    "calibration": "test_calibration.py",
}
if module in SPECIAL_MAP:
    test_candidates.insert(0, os.path.join("tests", SPECIAL_MAP[module]))

found = next((t for t in test_candidates if os.path.isfile(t)), None)
if found:
    print(
        f"[TEST-HINT] {fname} 有对应测试: "
        f"`pytest {found} -v`",
        flush=True,
    )
