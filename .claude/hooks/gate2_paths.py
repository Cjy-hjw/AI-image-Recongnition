"""Gate2 高风险路径判定 + pending/pass 标记文件读写的共享逻辑。

被 gate2_enforce.py / gate2_reminder.py / gate2_escalate_block.py / mark_gate2_pass.py /
modularity_check.py 共同导入。

单一事实源：避免"各 hook 各自维护一份路径判断/路径格式/pending读写逻辑，后续改动
漏同步一处"的 drift。两次真实教训都由此而来：
1. （2026-07-10 早）train_thyroidxl.py 在 scripts/、model_runner.py 在
   src/inference/，两者都不在旧版 HIGH_RISK_PREFIXES 里，提醒 hook 完全没触发。
2. （2026-07-10 晚，Gate2 审查抓出）Edit/Write 工具传入的 `tool_input.file_path`
   是**绝对路径**，而 `git diff --cached --name-only` 吐出的是**仓库相对路径**。
   gate2_reminder.py/gate2_escalate_block.py 曾各自对绝对路径 `os.path.normpath`
   后直接存盘，gate2_enforce.py 却拿仓库相对路径去比对——两者永不相等，导致
   "审查已通过但 commit 仍被 GATE2-BLOCK 永久拦截"。必须在写入 pending/pass 前
   统一转换成仓库相对路径，才能和 git diff 的输出正确比对。
"""
import json
import os
import re

# 目录前缀命中即高风险（gradient routing / optimizer binding / loss math 高发区）
HIGH_RISK_DIR_PREFIXES = (
    os.path.join("src", "models"),
    os.path.join("src", "training"),
    os.path.join("src", "inference"),  # QA_VP.md §1.7 推理侧红线：误报/维度契约/降级路径
)

# scripts/ 下按文件名模式命中（避免把 verify_model.py/deploy.py 等低风险脚本也算进来）
HIGH_RISK_SCRIPT_PATTERNS = (
    re.compile(r"^train_.*\.py$"),
    re.compile(r"^pretrain_.*\.py$"),
    re.compile(r"^merge_lora.*\.py$"),
    re.compile(r"^retrain_.*\.py$"),
)

# pending/pass 标记文件路径 + 升级阈值：4 个 hook 共用同一份，禁止各自重复字面量
PENDING_PATH = os.path.join(".claude", ".gate2_pending.json")
PASS_PATH = os.path.join(".claude", ".gate2_pass.json")
ESCALATE_THRESHOLD = 3


def to_repo_relative(file_path: str) -> str:
    """把 Edit/Write 的绝对路径 或 git diff 的相对路径统一成仓库相对路径。

    所有 hook 都从仓库根目录运行，故绝对路径用 relpath(cwd) 转换即可；已是相对
    路径的（如 git diff 输出）原样 normpath。这是修复 pending/pass 与
    `git diff --cached --name-only` 比对错位的关键一步——两者必须落在同一格式。
    """
    if os.path.isabs(file_path):
        file_path = os.path.relpath(file_path, os.getcwd())
    return os.path.normpath(file_path)


def is_high_risk(file_path: str) -> bool:
    """判定单个文件路径是否属于 Gate2 高风险范围。"""
    if not file_path.endswith(".py"):
        return False
    norm = os.path.normpath(file_path)

    if any(norm.__contains__(p) for p in HIGH_RISK_DIR_PREFIXES):
        return True

    scripts_prefix = "scripts" + os.sep
    if scripts_prefix in norm:
        fname = os.path.basename(norm)
        if any(pat.match(fname) for pat in HIGH_RISK_SCRIPT_PATTERNS):
            return True

    return False


def read_pending() -> list:
    """读 .gate2_pending.json 的 files 列表；文件不存在/损坏则返回空列表。"""
    if not os.path.exists(PENDING_PATH):
        return []
    try:
        with open(PENDING_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("files", [])
    except (json.JSONDecodeError, OSError):
        return []


def write_pending(files: list) -> None:
    """写 pending 列表；为空则直接删除标记文件（没有待审项目就不该留空文件）。"""
    if not files:
        if os.path.exists(PENDING_PATH):
            try:
                os.remove(PENDING_PATH)
            except OSError:
                pass
        return
    os.makedirs(os.path.dirname(PENDING_PATH), exist_ok=True)
    with open(PENDING_PATH, "w", encoding="utf-8") as f:
        json.dump({"files": files}, f, ensure_ascii=False, indent=2)
