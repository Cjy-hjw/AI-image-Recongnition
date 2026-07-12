"""Gate2 高风险路径判定 + pending/pass 标记文件读写的共享逻辑。

被 gate2_enforce.py / gate2_reminder.py / gate2_escalate_block.py / mark_gate2_pass.py /
modularity_check.py 共同导入。

单一事实源：避免"各 hook 各自维护一份路径判断/路径格式/pending读写逻辑，后续改动
漏同步一处"的 drift。三次真实教训都由此而来：
1. （2026-07-10 早）train_thyroidxl.py 在 scripts/、model_runner.py 在
   src/inference/，两者都不在旧版 HIGH_RISK_PREFIXES 里，提醒 hook 完全没触发。
2. （2026-07-10 晚，Gate2 审查抓出）Edit/Write 工具传入的 `tool_input.file_path`
   是**绝对路径**，而 `git diff --cached --name-only` 吐出的是**仓库相对路径**。
   gate2_reminder.py/gate2_escalate_block.py 曾各自对绝对路径 `os.path.normpath`
   后直接存盘，gate2_enforce.py 却拿仓库相对路径去比对——两者永不相等，导致
   "审查已通过但 commit 仍被 GATE2-BLOCK 永久拦截"。必须在写入 pending/pass 前
   统一转换成仓库相对路径，才能和 git diff 的输出正确比对。
3. （2026-07-11，用户审查发现）`is_high_risk` 曾无条件 `if not file_path.endswith(".py"):
   return False`——任何非 .py 文件永远判定为低风险，与"高风险"的真正含义（改错了
   代价大、影响面广）脱节。当轮 `requirements.txt` 新增依赖版本锁定
   （pylibjpeg 系列，须避开与 numpy<2 冲突的新版本）全程未经 Gate2 审查，
   而这类改动一旦锁错版本会让**整个交付容器 build 失败**，风险不比"高风险
   目录"里的代码改动小，只是恰好不落在旧判定的目录/扩展名范围内被放过。同理
   `Dockerfile`（决定容器能否构建）、`scripts/entrypoint.sh`（容器唯一入口，
   分发+防篡改逻辑）、`scripts/_src_tree_hash.sh`（防篡改指纹算法，须与
   `seal_container.py` 的 Python 实现字节级一致）此前也全部因为非 .py
   扩展名被系统性排除在 Gate2 覆盖之外。改为显式的高风险文件精确匹配清单
   （见 HIGH_RISK_EXACT_FILES），不再用扩展名兜底排除。
"""
import json
import os
import re

# 目录前缀命中即高风险（gradient routing / optimizer binding / loss math 高发区）
HIGH_RISK_DIR_PREFIXES = (
    os.path.join("src", "models"),
    os.path.join("src", "training"),
    os.path.join("src", "inference"),  # QA_VP.md §1.7 推理侧红线：误报/维度契约/降级路径
    os.path.join("src", "gatekeeper"),  # 一票否决安全链路：真实性/去重/篡改检测
    os.path.join("src", "utils"),  # report_generator.py：TI-RADS评分/临床报告生成
    # （2026-07-10 Gate2审查覆盖面扩容：gatekeeper/report_generator此前不在
    # 高风险清单里，本轮就在这两处抓出真实bug——authenticity.py权重缺失静默
    # 降级、tampering_catnet.py forward失败误判、report_generator.py NaN/越界
    # idx静默算错——同类"看似能跑、实则跑歪"风险不该被排除在Gate2覆盖外。
)

# scripts/ 下按文件名模式命中（避免把 verify_model.py/deploy.py 等低风险脚本也算进来）
HIGH_RISK_SCRIPT_PATTERNS = (
    re.compile(r"^train_.*\.py$"),
    re.compile(r"^pretrain_.*\.py$"),
    re.compile(r"^merge_lora.*\.py$"),
    re.compile(r"^retrain_.*\.py$"),
    re.compile(r"^seal_container\.py$"),  # 防篡改密封脚本，src树聚合哈希算法单一事实源
)

# 非 .py 但同样高风险的精确文件路径（2026-07-11 补，修复"仅 .py 才判高风险"的
# 系统性盲区）——依赖/构建/容器入口/防篡改指纹，改错代价是整个交付容器不可用
# 或防篡改契约破坏，不是"看着不像代码就低风险"。仓库相对路径精确匹配。
HIGH_RISK_EXACT_FILES = (
    "requirements.txt",                              # 依赖版本冲突可致容器 build 失败
    "Dockerfile",                                     # 基础镜像/COPY/系统依赖，决定可构建性
    os.path.join("scripts", "entrypoint.sh"),         # 容器唯一入口：分发路由 + 防篡改阻断
    os.path.join("scripts", "_src_tree_hash.sh"),     # 防篡改指纹算法，须与 seal_container.py 字节级一致
)

# 2026-07-12 教训（用户发现）：entrypoint.sh 的崩溃兜底改动曾被划入"主会话
# 自审即可"，跳过完整 `/code-review`+reviewer 子agent 流程；事后补审在真实
# bash 环境实测复现一个信号转发竞态 bug（详见 scripts/AGENTS.md §7.1.1）。
# 结论：容器唯一入口/构建定义这类改错即整体失败的文件，"自审"不够，必须
# `mark_gate2_pass.py --review-mode deep` 显式声明才放行。requirements.txt
# 版本锁定审查方式已是显式核对，暂不纳入。
CRITICAL_INFRA_EXACT_FILES = (
    os.path.join("scripts", "entrypoint.sh"),
    "Dockerfile",
    os.path.join("scripts", "_src_tree_hash.sh"),
)


def requires_deep_review(file_path: str) -> bool:
    """判定文件是否要求 `mark_gate2_pass.py --review-mode deep`（"self" 不够）。"""
    return os.path.normpath(file_path) in CRITICAL_INFRA_EXACT_FILES


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
    """判定单个文件路径是否属于 Gate2 高风险范围。

    2026-07-11 修复：不再无条件排除非 .py 文件——HIGH_RISK_EXACT_FILES 里的
    非 .py 文件（requirements.txt/Dockerfile/entrypoint.sh/_src_tree_hash.sh）
    优先判定，判完才走 .py-only 的目录/脚本名模式判定。
    """
    norm = os.path.normpath(file_path)

    if norm in HIGH_RISK_EXACT_FILES:
        return True

    if not file_path.endswith(".py"):
        return False

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
