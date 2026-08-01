"""推理服务的环境配置与官方路径约定(2026-08-01《赛事开发规范 V1.0》§二/§三)。

集中一处的理由:路径与端口散落在服务层各处时,平台侧改一个挂载点就要全仓 grep。
所有值均可用环境变量覆盖,便于本地测试不碰真实路径。
"""

import os

# 平台挂载的工作区根目录(规范 §二.2 / §三:脚本与产出都应落在此目录下)。
# 生产环境为 Linux 容器(`/2026aicompetition/workspace`);本地 Windows 测试须通过
# `AICOMP_WORKSPACE` 覆盖为平台无关路径,否则 `os.path.join` 会产出混合分隔符
# (`/2026aicompetition/workspace\answer`),`os.makedirs`/`os.replace` 可能失败。
WORKSPACE_ROOT = os.environ.get("AICOMP_WORKSPACE", "/2026aicompetition/workspace")

# 推理结果根目录。**该目录由平台创建**,但其下的 {evaluation_id} 子目录
# "必须由参赛者代码在推理时动态创建"(规范 §二.2 原文)。
ANSWER_ROOT = os.environ.get("AICOMP_ANSWER_ROOT", os.path.join(WORKSPACE_ROOT, "answer"))

# 训练日志目录(规范 §一):平台读不到指定日志文件"可能会被标记为可疑对象"。
LOGS_ROOT = os.environ.get("AICOMP_LOGS_ROOT", os.path.join(WORKSPACE_ROOT, "logs"))

# 服务端口固定 8000(规范 §三:健康检查接口必须在 8000 端口)。
SERVICE_PORT = int(os.environ.get("AICOMP_PORT", "8000"))
SERVICE_HOST = os.environ.get("AICOMP_HOST", "0.0.0.0")

# 训推平台回调基址。规范只写 "{训推平台服务地址}/api/competition/inference/callback/",
# 未给出具体主机 —— 必须由平台通过环境变量注入。**留空则回调被跳过**(仅记日志),
# 这样本地/离线自测不会因为连不上平台而让整条推理链路失败。
CALLBACK_BASE_URL = os.environ.get("AICOMP_CALLBACK_BASE_URL", "").rstrip("/")
CALLBACK_PATH = "/api/competition/inference/callback/"

# 回调重试:平台侧网络抖动不该让一次完整推理的结果丢失。总耗时上界约
# 2+4+8=14s,远小于评测超时,且指数退避避免瞬时重试打爆平台。
CALLBACK_MAX_RETRIES = int(os.environ.get("AICOMP_CALLBACK_RETRIES", "3"))
CALLBACK_TIMEOUT_S = float(os.environ.get("AICOMP_CALLBACK_TIMEOUT", "10"))

# 推理工作线程数。**必须为 1**:单卡 32G 显存(官方答疑 Q33)下并发推理会 OOM,
# 且模型本身非线程安全。串行执行、队列排队是唯一安全选择。
INFERENCE_WORKERS = int(os.environ.get("AICOMP_INFERENCE_WORKERS", "1"))


def callback_url() -> str:
    """完整回调地址;未配置基址时返回空串(调用方据此跳过回调)。"""
    return f"{CALLBACK_BASE_URL}{CALLBACK_PATH}" if CALLBACK_BASE_URL else ""


def answer_dir(evaluation_id) -> str:
    """`{ANSWER_ROOT}/{evaluation_id}` —— 该次评测的结果根目录。"""
    return os.path.join(ANSWER_ROOT, str(evaluation_id))


def relative_pred_path(evaluation_id) -> str:
    """回调 body 里 `predPath` 的值。

    规范示例写的是 `"predPath": "canswer/454899866564564"` —— 前缀 `canswer` 与
    §二.2 定义的目录 `/2026aicompetition/workspace/answer/{evaluation_id}` 对不上,
    高度疑似官方笔误(多打一个 c)。此处按**目录定义**取 `answer/{evaluation_id}`,
    因为平台最终要按这个路径去找文件,拼错则找不到结果。若实测平台按字面 `canswer`
    解析,设环境变量 `AICOMP_PRED_PATH_PREFIX=canswer` 即可切换,无需改代码。
    """
    prefix = os.environ.get("AICOMP_PRED_PATH_PREFIX", "answer")
    return f"{prefix}/{evaluation_id}"
