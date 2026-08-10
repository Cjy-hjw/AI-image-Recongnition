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

# 训推平台回调基址。
#
# 规范散文段落只写 "{训推平台服务地址}/api/competition/inference/callback/"(无主机),
# 但**同一份文档的官方推理服务代码示例给出了实际地址**(赛道七 V1.0 与 V2.0 均有,
# 该章节两版逐字相同):
#     CALLBACK_URL = "http://10.1.13.241:80/api/competition/inference/callback/"
# 该 IP 与容器 SSH 主机(`ssh root@10.1.13.241 -p 10638`)一致,即平台内网网关。
#
# 仍保留环境变量覆盖:① 平台若按队伍分配不同网关,注入即可改;② 本地自测设为空串
# 跳过回调。**默认值不设空串**——留空会让"忘了配环境变量"这种失误静默退化成
# "不回调",而平台等不到回调就一直挂到评测超时,比连不上报错更难排查。
_DEFAULT_CALLBACK_BASE = "http://10.1.13.241:80"
CALLBACK_BASE_URL = os.environ.get(
    "AICOMP_CALLBACK_BASE_URL", _DEFAULT_CALLBACK_BASE).rstrip("/")
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


def pred_path(evaluation_id) -> str:
    """回调 body 里 `predPath` 的值 —— **绝对路径**。

    ## 为什么是绝对路径(2026-08-09 修正,此前实现是错的)

    赛道七规范的 body 示例写 `"predPath": "canswer/454899866564564"`,前缀 `canswer`
    与 §二.2 定义的目录 `/2026aicompetition/workspace/answer/{evaluation_id}` 对不上,
    我方当时判为笔误(多打一个 c)、按目录定义取**相对**路径 `answer/{id}`。

    **同一份文档的官方推理服务代码示例才是权威写法**(V1.0/V2.0 均有,逐字相同):

        payload = {
            "request_id": request_id,
            "evaluationId": evaluation_id,
            "predPath": f"/2026aicompetition/workspace/answer/{evaluation_id}",
        }

    —— 是**完整绝对路径**,既不是 `canswer/...` 也不是相对的 `answer/...`。散文段落
    "predPath输出路径格式为 /2026aicompetition/workspace/answer/{evaluation_id}/{结果集}"
    与代码示例一致,三者中只有 body 示例的 `canswer/...` 是孤例,可判定为笔误。

    平台拿这个值去定位结果目录,路径形态不对就是**找不到结果 ⇒ 该次评测零分**,
    且回调本身返回 200、我方日志一切正常,是典型的静默失败。

    随 `ANSWER_ROOT` 走(而非硬编码字符串),使 `AICOMP_ANSWER_ROOT` 覆盖时
    predPath 自动跟随,不会出现"结果写在 A、却告诉平台去 B 找"的错位。
    """
    return answer_dir(evaluation_id).replace(os.sep, "/")


# 向后兼容别名。旧名 `relative_pred_path` 的语义(相对路径)已被证伪,保留仅为
# 不破坏既有调用点;新代码一律用 `pred_path`。
relative_pred_path = pred_path
