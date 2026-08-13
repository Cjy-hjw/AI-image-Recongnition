"""推理任务的后台执行与平台回调(规范 §二.1 异步响应 / §二.2 回调闭环)。

## 为什么必须用线程池而不是 asyncio.create_task

平台探活探针每 30 秒打一次 `/health`,**连续失败 3 次就重启测评容器**(规范 §三)。
torch 推理是同步阻塞的 CPU/GPU 代码 —— 若用 `asyncio.create_task` 直接在事件循环里
跑,单张图的推理就会把整个 event loop 堵住,`/health` 无法响应,90 秒后容器被平台
强制重启,**整场评测的进度全部作废且不可恢复**。

故推理一律丢进 `ThreadPoolExecutor`(worker=1,见 `config.INFERENCE_WORKERS`),
event loop 只负责收请求和回调,永远不阻塞。worker 应为 1 的理由:显存虽为官方
实测 32G(2026-08-12 nvidia-smi,控制台"14"是 vCPU 数),"单 GPU/容器 + 模型非线程安全"
仍使并发推理有 OOM/争用风险 —— 排队串行是唯一安全解。⚠️ 实际
`config.INFERENCE_WORKERS` 可由环境变量 `AICOMP_INFERENCE_WORKERS` 覆盖,运维注入 >1 即
破坏此串行/防 OOM 前提——交付容器勿设该变量。
"""

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from src.service import config

logger = logging.getLogger(__name__)

# 单例线程池。模块级而非每请求新建:线程池创建有成本,且 worker=1 的排队语义要求
# 所有任务共用同一个池(各自新建等于并发,直接违背串行前提)。
_executor = ThreadPoolExecutor(
    max_workers=config.INFERENCE_WORKERS, thread_name_prefix="infer")

# 任务状态表,仅供 /health 与排障观察,不参与评测闭环。加锁是因为 event loop 线程
# (写入 PENDING)与 worker 线程(写入 RUNNING/DONE)会并发访问。
_tasks = {}
_tasks_lock = threading.Lock()

STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_DONE = "DONE"
STATUS_FAILED = "FAILED"


def _set_status(request_id, status, **extra):
    with _tasks_lock:
        entry = _tasks.setdefault(request_id, {})
        entry["status"] = status
        entry["updated_at"] = time.time()
        entry.update(extra)


def get_task_status(request_id):
    with _tasks_lock:
        entry = _tasks.get(request_id)
        return dict(entry) if entry else None


def task_summary():
    """各状态的任务计数,供 /health 暴露(便于平台侧与人工判断服务是否卡死)。

    **必须保持极快且不可阻塞**:平台探活探针每 30 秒打一次 `/health`,连续失败 3 次
    即重启容器。若此处持锁时间长于锁的持有时长,探针会撞上锁竞争而超时。`_set_status`
    与 worker 线程都持有 `_tasks_lock`,但临界区只有 dict 赋值与浅拷贝(微秒级),
    正常场景无风险;仅当 `shutdown(wait=True)` 被错误地放进探针路径时才会死锁 ——
    本函数**绝不**触碰 `_executor`,只读 `_tasks` 内存快照。
    """
    with _tasks_lock:
        counts = {}
        for entry in _tasks.values():
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        return {"total": len(_tasks), "by_status": counts}


def post_callback(request_id, evaluation_id, pred_path=None):
    """回传推理完成信号(规范 §二.2)。

    规范原文:"选手必须保存 /call 接收到的 request_id 和 evaluation_id,并在回调时
    回传,否则评测无法闭环,校验将不通过" —— 这两个字段是评测闭环的唯一凭据。

    Returns:
        bool — 是否成功送达。未配置回调基址时返回 False 并记日志(本地自测场景),
        **不抛异常**:回调失败不该反过来毁掉已经算好的结果文件。
    """
    url = config.callback_url()
    if not url:
        logger.warning(
            "[callback] AICOMP_CALLBACK_BASE_URL 未配置,跳过回调 "
            "(request_id=%s evaluation_id=%s)。**上线前必须由平台注入该环境变量**,"
            "否则评测无法闭环。", request_id, evaluation_id)
        return False

    body = {
        "request_id": request_id,
        "evaluationId": evaluation_id,
        "predPath": pred_path or config.relative_pred_path(evaluation_id),
    }

    try:
        import requests
    except ImportError:
        logger.error("[callback] requests 未安装,无法回调 —— 评测无法闭环")
        return False

    for attempt in range(1, config.CALLBACK_MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=body, timeout=config.CALLBACK_TIMEOUT_S)
            if 200 <= resp.status_code < 300:
                logger.info("[callback] 成功 (attempt=%d) %s", attempt, body)
                return True
            logger.warning(
                "[callback] HTTP %s (attempt=%d/%d) body=%s",
                resp.status_code, attempt, config.CALLBACK_MAX_RETRIES, body)
        except Exception as e:  # noqa: BLE001 — 网络异常一律重试,不区分类型
            logger.warning(
                "[callback] %s: %s (attempt=%d/%d)",
                type(e).__name__, e, attempt, config.CALLBACK_MAX_RETRIES)

        if attempt < config.CALLBACK_MAX_RETRIES:
            time.sleep(2 ** attempt)  # 指数退避 2/4/8s,避免瞬时重试打爆平台

    logger.error("[callback] 重试耗尽,评测可能无法闭环: %s", body)
    return False


def _write_task_error(evaluation_id, request_id, message):
    """推理整体失败时,在结果目录留一份 `_error.json`。

    离线容器"按提交产出 JSON 是硬契约"(见 predict.py 的既有注释):裸崩溃等于该次
    评测零输出。此处同理 —— 即便推理全挂,也要留下可诊断的痕迹,而非空目录。
    """
    try:
        out_dir = config.answer_dir(evaluation_id)
        os.makedirs(out_dir, exist_ok=True)
        payload = {
            "status": "ERROR",
            "request_id": request_id,
            "evaluation_id": evaluation_id,
            "reason": message,
        }
        tmp = os.path.join(out_dir, f"_error.json.tmp.{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, os.path.join(out_dir, "_error.json"))
    except Exception:  # noqa: BLE001 — 兜底写盘失败不得再抛,否则掩盖原始错误
        logger.exception("[task] 写 _error.json 失败 (evaluation_id=%s)", evaluation_id)


def submit_inference(request_id, evaluation_id, dataset_path, runner):
    """把一次评测任务丢进线程池,立即返回(调用方据此在 5 秒内回 200)。

    Args:
        request_id: `/call` 入参,回调时必须原样回传。
        evaluation_id: `/call` 入参 `input.evaluation_id`,同上。
        dataset_path: `/call` 入参 `input.dataset_path`,测试数据集路径。
        runner: `callable(dataset_path, output_dir) -> None`,实际推理入口。
            以参数注入而非直接 import:① 测试可替换成假 runner,不必加载模型;
            ② 推理主流程(多结节改造)与服务层解耦,两边可独立演进。

    Returns:
        `concurrent.futures.Future` — 仅供测试同步等待,生产路径不读它。
    """
    _set_status(request_id, STATUS_PENDING,
                evaluation_id=evaluation_id, dataset_path=dataset_path)

    def _run():
        _set_status(request_id, STATUS_RUNNING)
        out_dir = config.answer_dir(evaluation_id)
        try:
            # {evaluation_id} 目录"必须由参赛者代码在推理时动态创建"(规范 §二.2)
            os.makedirs(out_dir, exist_ok=True)
            runner(dataset_path, out_dir)
            _set_status(request_id, STATUS_DONE)
        except Exception as e:  # noqa: BLE001 — worker 线程的异常无人接管,必须自兜
            logger.exception("[task] 推理失败 request_id=%s", request_id)
            # 先写 _error.json 落盘、再翻 FAILED:保证"观察到 FAILED"即蕴含"文件已存在",
            # 消除消费方(平台/人工/测试 _wait_status)在两行之间读到"FAILED 但结果目录空"
            # 的竞态(与"还没跑"无法区分)。写盘失败被 _write_task_error 内部吞掉,不阻断状态
            # 翻转;终态集合(FAILED + 文件)与交换前一致,纯顺序调整无逻辑变更。
            _write_task_error(evaluation_id, request_id, f"{type(e).__name__}: {e}")
            _set_status(request_id, STATUS_FAILED, error=f"{type(e).__name__}: {e}")
        finally:
            # **无论成败都回调**:平台靠回调判定该任务已终结。失败时不回调会让平台
            # 一直等到评测超时,比回调一个"结果目录里只有 _error.json"更糟。
            post_callback(request_id, evaluation_id)

    return _executor.submit(_run)


def shutdown(wait=True):
    """关停线程池(测试清理与优雅退出用)。"""
    _executor.shutdown(wait=wait)
