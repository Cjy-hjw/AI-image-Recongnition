"""FastAPI 推理服务(规范 §二 推理服务 API / §三 健康检查)。

## 官方硬约束与本实现的对应

| 规范要求 | 实现 |
|:--|:--|
| `POST /call`,5 秒内返回 200 | `call()` 只做入参解析 + 丢线程池,不等推理 |
| 异步响应机制 | `tasks.submit_inference` 走 ThreadPoolExecutor |
| `GET /health` 返 200,端口 8000 | `health()`;端口见 `config.SERVICE_PORT` |
| 回调回传 request_id + evaluationId + predPath | `tasks.post_callback`(worker 线程内,含重试) |
| 前台长进程 | `python -m src.service.app` → `uvicorn.run` 阻塞前台 |

## 入参容错为什么放得这么宽

规范只列出"入参关键字段"(`request_id` / `input.evaluation_id` / `input.dataset_path`),
没有给出完整 JSON Schema,也没说明字段是否可能出现在顶层。若严格按嵌套结构解析,
平台实际下发的报文只要有一处不同就会 422 —— 而**5 秒内没返 200 即判请求异常**,
且只有一次提交机会,没有试错空间。故 `_extract_fields` 同时接受嵌套与顶层两种位置,
并对 `request_id` 缺失的情形自动补一个 UUID(宁可回调时带一个我方生成的 id,也不要
整个请求 422 掉)。

真正不可缺的只有 `evaluation_id` —— 它决定结果写到哪个目录,缺了无处可写,只能 400。
"""

import logging
import os
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.service import config, tasks

logger = logging.getLogger(__name__)

app = FastAPI(title="Thyroid US Inference Service", version="1.0")

# 实际推理入口。默认 None = 骨架模式(只建目录不推理),便于服务层与模型改造并行推进;
# 由 `set_runner()` 在启动时注入真实实现。
_runner = None


def set_runner(fn):
    """注入推理实现 `callable(dataset_path, output_dir) -> None`。"""
    global _runner
    _runner = fn


def _placeholder_runner(dataset_path, output_dir):
    """骨架模式占位。**不产出任何预测**,只在结果目录留一份说明。

    存在的意义:让 `/call` → 线程池 → 目录创建 → 回调 这条链路能端到端自测,
    不必等模型侧多结节改造完成。**上线前必须由 `set_runner` 覆盖** —— 否则该次
    评测全部输出为空,等同零分。
    """
    logger.warning(
        "[runner] 未注入真实推理实现,当前为骨架模式(dataset_path=%s) —— 上线前必须调用 set_runner()",
        dataset_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "_placeholder.txt"), "w", encoding="utf-8") as f:
        f.write(
            "推理实现未注入(骨架模式)。此文件表示服务链路正常但无预测产出。\n"
            f"dataset_path={dataset_path}\n")


def _extract_fields(payload):
    """从请求体取三个关键字段,兼容嵌套 `input.*` 与顶层两种位置。

    Returns:
        `(request_id, evaluation_id, dataset_path)`;`evaluation_id` 为 None 表示
        无法定位结果目录,调用方应返回 400。
    """
    if not isinstance(payload, dict):
        return None, None, None

    inp = payload.get("input")
    if not isinstance(inp, dict):
        inp = {}

    def pick(*names):
        for name in names:
            if inp.get(name) is not None:
                return inp[name]
            if payload.get(name) is not None:
                return payload[name]
        return None

    request_id = pick("request_id", "requestId")
    evaluation_id = pick("evaluation_id", "evaluationId")
    dataset_path = pick("dataset_path", "datasetPath")
    return request_id, evaluation_id, dataset_path


@app.get("/health")
def health():
    """平台探针端点(启动探针 5s/次最长 1800s;探活 30s/次连续失败 3 次即重启容器)。

    **必须保持极轻**:不查模型、不读磁盘、不加锁等待推理。任何可能阻塞的检查都可能
    让探活连续超时,进而触发容器重启、整场评测作废。任务计数取自内存 dict 的浅拷贝,
    锁竞争可忽略。
    """
    return {"status": "ok", "tasks": tasks.task_summary()}


@app.post("/call")
async def call(request: Request):
    """接收评测任务,**立即**返回 200(规范:5 秒内未响应判定请求异常)。

    这里不做任何耗时操作 —— 连数据集路径是否存在都不检查(网络文件系统 stat 可能
    挂起数秒)。合法性交由 worker 线程判断,失败写 `_error.json` 并照常回调。
    """
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 — 非法 JSON 也要给出明确 400 而非 500
        return JSONResponse(
            status_code=400, content={"code": 400, "message": "invalid JSON body"})

    request_id, evaluation_id, dataset_path = _extract_fields(payload)

    if evaluation_id is None:
        # 唯一的硬性拒绝:没有 evaluation_id 就无处写结果,也无法回调闭环。
        logger.error("[/call] 缺 evaluation_id,payload keys=%s", list(payload)[:10])
        return JSONResponse(
            status_code=400,
            content={"code": 400, "message": "missing input.evaluation_id"})

    if request_id is None:
        # 宁可用我方生成的 id 回调,也不要因缺字段 422 掉整个请求。
        request_id = str(uuid.uuid4())
        logger.warning("[/call] 缺 request_id,自动生成 %s", request_id)

    tasks.submit_inference(
        request_id=request_id,
        evaluation_id=evaluation_id,
        dataset_path=dataset_path,
        runner=_runner or _placeholder_runner,
    )

    logger.info(
        "[/call] accepted request_id=%s evaluation_id=%s dataset_path=%s",
        request_id, evaluation_id, dataset_path)
    return {
        "code": 200,
        "message": "accepted",
        "request_id": request_id,
        "evaluationId": evaluation_id,
    }


@app.get("/task/{request_id}")
def task_status(request_id: str):
    """排障用:查单个任务状态。非规范要求,不参与评测闭环。"""
    entry = tasks.get_task_status(request_id)
    if entry is None:
        return JSONResponse(status_code=404, content={"message": "unknown request_id"})
    return entry


def main():
    """前台长进程入口(规范 §二.3:启动命令必须是长期运行的前台进程,否则服务被关闭)。"""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # 注入真实推理 runner(替换骨架模式 `_placeholder_runner`)。V2.1 生产入口必须
    # 产出逐病例 prediction.json,占位模式会让整次评测零分。模型构建是**惰性**的
    # (首次 /call 的 runner 真正跑时才 build_model),故此处 build_service_runner
    # 不会拖慢 8000 端口探针的就绪时间 —— 平台启动探针对启动期崩溃零容忍。
    from src.service.runner import build_service_runner
    set_runner(build_service_runner())

    logger.info(
        "启动推理服务 %s:%s | answer_root=%s | callback=%s",
        config.SERVICE_HOST, config.SERVICE_PORT, config.ANSWER_ROOT,
        config.callback_url() or "(未配置)")
    uvicorn.run(app, host=config.SERVICE_HOST, port=config.SERVICE_PORT, log_level="info")


if __name__ == "__main__":
    main()
