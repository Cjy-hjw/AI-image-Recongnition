# src/service/AGENTS.md — 常驻 HTTP 推理服务(V2.1)

V2.1《赛事开发规范》§二/§三 的 HTTP 交互层。与旧的离线 CLI(`predict.py` /
`video_predict.py`)并存:CLI 仍是本地调试与 holdout 批量评估入口,本包只负责把
同一套推理逻辑包装成平台要求的长驻 HTTP 服务。**不重写推理**,真实 runner 以参数
注入(见 §3)。

## 0. 状态(2026-08-10)

| 项 | 现状 |
|:--|:--|
| 路由 + 线程池 + 回调闭环 | **已实现 + 已测试**(`tests/test_service_api.py` 20 例) |
| 真实 runner 接线 | **已接线**(Phase 5 A2 落地):`build_service_runner` 已实现(`runner.py`),`app.py:main()` 启动时 `set_runner(build_service_runner(...))` 注入真实推理 runner。`_placeholder_runner` 仅存占位语义(`_runner or _placeholder_runner` 兜底),不再被生产路径使用。见 §3.4 |
| 部署形态 | Dockerfile `ENTRYPOINT` 已切 `python -m src.service.app`(uvicorn 常驻 8000);`entrypoint.sh` 降级为本地/CI 单文件调试旁路(见 `src/inference/AGENTS.md §11`) |

## 1. 文件职责

| File | Responsibility |
|:---|:---|
| `app.py` | FastAPI 路由(`/call` `/health` `/task/{request_id}`)+ runner 注入(`set_runner`/`_placeholder_runner`/全局 `_runner`)+ `_extract_fields` 入参容错 + `main()` uvicorn 前台启动(启动时 `set_runner(build_service_runner(...))` 注入真实 runner) |
| `config.py` | 端口/路径/回调常量(均可环境变量覆盖):`SERVICE_PORT=8000`、`INFERENCE_WORKERS=1`、`WORKSPACE_ROOT`/`ANSWER_ROOT`/`LOGS_ROOT`、`callback_url()`/`answer_dir()`/`pred_path()`、`CALLBACK_MAX_RETRIES=3`/`CALLBACK_TIMEOUT_S=10` |
| `tasks.py` | 单例 `ThreadPoolExecutor(max_workers=1)` + `submit_inference` + `post_callback`(指数退避重试)+ `_write_task_error`(原子写)+ `_tasks` 状态表 + `task_summary`/`get_task_status`/`shutdown` |
| `case_discovery.py` | **A2 新增** `discover_cases(dataset_path)` 纯函数:递归遍历目录,静/动分派(`.dcm`→static,`.nii.gz`/`.nii`→dynamic)+ 取 PatientId/subdir_uid(静态读 DICOM 头,动态按组委会口径取直接父目录名)。只做发现不接推理,无重依赖 |
| `runner.py` | **A2 新增** 真实推理 runner:模型只 build 一次跨病例复用 + 单病例失败隔离 + V2.1 落盘。见 §4.1 |

## 2. 官方硬约束 ↔ 实现

| 规范要求 | 实现 |
|:--|:--|
| `POST /call`,5 秒内返回 200 | `call()` 只做入参解析 + 丢线程池,不等推理(连数据集路径是否存在都不查 —— 网络文件系统 `stat` 可能挂起数秒) |
| 异步响应机制 | `tasks.submit_inference` 走 `ThreadPoolExecutor` |
| `GET /health` 返 200,端口 8000 | `health()`;端口见 `config.SERVICE_PORT` |
| 回调回传 `request_id` + `evaluationId` + `predPath` | `tasks.post_callback`(worker 线程内,含重试) |
| 前台长进程 | `python -m src.service.app` → `uvicorn.run` 阻塞前台(后台进程会被平台关闭) |

## 3. 关键约束与"为什么"

### 3.1 worker=1 串行(`config.INFERENCE_WORKERS=1`)
单卡 32G 显存(2026-08-12 云 agent 实测,见总纲 §3;2026-08-11 的"14~16G"误判已作废)下并发推理仍可能吃紧(模型非线程安全)。串行执行、
队列排队是唯一安全解。**线程池必须模块级单例**:worker=1 的排队语义要求所有任务
共用同一个池,各自新建等于并发,直接违背串行前提。

### 3.2 /health 必须极轻不阻塞
平台探活探针每 30 秒打一次 `/health`,**连续失败 3 次就重启测评容器**;启动探针
5s/次最长 1800s。torch 推理是同步阻塞的 CPU/GPU 代码 —— 若用 `asyncio.create_task`
直接在事件循环里跑,单张图推理就会把整个 event loop 堵住,`/health` 无法响应,90 秒后
容器被强制重启,**整场评测进度作废且不可恢复**。故推理一律丢线程池,event loop 只
负责收请求和回调,永远不阻塞。`health()` 与 `task_summary()` **绝不触碰 `_executor`**,
只读 `_tasks` 内存 dict 的浅拷贝(锁临界区微秒级)。

### 3.3 /call 立即返回不做耗时操作
规范:5 秒内未响应判定请求异常,且**只有一次提交机会**,没有试错空间。任何可能阻塞
的检查(模型加载、磁盘 stat、权重校验)都交由 worker 线程判断,失败写 `_error.json`
并照常回调。

### 3.4 runner 参数注入(已接线)
`runner = callable(dataset_path, output_dir) -> None`,以参数注入而非直接 import。
理由:① 测试可替换成假 runner,不必加载模型;② 推理主流程(多结节改造)与服务层
解耦,两边可独立演进。**`app.py:main()` 启动时 `set_runner(build_service_runner(...))`
注入真实 runner**(模型权重/config/precision 由环境变量 `AICOMP_WEIGHTS`/`AICOMP_CONFIG`/
`AICOMP_PRECISION` 注入,缺省回落 CLI 同款路径)。`_runner or _placeholder_runner` 回落到
`_placeholder_runner` 的占位语义**仅当 `set_runner` 未被调用时**才触发(Dockerfile ENTRYPOINT
已切 `python -m src.service.app`,生产路径必走 `main()` 注入,占位不再被生产使用)。

### 3.5 无论成败都回调
worker 线程 `finally` 块无条件调 `post_callback`。平台靠回调判定该任务已终结;
**失败不回调会让平台一直等到评测超时**,比回调一个"结果目录里只有 `_error.json`"更糟。

### 3.6 _error.json 先写盘再翻 FAILED
`_run()` 捕获异常后**先**`_write_task_error` 落盘、**再**`_set_status(STATUS_FAILED)` ——
保证"观察到 FAILED"即蕴含"_error.json 已存在",消除消费方(平台/人工/测试 `_wait_status`)
在两行之间读到"FAILED 但结果目录空"的竞态(与"还没跑"无法区分)。写盘失败被
`_write_task_error` 内部 `except` 吞掉,不阻断状态翻转。

## 4. runner 契约 + V2.1 输出

```
runner(dataset_path: str, output_dir: str) -> None
```

- 遍历 `dataset_path` 目录逐病例,输出到 `output_dir` = `config.answer_dir(evaluation_id)`
  = `{ANSWER_ROOT}/{evaluation_id}`(该子目录"必须由参赛者代码在推理时动态创建",
  规范 §二.2 原文;`app._run` 已 `os.makedirs(out_dir, exist_ok=True)` 预建)。
- V2.1 逐病例产物:`{output_dir}/{PatientId}/prediction.json` + 逐结节 `.nii.gz`
  mask(0/1,H×W 与原图严格一致)。**真实 runner 已实现**(见 §4.1),不再是占位。

### 4.1 `runner.py` — 真实推理 runner(Phase 5 A2 落地)

`build_service_runner(config_path=None, weights=None, precision=None, max_frames=None)`
→ `runner(dataset_path, output_dir) -> None`,签名与 `tasks.submit_inference` 调用点逐字对齐。

**三条红线(对应 QA_VP §1.7 / rules.md §5)**:
1. **模型只 build 一次、跨病例复用**:`build_model`/`load_config`/`TemperatureScaler`/
   gatekeeper 三件重型对象(dedup/auth/tamper)在闭包 `_ensure_built` 里建一次,绝不每病例
   重载(批量每病例重读权重 + 重播种 200 张图会把延迟打爆,赛道有推理时延评分)。模块级
   `_RUNNER_CACHE`(key=路径精度参数)+ 锁让「同参数重复调 build_service_runner」(多次 `/call`
   或测试)也只建一次。**惰性构建**:真正 build 推迟到 runner 首次被调用,避免 `main()` 启动期
   因模型未就位崩掉 8000 探针(平台连续 3 次探活失败重启容器)。
2. **单病例失败隔离**:`_process_case` 每病例独立 `try/except Exception`,一例崩溃(权重不匹配/
   CUDA OOM/NIfTI 损坏/数值 NaN)只落该例的 `prediction.json` ERROR 占位 + 记日志,**绝不 raise
   拖垮整批**(平台只有一次评测机会,一例崩掉整批 = 全部零输出)。
3. **匿名化/gatekeeper 不因批量而跳过**:每病例照跑 `load_input_image` 的 DICOM PHI 匿名化与
   逐帧 gatekeeper 审计,与 CLI 同语义。gatekeeper 默认 soft(赛道七豁免),`passed` 恒 True;
   仅显式 `enforcement_mode: hard` 且命中硬判据才走拒绝分支(`_rejection_result`,落 REJECTED
   占位不产诊断)。

**失败/拒绝占位**:`_write_case_error` 复用 write_case_v21 的原子写(V2.1 目录里不出现 V2.0 的
`result.json`),占位 nodule 只保 JSON 结构合规(数值保守占位 + `Interpretation.Conclusion` 写明
ERROR/REJECTED 原因)。**该函数自身绝不抛**(写盘失败只记日志),否则隔离语义自相矛盾。

**`subdir_uid`/`orig_hw` 透传**:`case_discovery` 定的 `subdir_uid` 是 V2.1 动态输出子目录名
(= AccessionNumber,组委会口径与 PatientId 同源);`_run_static_case` 透传 `orig_hw=img_np.shape[:2]`
(缺它 write_case_v21 无法把 mask resize 回原图)。

**@云TODO(云端首跑校正)**:权重/config 路径来自环境变量(缺省回落 CLI 同款路径),容器内挂载点
不同须云端首跑核对;gatekeeper 批量耗时未实测;动态 AccessionNumber 来源已定(组委会口径),但
推理输入目录层级仍待云端首跑核对。

## 5. 入参容错(`_extract_fields`)

规范只列"入参关键字段"(`request_id` / `input.evaluation_id` / `input.dataset_path`),
未给完整 JSON Schema,也没说明字段是否可能在顶层。严格按嵌套解析会让平台实际报文
有一处不同就 422 —— 而 5 秒内没返 200 即判异常,且只有一次提交机会。故:

- **兼容嵌套 `input.*` 与顶层两种位置**(同时尝试 `request_id`/`requestId`、
  `evaluation_id`/`evaluationId`、`dataset_path`/`datasetPath` 两种命名);
- **`request_id` 缺失自动补 UUID**(宁可回调时带我方生成的 id,也不要整个请求 422 掉);
- **非法 JSON 给 400 不给 500**(明确 `{"code":400,"message":"invalid JSON body"}`);
- **唯一硬性拒绝:缺 `evaluation_id`** → 400(无处写结果、无法回调闭环);
  其余字段缺失一律放行,交 worker 线程判断失败再兜。

## 6. 配置常量速查(均可环境变量覆盖)

| 常量 | 默认 | env | 说明 |
|:--|:--|:--|:--|
| `SERVICE_PORT` | `8000` | `AICOMP_PORT` | 规范 §三 硬约束,健康检查必须在 8000 |
| `SERVICE_HOST` | `0.0.0.0` | `AICOMP_HOST` | 容器须监听 0.0.0.0 |
| `INFERENCE_WORKERS` | `1` | `AICOMP_INFERENCE_WORKERS` | **必须为 1**(见 §3.1) |
| `WORKSPACE_ROOT` | `/2026aicompetition/workspace` | `AICOMP_WORKSPACE` | 平台挂载的工作区根 |
| `ANSWER_ROOT` | `{WORKSPACE_ROOT}/answer` | `AICOMP_ANSWER_ROOT` | 推理结果根;`{eval_id}` 子目录由代码动态建 |
| `LOGS_ROOT` | `{WORKSPACE_ROOT}/logs` | `AICOMP_LOGS_ROOT` | 平台读不到指定日志"可能被标记可疑" |
| `CALLBACK_BASE_URL` | `http://10.1.13.241:80` | `AICOMP_CALLBACK_BASE_URL` | 平台内网网关(与容器 SSH 主机一致);**默认不设空串** —— 留空会让"忘了配环境变量"静默退化成"不回调",平台等不到回调挂到评测超时,比连不上报错更难排查 |
| `CALLBACK_PATH` | `/api/competition/inference/callback/` | — | 回调固定路径 |
| `CALLBACK_MAX_RETRIES` | `3` | `AICOMP_CALLBACK_RETRIES` | 指数退避 2/4/8s,总耗时上界约 14s |
| `CALLBACK_TIMEOUT_S` | `10` | `AICOMP_CALLBACK_TIMEOUT` | 单次回调 HTTP 超时 |

### predPath 必须是绝对路径(2026-08-09 修正)
回调 body 的 `predPath` 取 `config.pred_path(evaluation_id)` = `answer_dir(evaluation_id)`
的绝对路径(`/` 归一化)。**此前实现是错的**(按 body 示例 `canswer/...` 判为笔误、
取相对路径 `answer/{id}`) —— 官方推理服务代码示例才是权威写法:
`"predPath": f"/2026aicompetition/workspace/answer/{evaluation_id}"`。
平台拿这个值定位结果目录,路径形态不对就是**找不到结果 → 该次评测零分**,
且回调返回 200、我方日志一切正常,典型静默失败。`pred_path` 随 `ANSWER_ROOT` 走
(而非硬编码字符串),`AICOMP_ANSWER_ROOT` 覆盖时 predPath 自动跟随。
`relative_pred_path` 是向后兼容别名,语义已证伪,保留仅为不破坏旧调用点。

## 7. 测试覆盖

| 测试文件 | 覆盖点 | 例数 |
|:--|:--|:--|
| `tests/test_service_api.py` | 服务层 HTTP 路由/回调/容错(见 src/service/AGENTS.md 正文) | 20 |
| `tests/test_service_runner.py` | runner 批量编排:逐病例调 write_case_v21、单病例失败隔离落 ERROR 占位、模型只 build 一次(跨病例 + 跨多次 build_service_runner 缓存)、空/无效数据集不崩 | 12 |
| `tests/test_case_discovery.py` | `discover_cases` 静/动分派、静态 DICOM 头读取、动态按组委会口径取直接父目录名、路径推断、空目录不崩、**跨文件系统确定性顺序契约**(反转 os.scandir 序仍恒为字典序 DFS,钉死 `dirs.sort()`;2026-08-12 CI flaky 修复) | 40 |

`TestPlaceholderRunner` 用例(占位 runner 明确标注输出为空)仍在 `test_service_api.py` 中,
验证 `_runner` 未注入时占位语义存在(向后兼容,不参与生产路径)。
