# 🔴 2026-08-30 对齐云端实际部署镜像(此前是 pytorch:2.1.0,与云端官方镜像
# pytorch:2.8.0-cuda12.8 脱节 —— requirements 的 numpy<2 旧约束正是因此
# 在云端把 numpy2 降级到 1.26.4 造成部署隐患,CI 又只在 2.1.0 下跑测不出)。
# 本 Dockerfile 仅供 CI 冒烟,云上直接用官方镜像 + pip install,但环境必须一致。
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# 设置容器工作目录
WORKDIR /workspace

# 设置环境变量，防止 Python 生成 pyc 缓存和缓冲输出
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 2026-08-11 #115 注记(此处不设 ENV PRODUCTION=1,原因见下):
# 主诊断模型权重缺失的硬报错(model_runner.py build_model 的 PRODUCTION 分流)本意是
# 防交付容器漏 COPY 权重时静默随机初始化出垃圾诊断。但**镜像本身不打权重**
# (checkpoints/ 走官方平台人工上传,见 ci.yml 铁律3注释),CI 的 docker-build-smoke /
# entrypoint-dispatch-smoke 正是靠「无权重→随机降级」跑机制层冒烟。若在此无条件
# ENV PRODUCTION=1,会让这些 CI job 因缺权重硬报错而崩(自造回归)。
# 正解:PRODUCTION=1 由**交付封装那一步**(最终权重就位后,见交付 ToDO #111/#112 +
# seal_container)注入,或部署时 docker run -e PRODUCTION=1。届时权重已在,硬报错只在
# 「该有权重却没有」的真故障时触发,不误伤 CI 无权重冒烟。

# 安装系统级底层依赖（OpenCV、基础构建工具、entrypoint.sh 内容嗅探依赖的 file 命令）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    file \
    && rm -rf /var/lib/apt/lists/*

# 复制项目依赖描述并安装（使用清华源加速离线包安装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 拷贝项目全部核心代码和配置文件
COPY configs/ /workspace/configs/
COPY src/ /workspace/src/
COPY predict.py /workspace/predict.py
COPY video_predict.py /workspace/video_predict.py
# 2026-07-11 修复：此前只拷贝 entrypoint.sh 单个文件，但该脚本内部
# `source "$SCRIPT_DIR/_src_tree_hash.sh"` 依赖同目录下的 _src_tree_hash.sh
# （src 树指纹校验函数，与 tests/test_entrypoint_dispatch.py 的 parity 测试共用
# 同一份实现）——只拷贝单文件会让该 source 在镜像里找不到文件，容器一启动
# entrypoint 就直接崩溃退出，PRODUCTION=1/开发模式均无法启动。
# 精确拷贝这两个文件而非整个 scripts/（该目录下还有 deploy.py/train_*.py/
# verify/ 等一大堆训练部署期脚本，与推理容器无关，塞进去只会不必要地
# 膨胀镜像体积、扩大生产环境攻击面）。
COPY scripts/entrypoint.sh /workspace/scripts/entrypoint.sh
COPY scripts/_src_tree_hash.sh /workspace/scripts/_src_tree_hash.sh

# 建立数据输出和 checkpoints 目录
RUN mkdir -p /workspace/checkpoints /workspace/data/output

# 赋予入口启动脚本可执行权限
RUN chmod +x /workspace/scripts/entrypoint.sh

# 默认设置 entrypoint 启动容器时执行哈希验证与推理
#
# 2026-08-10 V2.1 容器入口切换(阶段 A2):ENTRYPOINT 由离线单文件 CLI
# (entrypoint.sh)改为 **常驻 HTTP 推理服务** `python -m src.service.app`
# (uvicorn 监听 8000,/call 收批量目录任务,/health 供平台探针)。
#
# 背景:官方赛事开发规范 V2.1 把交付形态整体改写——不再是「跑一次推理就 exit」,
# 而是「常驻 HTTP 服务 + 批量目录推理 + 逐病例 V2.1 prediction.json 输出」。
# entrypoint.sh 的 30 个 dispatch 测试 / 封印(seal_container) / 信号转发 / 崩溃兜底
# 仍是已验证资产,**保留作本地/CI 单文件调试旁路**(开发模式下直接运行),但不再是
# 容器生产入口 —— 平台探针打的是 8000,不是 entrypoint 的退出码。
#
# `src.service.app:main()` 启动时 `set_runner(build_service_runner(...))` 注入真实
# 推理 runner(见 src/service/runner.py);模型权重/config/precision 由环境变量
# `AICOMP_WEIGHTS`/`AICOMP_CONFIG`/`AICOMP_PRECISION` 注入,缺省回落 CLI 同款路径。
#
# ⚠️ 调试旁路调用方式(2026-08-10 CI 踩坑修复):**不要** `docker run <image> <input>`
# —— 位置参数会被默认 HTTP ENTRYPOINT(uvicorn)忽略,起成不退出的服务而挂起。
# 要在容器里走单文件 CLI 分发(predict.py/video_predict.py),必须显式覆盖入口:
#   docker run --rm --entrypoint /workspace/scripts/entrypoint.sh <image> <input>
# 生产形态(平台探针打 8000)用下方默认 ENTRYPOINT,不加 --entrypoint。
ENTRYPOINT ["python", "-m", "src.service.app"]
