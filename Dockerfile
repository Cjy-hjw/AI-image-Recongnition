# 使用带 CUDA 12.1 和 Python 3.10 的官方 PyTorch 基础镜像
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 设置容器工作目录
WORKDIR /workspace

# 设置环境变量，防止 Python 生成 pyc 缓存和缓冲输出
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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
ENTRYPOINT ["python", "-m", "src.service.app"]
