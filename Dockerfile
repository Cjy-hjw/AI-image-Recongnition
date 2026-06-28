# 使用带 CUDA 12.1 和 Python 3.10 的官方 PyTorch 基础镜像
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 设置容器工作目录
WORKDIR /workspace

# 设置环境变量，防止 Python 生成 pyc 缓存和缓冲输出
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装系统级底层依赖（OpenCV 和基础构建工具）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目依赖描述并安装（使用清华源加速离线包安装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 拷贝项目全部核心代码和配置文件
COPY configs/ /workspace/configs/
COPY src/ /workspace/src/
COPY predict.py /workspace/predict.py
COPY video_predict.py /workspace/video_predict.py
COPY scripts/entrypoint.sh /workspace/scripts/entrypoint.sh

# 建立数据输出和 checkpoints 目录
RUN mkdir -p /workspace/checkpoints /workspace/data/output

# 赋予入口启动脚本可执行权限
RUN chmod +x /workspace/scripts/entrypoint.sh

# 默认设置 entrypoint 启动容器时执行哈希验证与推理
ENTRYPOINT ["/workspace/scripts/entrypoint.sh"]
