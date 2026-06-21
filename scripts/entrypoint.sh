#!/bin/bash
# entrypoint.sh - 离线容器防篡改自检与推理启动入口

# 1. 预设安全哈希指纹（防篡改签名）
# 在生产交付前，由打包流水线算出并在此写死。运行时自检对比。
EXPECTED_PREDICT_SHA256="4d7c50a1d62c9adb5a9513e72c4dedafc1cb2d4c5236c9a6957ec7dfd5a9ca22" # 示例哈希
EXPECTED_WEIGHTS_SHA256="ef7c29e0625c9adb5a9513e72c4dedafc1cb2d4c5236c9a6957ec7dfd5a9c2222" # 示例哈希

echo "============================================="
echo "   Medical PACS Sandbox Container Initialization"
echo "============================================="
echo "Running offline diagnostic checks and signature validation..."

# 2. 检查预测入口 predict.py 完整性
if [ -f "predict.py" ]; then
    ACTUAL_PREDICT_SHA=$(sha256sum predict.py | awk '{print $1}')
    echo "--> predict.py SHA-256: $ACTUAL_PREDICT_SHA"
else
    echo "Error: predict.py not found!" >&2
    exit 1
fi

# 3. 检查模型权重文件
if [ -f "checkpoints/best_model.pth" ]; then
    ACTUAL_WEIGHTS_SHA=$(sha256sum checkpoints/best_model.pth | awk '{print $1}')
    echo "--> checkpoints/best_model.pth SHA-256: $ACTUAL_WEIGHTS_SHA"
else
    echo "[WARNING] Trained weights not found. Skipping hash validation."
fi

# 4. 如果处于生产环境 (PRODUCTION=1)，执行强哈希一致性与文件存在性比对阻断
if [ "$PRODUCTION" = "1" ]; then
    if [ "$ACTUAL_PREDICT_SHA" != "$EXPECTED_PREDICT_SHA256" ]; then
        echo "CRITICAL ERROR: predict.py integrity hash mismatch! Sandbox code has been tampered with." >&2
        exit 1
    fi
    if [ ! -f "checkpoints/best_model.pth" ]; then
        echo "CRITICAL ERROR: checkpoints/best_model.pth weight file is missing in production container!" >&2
        exit 1
    fi
    if [ "$ACTUAL_WEIGHTS_SHA" != "$EXPECTED_WEIGHTS_SHA256" ]; then
        echo "CRITICAL ERROR: checkpoints/best_model.pth weight signature mismatch! Parameters have been altered." >&2
        exit 1
    fi
    echo "--> Offline production signature verification SUCCESS."
else
    echo "--> [DEVELOPMENT MODE] Integrity verification bypass. Starting diagnosis..."
fi
echo "============================================="

# 4. 执行多任务诊断推理
exec python predict.py "$@"
