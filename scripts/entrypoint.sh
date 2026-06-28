#!/bin/bash
# entrypoint.sh - 离线容器防篡改自检与推理启动入口

# 1. 预设安全哈希指纹（防篡改签名）
# 在生产交付前，由打包流水线算出并在此写死。运行时自检对比。
# predict.py(单图)与 video_predict.py(动态图/cine)同为交付入口,均纳入完整性校验。
EXPECTED_PREDICT_SHA256="4d7c50a1d62c9adb5a9513e72c4dedafc1cb2d4c5236c9a6957ec7dfd5a9ca22" # 示例哈希
EXPECTED_VIDEO_PREDICT_SHA256="0000000000000000000000000000000000000000000000000000000000000000" # 打包时由流水线写死
EXPECTED_WEIGHTS_SHA256="ef7c29e0625c9adb5a9513e72c4dedafc1cb2d4c5236c9a6957ec7dfd5a9c2222" # 示例哈希

echo "============================================="
echo "   Medical PACS Sandbox Container Initialization"
echo "============================================="
echo "Running offline diagnostic checks and signature validation..."

# 2. 检查预测入口完整性(两个入口都要在)
for entry in predict.py video_predict.py; do
    if [ -f "$entry" ]; then
        ACTUAL_SHA=$(sha256sum "$entry" | awk '{print $1}')
        echo "--> $entry SHA-256: $ACTUAL_SHA"
        if [ "$entry" = "predict.py" ]; then ACTUAL_PREDICT_SHA="$ACTUAL_SHA"; fi
        if [ "$entry" = "video_predict.py" ]; then ACTUAL_VIDEO_PREDICT_SHA="$ACTUAL_SHA"; fi
    else
        echo "Error: $entry not found!" >&2
        exit 1
    fi
done

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
    if [ "$ACTUAL_VIDEO_PREDICT_SHA" != "$EXPECTED_VIDEO_PREDICT_SHA256" ]; then
        echo "CRITICAL ERROR: video_predict.py integrity hash mismatch! Sandbox code has been tampered with." >&2
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

# 5. 按输入文件扩展名自动分发入口:
#    视频(mp4/avi/mov/mkv)→ video_predict.py;DICOM(dcm/dicom)可能单帧或多帧 cine,
#    统一交给 video_predict.py(其内部 _load_dicom_cine 对 2D 单帧也安全);其余静态图
#    (png/jpg/...)→ predict.py。第一个位置参数为输入路径,其余 flag 原样透传。
INPUT_PATH="$1"
shift || true
ext_lc=$(echo "${INPUT_PATH##*.}" | tr 'A-Z' 'a-z')
case "$ext_lc" in
    mp4|avi|mov|mkv|dcm|dicom)
        echo "--> Detected dynamic/cine input (.$ext_lc) -> video_predict.py"
        exec python video_predict.py --input_video "$INPUT_PATH" "$@"
        ;;
    *)
        echo "--> Detected static image input (.$ext_lc) -> predict.py"
        exec python predict.py --input_image "$INPUT_PATH" "$@"
        ;;
esac
