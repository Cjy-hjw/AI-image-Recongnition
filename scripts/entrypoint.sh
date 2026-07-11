#!/bin/bash
# entrypoint.sh - 离线容器防篡改自检与推理启动入口
#
# 防护边界(威胁模型,reviewer D):本机制是「自包含的篡改检测」,非密码学防篡改。
#   能挡:容器运行时 src/入口/权重被替换或损坏(入口脚本烤进只读镜像层、
#         src/权重经可写 volume 落地的典型场景)、不完整拷贝。
#   挡不住:EXPECTED_* 写死在本脚本自身,且 src 树指纹不含 entrypoint.sh ——
#         能改 src 的攻击者通常也能改本脚本(重算哈希填回/删校验块)即可绕过。
#   真防篡改需信任锚外移:镜像签名 + 外部验签(cosign/attestation)。此处自检
#   仅作交付完整性兜底,不构成对抗强攻击者的安全承诺(医疗审计场景须知此边界)。

# 1. 预设安全哈希指纹（防篡改签名）
# 在生产交付前，由打包流水线算出并在此写死。运行时自检对比。
# predict.py(单图)与 video_predict.py(动态图/cine)同为交付入口,均纳入完整性校验。
EXPECTED_PREDICT_SHA256="4d7c50a1d62c9adb5a9513e72c4dedafc1cb2d4c5236c9a6957ec7dfd5a9ca22" # 示例哈希
EXPECTED_VIDEO_PREDICT_SHA256="0000000000000000000000000000000000000000000000000000000000000000" # 打包时由流水线写死
EXPECTED_SRC_TREE_SHA256="0000000000000000000000000000000000000000000000000000000000000000" # src/ 全树聚合指纹,由 seal_container.py 写死
EXPECTED_WEIGHTS_SHA256="ef7c29e0625c9adb5a9513e72c4dedafc1cb2d4c5236c9a6957ec7dfd5a9c222" # 示例哈希(2026-07-11修复:此前65位多一位,标准SHA256十六进制应为64位)

# src/ 全树聚合指纹函数:从独立文件 source(与 tests 共用单一事实来源,杜绝 drift)。
# 算法与 scripts/seal_container.py 字节级配对,边界说明见 _src_tree_hash.sh。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_src_tree_hash.sh"

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
    ACTUAL_SRC_TREE_SHA=$(compute_src_tree_sha256)
    echo "--> src/ tree SHA-256: $ACTUAL_SRC_TREE_SHA"
    if [ "$ACTUAL_SRC_TREE_SHA" != "$EXPECTED_SRC_TREE_SHA256" ]; then
        echo "CRITICAL ERROR: src/ tree integrity hash mismatch! Core inference code has been tampered with." >&2
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

# 5. 输入格式识别与分发:扩展名判断为主,文件内容 magic bytes 嗅探为交叉校验兜底。
#    视频(mp4/avi/mov/mkv)→ video_predict.py;DICOM(dcm/dicom)可能单帧或多帧 cine,
#    统一交给 video_predict.py(其内部 _load_dicom_cine 对 2D 单帧也安全);其余静态图
#    (png/jpg/...)→ predict.py。第一个位置参数为输入路径,其余 flag 原样透传。
#
#    背景(见 docs/比赛对齐与风险.md「动态图交付格式未知风险」):官方"动态图"具体交付
#    格式(容器扩展名?DICOM多帧?其他封装?)从未在任何官方答疑中明确过。纯扩展名白名单
#    在实际格式超出预期时(如用了白名单外的容器扩展名、或扩展名与真实内容不符)会静默
#    误判成静态图,导致该动态图样本 100% 只拿到 ERROR、丢失全部有效预测。
#
#    嗅探策略(不猜测,不静默):
#      ① 文件不存在/不可读 → 无法嗅探,退化为纯扩展名路由(不引入新失败模式,保持
#         对不存在输入文件的既有行为——是否存在由下游 predict.py/video_predict.py
#         的加载失败分支负责报错,不是本层职责)。
#      ② DICOM:offset 128 起紧跟 4 字节 "DICM" 魔数,与文件开头 128 字节任意
#         preamble、扩展名、file(1) magic db 版本均无关,单独显式判定(不能依赖
#         file --mime-type,不同环境/精简容器的 magic db 是否收录 DICOM 签名不保证)。
#      ③ 其余用 `file --mime-type -b` 取 image/*、video/* 前缀判定;`file` 命令
#         本身不可用时同样退化为纯扩展名路由。
#      ④ 嗅探结果与扩展名判断的路由桶不一致 → 以内容嗅探为准重新路由,打印
#         [WARNING] 说明,不静默按(可能错误的)扩展名继续走。
#      ⑤ 嗅探不出来(文件可读,但既非 DICOM 也非 image/*、video/*)→ 不猜测路由,
#         直接判定 ERROR、落 result.json、exit 1、不 exec 任何 python 入口——
#         ERROR 是诚实的失败状态,比误判成静态图导致整段动态图样本丢失预测更安全。
INPUT_PATH="$1"
shift || true
ext_lc=$(echo "${INPUT_PATH##*.}" | tr 'A-Z' 'a-z')
case "$ext_lc" in
    mp4|avi|mov|mkv|dcm|dicom) ext_kind="dynamic" ;;
    *)                         ext_kind="static" ;;
esac

# sniff_kind: dynamic|static|unknown|skip("skip"= 文件不存在/不可读/file命令不可用,
# 无嗅探依据,退化为纯扩展名路由,不当作"unknown"处理——两者语义不同,前者不该硬失败)
sniff_kind="skip"
if [ -f "$INPUT_PATH" ] && [ -r "$INPUT_PATH" ]; then
    dicom_magic=$(dd if="$INPUT_PATH" bs=1 skip=128 count=4 2>/dev/null | tr -d '\0')
    if [ "$dicom_magic" = "DICM" ]; then
        sniff_kind="dynamic"
    elif command -v file >/dev/null 2>&1; then
        content_mime=$(file --mime-type -b "$INPUT_PATH" 2>/dev/null)
        case "$content_mime" in
            image/*) sniff_kind="static" ;;
            video/*) sniff_kind="dynamic" ;;
            *)       sniff_kind="unknown" ;;
        esac
    fi
fi

if [ "$sniff_kind" = "unknown" ]; then
    echo "CRITICAL ERROR: Cannot identify input format by content -- neither DICOM preamble nor image/video magic bytes matched (path=$INPUT_PATH, ext=.$ext_lc)." >&2
    # 从透传参数里找 --output_dir(等号/空格两种写法都认),没传则落 predict.py 同款默认值,
    # 保证 ERROR result.json 落在评测管线实际会去读的目录,而非静默丢弃。
    out_dir="data/output"
    prev_arg=""
    for arg in "$@"; do
        case "$arg" in
            --output_dir=*) out_dir="${arg#--output_dir=}" ;;
        esac
        if [ "$prev_arg" = "--output_dir" ]; then out_dir="$arg"; fi
        prev_arg="$arg"
    done
    mkdir -p "$out_dir" 2>/dev/null
    printf '{\n  "status": "ERROR",\n  "reason": "Unrecognized input format: extension .%s and content magic bytes both failed to identify image/video/DICOM type"\n}\n' \
        "$ext_lc" > "$out_dir/result.json"
    exit 1
fi

final_kind="$ext_kind"
if [ "$sniff_kind" != "skip" ] && [ "$sniff_kind" != "$ext_kind" ]; then
    echo "--> [WARNING] Extension-based routing (.$ext_lc -> $ext_kind) mismatches content sniff ($sniff_kind) -- rerouting by content, not extension."
    final_kind="$sniff_kind"
fi

case "$final_kind" in
    dynamic)
        echo "--> Detected dynamic/cine input (.$ext_lc) -> video_predict.py"
        exec python video_predict.py --input_video "$INPUT_PATH" "$@"
        ;;
    *)
        echo "--> Detected static image input (.$ext_lc) -> predict.py"
        exec python predict.py --input_image "$INPUT_PATH" "$@"
        ;;
esac
