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
if [ -f "checkpoints/thyroidxl_merged_v2.pth" ]; then
    ACTUAL_WEIGHTS_SHA=$(sha256sum checkpoints/thyroidxl_merged_v2.pth | awk '{print $1}')
    echo "--> checkpoints/thyroidxl_merged_v2.pth SHA-256: $ACTUAL_WEIGHTS_SHA"
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
    if [ ! -f "checkpoints/thyroidxl_merged_v2.pth" ]; then
        echo "CRITICAL ERROR: checkpoints/thyroidxl_merged_v2.pth weight file is missing in production container!" >&2
        exit 1
    fi
    if [ "$ACTUAL_WEIGHTS_SHA" != "$EXPECTED_WEIGHTS_SHA256" ]; then
        echo "CRITICAL ERROR: checkpoints/thyroidxl_merged_v2.pth weight signature mismatch! Parameters have been altered." >&2
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

# out_dir 解析(等号/空格两种 --output_dir 写法都认,没传则落 predict.py 同款默认值)。
# 提前到分发前解析,供下方"未知格式"ERROR分支与"子进程崩溃兜底"分支共用同一份逻辑,
# 避免两处重复解析 drift。
out_dir="data/output"
prev_arg=""
for arg in "$@"; do
    case "$arg" in
        --output_dir=*) out_dir="${arg#--output_dir=}" ;;
    esac
    if [ "$prev_arg" = "--output_dir" ]; then out_dir="$arg"; fi
    prev_arg="$arg"
done

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

# 6. 子进程崩溃兜底(2026-07-12,P2红线修复C级)。
#    背景:此前用 `exec python ...` 直接把当前 bash 进程替换成 python 进程——
#    若 python 侧发生原生崩溃(segfault)或被 OOM-killer 发 SIGKILL,没有任何存活的
#    父进程能补写 result.json,评测方拿到的是空目录/零输出,直接判零分(比任何
#    数值误判都严重)。改为前台子进程 + 父 shell 存活等待 + 崩溃兜底,但**诚实
#    说明边界**:此修法能接住 (a) python 自身 segfault (b) python 子进程单独被
#    OOM-killer 选中 SIGKILL 这两类主要情形;接不住 (c) 整个 cgroup/容器被外部
#    杀掉(父 shell 与子进程一起消失,无法留下任何痕迹)——(c) 这种情况下父 shell
#    自身也不存在了,任何用户态兜底都无能为力,需要容器编排层(如 k8s 存活探针
#    + 重启策略)兜底,不是本脚本能解决的问题,不在此虚假宣称"已解决"。
#
#    陈旧文件规避:先把上一次调用可能残留的 result.json 移开,这样"本次是否
#    产出新结果" = "文件是否重新出现"——否则子进程崩溃但 out_dir 里躺着上一次
#    调用的旧 result.json,会被误判成"这次成功了",而实际评测的是陈旧/错误
#    归属的结果,比空目录更危险(空目录至少诚实地是"没有结果")。
#
#    信号转发:不再 exec 替换进程,故 bash 收到的 SIGTERM/SIGINT(如容器编排器
#    优雅关闭发出)不会自动传给子进程——显式 trap 转发,避免子进程变孤儿、
#    容器优雅关闭超时后被强杀的情况。
result_file="$out_dir/result.json"
mkdir -p "$out_dir" 2>/dev/null
if [ -f "$result_file" ]; then
    mv -f "$result_file" "$result_file.stale" 2>/dev/null
    if [ -f "$result_file" ]; then
        # mv 失败(只读文件系统/权限问题)会被 2>/dev/null 静默吞掉;显式核实一次,
        # 失败时至少把风险打到 stderr——下面"result.json 是否存在"的判断此时
        # 已经不可靠(陈旧文件仍占着位置),不诚实地假装陈旧文件规避生效了。
        echo "--> [WARNING] failed to move stale result.json aside (readonly fs / permission issue?) -- crash-fallback staleness guard is degraded for this invocation" >&2
    fi
fi

child_pid=""
signal_missed=""
forward_signal() {
    # child_pid 已可用时直接转发,不重复标记;child_pid 还没赋值(窄窗口内)
    # 时才记为"漏转发",交给下面 child_pid=$! 之后的补偿逻辑发**这一次**——
    # 避免同一个信号被 trap 本身 + 补偿逻辑各发一次、变成转发两次(2026-07-12
    # Gate2 复审 reviewer 子agent 发现的双发问题,当前子进程无自定义信号
    # handler 故双发无害,但严谨起见仍去重)。
    if [ -n "$child_pid" ]; then
        kill -TERM "$child_pid" 2>/dev/null
    else
        signal_missed="1"
    fi
}
trap forward_signal TERM INT

case "$final_kind" in
    dynamic)
        echo "--> Detected dynamic/cine input (.$ext_lc) -> video_predict.py"
        python video_predict.py --input_video "$INPUT_PATH" "$@" &
        ;;
    *)
        echo "--> Detected static image input (.$ext_lc) -> predict.py"
        python predict.py --input_image "$INPUT_PATH" "$@" &
        ;;
esac
child_pid=$!
# 窄窗口补偿(2026-07-12 Gate2 复审发现):trap 在 child_pid 真正赋值前(上方
# "trap forward_signal TERM INT" 那行)就已生效——若信号恰好落在 case 分支
# 选择 ~ child_pid=$! 赋值之间到达,forward_signal 里 `[ -n "$child_pid" ]`
# 判假,只记 signal_missed 不发送。这里补发**且只发这一次**:child_pid 一旦
# 可用,若该窗口期间确实漏转发过就现在补发;若 forward_signal 已经直接发过
# (child_pid 当时已可用),signal_missed 不会被置位,不会重复发送。
if [ -n "$signal_missed" ]; then
    kill -TERM "$child_pid" 2>/dev/null
fi

# wait 在被 trap 捕获的信号打断时会提前返回(bash 语义:trap handler 执行完
# 后 wait 立即返回一个 128+信号 的合成退出码),而子进程此时可能仍未真正
# 退出——已用真实 bash 环境实测复现(trap 转发 TERM 后 wait 立即返回,但
# `ps -p "$child_pid"` 确认子进程仍存活)。用 kill -0 探活循环,只有子进程
# 真正消失后才采信 exit_code,避免子进程"正准备写出合法 result.json"时被
# 提前判定为已结束/崩溃。
while :; do
    wait "$child_pid"
    exit_code=$?
    kill -0 "$child_pid" 2>/dev/null || break
done

# 收尾窗口(补写崩溃兜底 JSON、清理 .stale)故意不提前清 trap:若二次信号在
# 这段窗口到达,trap 仍能吸收它(bash 对已设 trap 的信号默认不会终止 shell),
# 避免 shell 自身被第二次 TERM 打断、来不及写完兜底 JSON——解除转发放在最后
# exit 前一刻。

if [ ! -f "$result_file" ] && [ "$exit_code" -ne 0 ]; then
    mkdir -p "$out_dir" 2>/dev/null
    printf '{\n  "status": "ERROR",\n  "reason": "Inference subprocess exited with code %s without producing result.json (possible native crash/segfault or OOM-kill; NOT covered: whole-container/cgroup kill, see entrypoint.sh comments)"\n}\n' \
        "$exit_code" > "$result_file.tmp" && mv -f "$result_file.tmp" "$result_file"
    echo "--> [entrypoint] subprocess exit=$exit_code, no result.json produced -> wrote fallback ERROR json" >&2
fi
rm -f "$result_file.stale" 2>/dev/null

trap - TERM INT
exit "$exit_code"
