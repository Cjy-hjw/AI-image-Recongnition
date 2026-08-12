"""单病例推理核心(静态图 + 动态图),从 `predict.py`/`video_predict.py` 抽出。

## 为什么抽这一层

V2.1 交付把形态从「entrypoint.sh 单文件离线 CLI」改为「HTTP 常驻批量服务」
(`src/service/`)。服务层 runner 要对一个目录里的**批量病例**逐个推理,不能复用
CLI 的 `if __name__=="__main__"` 主函数。故把两个 CLI `main()` 里「输入加载完成 →
落盘之前」那段**纯推理核心**抽到此处,CLI 与 service 共用同一份诊断逻辑,只在
两端各自落盘(CLI 落 V2.0 `result.json`,service 落 V2.1 `prediction.json`)。

## 边界:什么在这里,什么留在调用方

**留在调用方(main/runner)**:
- `build_model` / `TemperatureScaler` 构造 —— 有构建成本,service 要跨病例复用一次;
- 输入加载(`load_input_image`/`load_video_frames`/`load_nifti_frames`)—— 含 DICOM
  匿名化写盘副作用,且失败要落 rejection JSON(控制流属调用方);
- gatekeeper 审计 —— 有副作用(逐帧审计、写 rejection),且 hard 模式的一票否决
  return 语义属调用方控制流;
- 落盘(V2.0 `save_prediction_artifacts` / V2.1 `write_case_v21`)与 Grad-CAM(CLI 专属)。

**在这里(纯核心,无副作用、无落盘)**:
- 静态:`preprocess → run_forward → decode → mask resize → tirads breakdown`;
- 动态:逐帧诊断(坏帧隔离)+ 聚合 + 关键帧选择 + 关键帧张量防错位。

## 无有效结果的两个分支

动态图两种「无法产出任何诊断」的情形(全帧数值失败 / 无有效关键帧)在原 `main()`
里是 `save_rejection_json` + `return`。抽出后本层不落盘,改为 `raise DynamicCaseError`
携带诊断字段,由调用方转成对应 rejection —— 保持与原行为逐字节一致(reviewer P)。
"""

import cv2

from src.inference.model_runner import preprocess_image, run_forward
from src.inference.postprocess import decode_predictions, build_tirads_breakdown
from src.inference.video_aggregate import aggregate_frame_predictions
# NonFiniteProbabilityError 逐帧兜底信号;AssertionError 为帧形状非法(preprocess 校验)。
from src.inference.postprocess import NonFiniteProbabilityError


class DynamicCaseError(Exception):
    """动态图无有效诊断结果(全帧失败 / 无有效关键帧)。

    携带 `reason` 与 `diagnostic`(n_frames_* / failed_* 字段),供调用方原样落
    rejection JSON —— 本层不落盘,把「怎么落」的控制流交回 main/runner(与原
    `video_predict.main()` 的两处 `save_rejection_json`+`return` 逐字节等价)。
    """

    def __init__(self, reason, diagnostic):
        super().__init__(reason)
        self.reason = reason
        self.diagnostic = diagnostic


def diagnose_static_case(model, img_np, config, calibrator, precision, device):
    """单张静态图的推理核心(输入已加载+已过 gatekeeper)。

    Args:
        model: 已 `build_model` 的模型(调用方持有,service 跨病例复用)。
        img_np: 已加载(且必要时匿名化)的 uint8 灰度图 HxW。
        config: 解析后的配置 dict(内部取 `data.img_size`)。
        calibrator: `TemperatureScaler` 或 None(全 T=1.0 时逐位等同无校准)。
        precision: "fp32"/"fp16"/"bf16"。
        device: torch.device。

    Returns:
        dict:
          - `preds`: `decode_predictions` 全量输出(含软 `pred_mask`、`mask_binary`、
            各头 `*_idx`/`*_prob`、`mal_prob`/`mal_probs`);
          - `img_tensor`: `preprocess_image` 产出的普通张量(供调用方在 inference_mode
            之外重跑 Grad-CAM 复用);
          - `mask_resized`: `mask_binary` 用 `INTER_NEAREST` 缩放回原图 HxW(V2.0 落盘用);
          - `breakdown`: `build_tirads_breakdown` 的 TI-RADS 分项。

    Raises:
        NonFiniteProbabilityError: 概率/mask 含 NaN/Inf(`decode_predictions` 硬拦),
            由调用方转 ERROR —— 与原 `predict.main()` 的 except Exception 兜底一致。
    """
    img_size = config["data"]["img_size"]
    img_tensor = preprocess_image(img_np, img_size, device)
    outputs = run_forward(model, img_tensor, precision, device)
    preds = decode_predictions(outputs, calibrator=calibrator)
    mask_resized = cv2.resize(
        preds["mask_binary"], (img_np.shape[1], img_np.shape[0]),
        interpolation=cv2.INTER_NEAREST)
    breakdown = build_tirads_breakdown(
        preds["comp_idx"], preds["echo_idx"],
        preds["shape_idx"], preds["margin_idx"], preds["foci_prob"])
    return {
        "preds": preds,
        "img_tensor": img_tensor,
        "mask_resized": mask_resized,
        "breakdown": breakdown,
    }


def _diagnose_frame(model, frame_np, precision, device, img_size, calibrator=None):
    """单帧前向 → 解码,返回 (preds, img_tensor)。img_tensor 留作关键帧 Grad-CAM 复用。

    与原 `video_predict.diagnose_frame` 同义(该函数保留为向后兼容 shim)。
    """
    img_tensor = preprocess_image(frame_np, img_size, device)
    outputs = run_forward(model, img_tensor, precision, device)
    return decode_predictions(outputs, calibrator=calibrator), img_tensor


def diagnose_dynamic_case(model, frames, frame_indices, config, calibrator,
                          precision, device):
    """动态图(cine)的推理核心:逐帧诊断 + 聚合 + 关键帧(输入已抽帧+已过 gatekeeper)。

    坏帧隔离(与原 `video_predict.main()` 逐字节一致):`NonFiniteProbabilityError`
    (概率/mask NaN/Inf)与 `AssertionError`(帧形状/维度非法)**只丢该帧不拖垮整段**;
    `RuntimeError`(CUDA OOM/权重维度不符)、`cv2.error` 等真故障**冒泡**由调用方转 ERROR。

    关键帧张量防错位(与原实现一致):`img_tensor`(GPU 张量)只保留**当前最恶性帧**
    那一个(边扫边留 argmax 帧,显存从 N 帧降到 1 帧,防官方 14G 上界 OOM);判据用
    严格 `>`,与 `aggregate_frame_predictions` 的 `max`(相等取首个)严格同序。

    Args:
        model/config/calibrator/precision/device: 同 `diagnose_static_case`。
        frames: 已抽样的帧列表(uint8 灰度 HxW),gatekeeper 已在调用方完成。
        frame_indices: 各帧的**原始帧号**(与 frames 等长同序)。

    Returns:
        dict:
          - `per_frame`: 每帧 `{frame_index, mal_prob, preds, frame_np}`(preds 含软
            `pred_mask`,供 V2.1 逐帧 mask 落盘);
          - `agg`: `aggregate_frame_predictions` 结果(mal_prob_max/mean、keyframe_*);
          - `kf_preds`/`kf_frame_np`/`kf_tensor`: 关键帧的解码结果/原图/张量;
          - `failed_frames`: `[{frame_index, reason}, ...]` 被跳过的坏帧;
          - `breakdown`: 关键帧 TI-RADS 分项;
          - `mask_resized`: 关键帧 `mask_binary` 缩放回原图 HxW(V2.0 落盘用)。

    Raises:
        DynamicCaseError: 全帧数值失败 或 无有效关键帧 —— 携带诊断字段,调用方落 ERROR。
    """
    img_size = config["data"]["img_size"]

    # --- 逐帧诊断(坏帧隔离) ---
    per_frame = []
    failed_frames = []
    kf_tensor = None          # 仅当前最恶性帧的张量(Grad-CAM 复用)
    kf_img_tensor_pos = -1    # 它在 per_frame 里的下标
    for orig_idx, frame_np in zip(frame_indices, frames):
        try:
            preds, img_tensor = _diagnose_frame(
                model, frame_np, precision, device, img_size, calibrator=calibrator)
        except (NonFiniteProbabilityError, AssertionError) as e:
            failed_frames.append({"frame_index": int(orig_idx), "reason": str(e)})
            print(f"--> [WARNING] frame {orig_idx} 异常已跳过({type(e).__name__}: {e})"
                  f";整段继续", flush=True)
            continue
        cur = {
            "frame_index": int(orig_idx),
            "mal_prob": float(preds["mal_prob"]),
            "preds": preds,
            "frame_np": frame_np,
        }
        # 严格 `>`:只在**真正更恶性**时刷新保留帧,平分保留首帧,与聚合器 max 同序。
        if kf_tensor is None or cur["mal_prob"] > per_frame[kf_img_tensor_pos]["mal_prob"]:
            kf_tensor = img_tensor
            kf_img_tensor_pos = len(per_frame)
        per_frame.append(cur)

    # 全帧失败 → 无任何可用诊断。抛 DynamicCaseError 由调用方落 ERROR(原为
    # save_rejection_json+return,字段逐字节一致)。
    if not per_frame:
        reason = f"All {len(frames)} sampled frames failed numeric validation"
        print(f"[FATAL] {reason}")
        raise DynamicCaseError(reason, {
            "n_frames_sampled": len(frames),
            "n_frames_diagnosed": 0,
            "n_frames_failed": len(failed_frames),
            "failed_frame_indices": [f["frame_index"] for f in failed_frames],
            "failed_frames": failed_frames,
        })

    # --- 聚合 + 关键帧 ---
    agg = aggregate_frame_predictions(
        [{"frame_index": f["frame_index"], "mal_prob": f["mal_prob"]} for f in per_frame])
    # 双重守卫:keyframe_index<0 绝不能拿去索引(Python 负索引静默取末帧致错位)。
    if agg["keyframe_index"] < 0:
        raise DynamicCaseError(
            f"no valid keyframe (keyframe_index={agg['keyframe_index']})", {
                "n_frames_sampled": len(frames),
                "n_frames_diagnosed": agg["n_frames"],
                "n_frames_failed": len(failed_frames),
                "failed_frame_indices": [f["frame_index"] for f in failed_frames],
                "failed_frames": failed_frames,
            })
    kf = per_frame[agg["keyframe_index"]]
    kf_preds = kf["preds"]
    kf_frame_np = kf["frame_np"]
    # 一致性锁:保留帧下标须等于聚合关键帧下标,不等说明两条 max 判据分叉(近乎
    # 不可能),此时不静默用错帧张量——打 warning 并重算该关键帧张量。
    if kf_img_tensor_pos != agg["keyframe_index"]:
        print(f"--> [WARNING] 关键帧张量下标({kf_img_tensor_pos})与聚合关键帧"
              f"({agg['keyframe_index']})不一致,Grad-CAM 改用重算张量", flush=True)
        _, kf_tensor = _diagnose_frame(
            model, kf_frame_np, precision, device, img_size, calibrator=calibrator)

    mask_resized = cv2.resize(
        kf_preds["mask_binary"], (kf_frame_np.shape[1], kf_frame_np.shape[0]),
        interpolation=cv2.INTER_NEAREST)
    breakdown = build_tirads_breakdown(
        kf_preds["comp_idx"], kf_preds["echo_idx"],
        kf_preds["shape_idx"], kf_preds["margin_idx"], kf_preds["foci_prob"])

    return {
        "per_frame": per_frame,
        "agg": agg,
        "kf_preds": kf_preds,
        "kf_frame_np": kf_frame_np,
        "kf_tensor": kf_tensor,
        "failed_frames": failed_frames,
        "breakdown": breakdown,
        "mask_resized": mask_resized,
    }
