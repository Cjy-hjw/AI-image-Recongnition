"""动态图/视频(cine)离线推理入口。

复用单帧多任务模型逐帧诊断,再按 `video_aggregate` 聚合:恶性概率取最大帧做
最终判定(癌症检测宁可召回),关键帧(最恶性帧)产出 mask/报告/可选 Grad-CAM。

与 `predict.py`(单图)互不影响——单图路径完全不变。

Gatekeeper 说明:真实性/拼接/copy-move 为一票否决项,对**每个抽样帧**审计,任一
帧失败即整段拒绝。**不做 cine 帧间去重**——同一 cine 相邻帧天然近重复,去重仅针对
跨提交的重复影像(`Deduplicator` 只比对训练集种子,不把本段帧加入索引,故无误判)。
"""

import argparse
import json
import os

import cv2
import torch
import yaml

from src.inference.video_io import load_video_frames
from src.inference.video_aggregate import aggregate_frame_predictions
from src.inference.gatekeeper_runner import run_gatekeeper_audit, build_seeded_deduplicator
from src.inference.model_runner import build_model, preprocess_image, run_forward
from src.inference.postprocess import decode_predictions, build_tirads_breakdown
from src.inference.export import save_prediction_artifacts, maybe_save_gradcam
from src.utils.report_generator import generate_report


def diagnose_frame(model, frame_np, precision, device, img_size):
    """单帧前向 → 解码,返回 (preds, img_tensor)。img_tensor 留作关键帧 Grad-CAM 复用。"""
    img_tensor = preprocess_image(frame_np, img_size, device)
    outputs = run_forward(model, img_tensor, precision, device)
    return decode_predictions(outputs), img_tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_video", type=str, required=True,
                   help="Path to input ultrasound cine/video (MP4/AVI/MOV/MKV or multi-frame DCM)")
    p.add_argument("--config", type=str, default="configs/thyroid_config.yaml")
    p.add_argument("--weights", type=str, default="checkpoints/best_model.pth")
    p.add_argument("--output_dir", type=str, default="data/output")
    p.add_argument("--precision", type=str, default="fp16",
                   choices=["fp32", "fp16", "bf16"], help="Inference precision")
    p.add_argument("--max_frames", type=int, default=32,
                   help="均匀抽样的最大帧数(<=0 表示全部)")
    p.add_argument("--enable_gradcam", action="store_true",
                   help="对关键帧生成 Grad-CAM 恶性热图")
    p.add_argument("--skip_gatekeeper", action="store_true",
                   help="跳过逐帧 gatekeeper 审计(仅调试/无训练权重时用)")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    device = torch.device(
        config["environment"]["device"] if torch.cuda.is_available() else "cpu")
    img_size = config["data"]["img_size"]

    # 1. 抽帧 & 脱敏
    frames, frame_indices, is_dicom, error_msg = load_video_frames(
        args.input_video, args.output_dir, args.max_frames)
    if frames is None:
        print(f"Error: {error_msg}")
        return
    print(f"--> Decoded {len(frames)} sampled frames from {os.path.basename(args.input_video)}")

    model = build_model(config, args.weights, device)

    # 2. 逐帧 gatekeeper(一票否决,任一帧失败整段拒绝)
    if not args.skip_gatekeeper:
        # 去重库只播种一次,跨帧复用(避免每帧重复 ResNet 编码训练集)
        deduplicator = build_seeded_deduplicator(config, device)
        for pos, frame_np in zip(frame_indices, frames):
            passed, rejection_json = run_gatekeeper_audit(
                frame_np, config, device, args.weights, args.output_dir,
                deduplicator=deduplicator)
            if not passed:
                rejection_json["rejected_frame_index"] = int(pos)
                with open(os.path.join(args.output_dir, "result.json"), "w") as f:
                    json.dump(rejection_json, f, indent=2)
                print(f"--> [REJECTED] frame {pos} failed gatekeeper: {rejection_json.get('reason')}")
                return

    # 3. 逐帧诊断
    print(f"Running per-frame multi-task diagnosis under {args.precision.upper()}...")
    per_frame = []
    key_tensors = {}  # pos → img_tensor(供关键帧 Grad-CAM 复用,避免重算)
    for list_pos, (orig_idx, frame_np) in enumerate(zip(frame_indices, frames)):
        preds, img_tensor = diagnose_frame(model, frame_np, args.precision, device, img_size)
        per_frame.append({
            "frame_index": int(orig_idx),
            "mal_prob": float(preds["mal_prob"]),
            "preds": preds,
        })
        key_tensors[list_pos] = (img_tensor, frame_np)

    # 4. 聚合 + 关键帧
    agg = aggregate_frame_predictions(
        [{"frame_index": f["frame_index"], "mal_prob": f["mal_prob"]} for f in per_frame])
    kf_pos = agg["keyframe_index"]
    kf = per_frame[kf_pos]
    kf_preds = kf["preds"]
    kf_tensor, kf_frame_np = key_tensors[kf_pos]

    # 5. 关键帧 mask / TI-RADS / 报告
    mask_resized = cv2.resize(
        kf_preds["mask_binary"], (kf_frame_np.shape[1], kf_frame_np.shape[0]),
        interpolation=cv2.INTER_NEAREST)
    breakdown = build_tirads_breakdown(
        kf_preds["comp_idx"], kf_preds["echo_idx"],
        kf_preds["shape_idx"], kf_preds["margin_idx"], kf_preds["foci_prob"])
    report_txt = generate_report(
        image_id=f"{os.path.basename(args.input_video)}#frame{kf['frame_index']}",
        patient_id="ANONYMOUS_PACS_PATIENT" if is_dicom else "ANONYMOUS",
        breakdown=breakdown, malignancy_prob=kf_preds["mal_prob"])
    report_txt += (
        f"\n\n  [动态图聚合] 抽样 {agg['n_frames']} 帧;恶性概率 "
        f"max={agg['mal_prob_max']:.3f} / mean={agg['mal_prob_mean']:.3f};"
        f"关键帧=原始帧 {agg['keyframe_frame_index']}。最终判定按最恶性帧。")

    # 5b. 关键帧 Grad-CAM(可选,inference_mode 外复用关键帧张量)
    gradcam_done = 0
    if args.enable_gradcam:
        gradcam_done, _ = maybe_save_gradcam(model, kf_tensor, kf_frame_np, args.output_dir)
        if gradcam_done:
            report_txt += "\n  Grad-CAM 可视化解释: 见 gradcam_cls.png(关键帧恶性激活热图)"

    out_json = {
        "status": "PASSED",
        "input_type": "video_cine",
        "n_frames_sampled": agg["n_frames"],
        "malignancy_probability": agg["mal_prob_max"],
        "malignancy_probability_mean": agg["mal_prob_mean"],
        "is_malignant": agg["final_is_malignant"],
        "keyframe_frame_index": agg["keyframe_frame_index"],
        "tirads_points": breakdown["total_points"],
        "tirads_level": breakdown["tr_level"], "risk_percentage": breakdown["risk"],
        "foci_probabilities": kf_preds["foci_prob"].tolist(),
        "gradcam_generated": gradcam_done,
        "per_frame_malignancy": [
            {"frame_index": f["frame_index"], "mal_prob": f["mal_prob"]} for f in per_frame],
    }
    save_prediction_artifacts(args.output_dir, mask_resized, report_txt, out_json)
    print(f"--> Video inference completed! Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
