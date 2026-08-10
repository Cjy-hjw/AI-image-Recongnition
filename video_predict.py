"""动态图/视频(cine)离线推理入口。

复用单帧多任务模型逐帧诊断,再按 `video_aggregate` 聚合:恶性概率取最大帧做
最终判定(癌症检测宁可召回),关键帧(最恶性帧)产出 mask/报告/可选 Grad-CAM。

与 `predict.py`(单图)互不影响——单图路径完全不变。

Gatekeeper 说明:真实性/拼接/copy-move 对**每个抽样帧**审计。默认 `soft` 模式下命中
只记 flag(汇总进 `result.json` 的 `gatekeeper` 字段),**不中断整段诊断**;仅
`gatekeeper.enforcement_mode: hard` 时才恢复「任一帧失败即整段拒绝」。**不做 cine
帧间去重**——同一 cine 相邻帧天然近重复,去重仅针对跨提交的重复影像(`Deduplicator`
只比对训练集种子,不把本段帧加入索引,故无误判)。
"""

import argparse
import os

import torch

from src.inference.video_io import load_video_frames
from src.inference.gatekeeper_runner import (
    run_gatekeeper_audit, build_seeded_deduplicator,
    build_auth_module, build_tamper_detector, resolve_enforcement_mode,
)
from src.inference.model_runner import build_model, load_config, preprocess_image, run_forward
from src.inference.postprocess import decode_predictions, build_probability_export
from src.inference.calibration import TemperatureScaler
from src.inference.case_pipeline import diagnose_dynamic_case, DynamicCaseError
from src.inference.export import save_prediction_artifacts, save_rejection_json, maybe_save_gradcam
from src.utils.report_generator import generate_report


def diagnose_frame(model, frame_np, precision, device, img_size, calibrator=None):
    """单帧前向 → 解码,返回 (preds, img_tensor)。

    向后兼容 shim:核心逻辑已抽到 `case_pipeline._diagnose_frame`,此函数保留供既有
    测试(test_video 等)与外部调用方直接使用,签名与语义不变。
    """
    img_tensor = preprocess_image(frame_np, img_size, device)
    outputs = run_forward(model, img_tensor, precision, device)
    return decode_predictions(outputs, calibrator=calibrator), img_tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_video", type=str, required=True,
                   help="Path to input ultrasound cine/video (MP4/AVI/MOV/MKV or multi-frame DCM)")
    p.add_argument("--config", type=str, default="configs/thyroid_config.yaml")
    p.add_argument("--weights", type=str, default="checkpoints/thyroidxl_merged_v2.pth")
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

    # 配置加载单独一层 try：与 predict.py 同一契约（见 model_runner.load_config
    # docstring），路径不存在/YAML 格式错误/键缺失都必须兜成 ERROR result.json，
    # 不能裸崩溃。os.makedirs 本身不纳入 try，理由同 predict.py。
    try:
        config = load_config(args.config)
        device = torch.device(
            config["environment"]["device"] if torch.cuda.is_available() else "cpu")
        # data.img_size 早期校验探针:诊断核心(case_pipeline)内部才真正取用它,但按本
        # try 块契约(注释见上:"键缺失都必须兜成 ERROR"),缺该键须在此处 KeyError→
        # 落 "Config loading failed" ERROR,而非拖到推理层才 "Inference pipeline failed"
        # (两者 result.json.reason 同为 "KeyError: 'img_size'",此探针只固定 stdout 归属)。
        _ = config["data"]["img_size"]
    except Exception as e:
        print(f"[FATAL] Config loading failed ({type(e).__name__}): {e}")
        save_rejection_json(args.output_dir, {
            "status": "ERROR", "reason": f"{type(e).__name__}: {e}",
        })
        return

    # 1. 抽帧 & 脱敏
    frames, frame_indices, is_dicom, error_msg = load_video_frames(
        args.input_video, args.output_dir, args.max_frames)
    if frames is None:
        print(f"Error: {error_msg}")
        save_rejection_json(args.output_dir, {"status": "ERROR", "reason": error_msg})
        return
    print(f"--> Decoded {len(frames)} sampled frames from {os.path.basename(args.input_video)}")

    # 2-5. 建模 + gatekeeper + 逐帧推理 + 聚合 + 导出。整体包一层 except Exception：
    # 同 predict.py，只兜 build_model() 一处不够——torch.load 遇损坏 checkpoint
    # 会抛 pickle.UnpicklingError（不是 RuntimeError 子类）、CUDA OOM、逐帧推理/
    # 聚合/报告生成任一步也都可能因边界数据崩溃，容器必须始终落 result.json
    # （entrypoint.sh 按扩展名分发到本脚本，是唯一动态图入口）。异常类型名随
    # str(e) 一起落盘辅助诊断（2026-07-10 Gate2审查发现）。
    try:
        model = build_model(config, args.weights, device)
        # 校准器构造一次,跨帧复用(缺参数文件即全 T=1.0 降级,逐帧行为不变)
        cal_path = config.get("calibration", {}).get("param_file", "checkpoints/calibration.json")
        calibrator = TemperatureScaler.from_file(cal_path)

        # 2. 逐帧 gatekeeper。默认 soft 模式:命中只记 flag,**不中断整段**——
        # 旧逻辑任一帧命中即 return,一次误判就白丢整段视频的全部临床得分,
        # 而赛道七 gatekeeper 零计分(见 gatekeeper_runner.resolve_enforcement_mode)。
        # 逐帧审计放大了这个风险(32 帧任一误报即整段归零),故这里是软化的重点。
        gatekeeper_audit = {"skipped": True}
        if not args.skip_gatekeeper:
            # 三个重型对象只构建一次,跨帧复用(避免每帧重复加载权重/播种训练集)
            deduplicator = build_seeded_deduplicator(config, device)
            auth_module = build_auth_module(args.weights)
            tamper_detector = build_tamper_detector(config)
            frame_audits = []
            for pos, frame_np in zip(frame_indices, frames):
                passed, audit = run_gatekeeper_audit(
                    frame_np, config, device, args.weights, args.output_dir,
                    deduplicator=deduplicator,
                    auth_module=auth_module,
                    tamper_detector=tamper_detector)
                if not passed:
                    # 仅 hard 模式可达:保留原「一票否决整段」语义。
                    audit["rejected_frame_index"] = int(pos)
                    save_rejection_json(args.output_dir, audit)
                    print(f"--> [REJECTED] frame {pos} failed gatekeeper: {audit.get('reason')}")
                    return
                if audit["flagged"]:
                    frame_audits.append({"frame_index": int(pos), **audit})
            gatekeeper_audit = {
                "skipped": False,
                "enforcement_mode": resolve_enforcement_mode(config),
                "n_frames_audited": len(frames),
                "n_frames_flagged": len(frame_audits),
                "flagged_frame_indices": [a["frame_index"] for a in frame_audits],
                "any_flagged": bool(frame_audits),
                "any_would_reject_in_hard_mode": any(
                    a["would_reject_in_hard_mode"] for a in frame_audits),
                # 与 predict.py(单图)的键名对齐:单图 audit 用 flagged /
                # would_reject_in_hard_mode,视频聚合层原先只有 any_* 前缀,消费方
                # (评测脚本/人工复核)无法对两条链路写通用读取代码。any_* 保留不删
                # (向后兼容已写好的读取方),新增无前缀别名作为**两入口共有键**。
                "flagged": bool(frame_audits),
                "would_reject_in_hard_mode": any(
                    a["would_reject_in_hard_mode"] for a in frame_audits),
                "flagged_frames": frame_audits,
            }
            if frame_audits:
                print(f"--> [Gatekeeper 软 flag] {len(frame_audits)}/{len(frames)} 帧命中审计,"
                      f"整段继续诊断(帧号 {gatekeeper_audit['flagged_frame_indices']})", flush=True)

        # 3-4. 逐帧诊断(坏帧隔离)+ 聚合 + 关键帧选择,全在 case_pipeline(与 service
        # 共用同一诊断核心)。坏帧隔离/关键帧张量防错位/显存只留 1 帧等逻辑原样搬入,
        # 语义逐字节不变。两种「无有效诊断」情形(全帧数值失败 / 无有效关键帧)在核心里
        # 抛 DynamicCaseError 携带诊断字段,此处转成与原实现完全一致的 ERROR rejection。
        print(f"Running per-frame multi-task diagnosis under {args.precision.upper()}...")
        try:
            case = diagnose_dynamic_case(
                model, frames, frame_indices, config, calibrator, args.precision, device)
        except DynamicCaseError as e:
            save_rejection_json(args.output_dir, {"status": "ERROR", "reason": e.reason,
                                                  **e.diagnostic})
            return
        per_frame = case["per_frame"]
        agg = case["agg"]
        kf_preds = case["kf_preds"]
        kf_frame_np = case["kf_frame_np"]
        kf_tensor = case["kf_tensor"]
        failed_frames = case["failed_frames"]
        breakdown = case["breakdown"]
        mask_resized = case["mask_resized"]

        # 5. 关键帧报告
        report_txt = generate_report(
            image_id=f"{os.path.basename(args.input_video)}#frame{agg['keyframe_frame_index']}",
            patient_id="ANONYMOUS_PACS_PATIENT" if is_dicom else "ANONYMOUS",
            breakdown=breakdown, malignancy_prob=kf_preds["mal_prob"])
        report_txt += (
            f"\n\n  [动态图聚合] 抽样 {len(frames)} 帧、成功诊断 {agg['n_frames']} 帧;"
            f"恶性概率 max={agg['mal_prob_max']:.3f} / mean={agg['mal_prob_mean']:.3f};"
            f"关键帧=原始帧 {agg['keyframe_frame_index']}。最终判定按最恶性帧。")
        if failed_frames:
            report_txt += (
                f"\n  [注意] {len(failed_frames)} 帧因数值异常被跳过"
                f"(帧号 {[f['frame_index'] for f in failed_frames]}),聚合仅基于其余帧。")

        # 5b. 关键帧 Grad-CAM(可选,inference_mode 外复用关键帧张量)
        gradcam_done = 0
        if args.enable_gradcam:
            gradcam_done, _ = maybe_save_gradcam(model, kf_tensor, kf_frame_np, args.output_dir)
            if gradcam_done:
                report_txt += "\n  Grad-CAM 可视化解释: 见 gradcam_cls.png(关键帧恶性激活热图)"

        # 关键帧本身是否被审计 flag——mask/报告/Grad-CAM 全出自这一帧,人工复核与
        # 评测方应一眼看见"这份报告出自一张被 flag 的帧",而不是自己拿
        # keyframe_frame_index 去和 flagged_frame_indices 做集合运算(Gate2 审查 P2-2)。
        # 二者同一坐标系:都来自 frame_indices 的原始帧号。
        if not gatekeeper_audit.get("skipped"):
            gatekeeper_audit["keyframe_flagged"] = (
                agg["keyframe_frame_index"] in gatekeeper_audit["flagged_frame_indices"])

        out_json = {
            "status": "PASSED",
            "input_type": "video_cine",
            # n_frames_sampled 取**抽样帧数**(len(frames)),不是 agg["n_frames"]——
            # 后者只统计诊断成功的帧,数值异常跳帧后二者不再相等。分开两个键如实反映
            # "抽了多少 / 成功诊断多少 / 失败多少",不静默少算(无失败时三者退化为原行为)。
            "n_frames_sampled": len(frames),
            "n_frames_diagnosed": agg["n_frames"],
            "n_frames_failed": len(failed_frames),
            "failed_frame_indices": [f["frame_index"] for f in failed_frames],
            "failed_frames": failed_frames,
            "malignancy_probability": agg["mal_prob_max"],
            "malignancy_probability_mean": agg["mal_prob_mean"],
            "is_malignant": agg["final_is_malignant"],
            "keyframe_frame_index": agg["keyframe_frame_index"],
            "tirads_points": breakdown["total_points"],
            "tirads_level": breakdown["tr_level"], "risk_percentage": breakdown["risk"],
            "foci_probabilities": kf_preds["foci_prob"].tolist(),
            "gradcam_generated": gradcam_done,
            "gatekeeper": gatekeeper_audit,
            "per_frame_malignancy": [
                {"frame_index": f["frame_index"], "mal_prob": f["mal_prob"],
                 "malignancy_probabilities": f["preds"]["mal_probs"].tolist()}
                for f in per_frame],
            # 概率分布取自**关键帧**(最恶性帧),与 tirads_*/foci_probabilities 同源;
            # 视频级 malignancy_probability 是跨帧 max 聚合,与关键帧的
            # malignancy_probabilities[1] 一致(max 帧即关键帧)。schema 与静态图共用
            # build_probability_export,两条链路严格一致。
            **build_probability_export(kf_preds),
        }
        save_prediction_artifacts(args.output_dir, mask_resized, report_txt, out_json)
        print(f"--> Video inference completed! Outputs saved to {args.output_dir}")
    except Exception as e:
        print(f"[FATAL] Inference pipeline failed ({type(e).__name__}): {e}")
        save_rejection_json(args.output_dir, {
            "status": "ERROR", "reason": f"{type(e).__name__}: {e}",
        })


if __name__ == "__main__":
    main()
