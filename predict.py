import argparse
import os
import cv2
import torch
from src.inference.io import load_input_image
from src.inference.gatekeeper_runner import run_gatekeeper_audit
from src.inference.model_runner import build_model, load_config, preprocess_image, run_forward
from src.inference.postprocess import decode_predictions, build_tirads_breakdown
from src.inference.calibration import TemperatureScaler
from src.inference.export import save_prediction_artifacts, save_rejection_json, maybe_save_gradcam
from src.utils.report_generator import generate_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_image", type=str, required=True,
                   help="Path to input B-mode ultrasound image (PNG/JPG/DCM)")
    p.add_argument("--config", type=str, default="configs/thyroid_config.yaml")
    p.add_argument("--weights", type=str, default="checkpoints/best_model.pth")
    p.add_argument("--output_dir", type=str, default="data/output")
    p.add_argument("--precision", type=str, default="fp16",
                   choices=["fp32", "fp16", "bf16"], help="Inference precision")
    p.add_argument("--enable_gradcam", action="store_true",
                   help="生成 Grad-CAM 恶性热图(可解释性);需在 inference_mode 外重跑前向+反向,略增耗时")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 配置加载单独一层 try：路径不存在/YAML 格式错误/键缺失都必须兜成 ERROR
    # result.json，不能裸崩溃（2026-07-11 修复，见 model_runner.load_config
    # 的 docstring）。os.makedirs 本身不纳入 try——它一旦失败，output_dir 都
    # 不存在，没有地方能写 result.json，只能任其以非零 exit code 退出（这是
    # 部署方控制的启动参数，不是输入数据能触达的攻击面，风险可接受）。
    try:
        config = load_config(args.config)
        device = torch.device(
            config["environment"]["device"] if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"[FATAL] Config loading failed ({type(e).__name__}): {e}")
        save_rejection_json(args.output_dir, {
            "status": "ERROR", "reason": f"{type(e).__name__}: {e}",
        })
        return

    # 1. Load & anonymize
    img_np, is_dicom, error_msg = load_input_image(args.input_image, args.output_dir)
    if img_np is None:
        print(f"Error: {error_msg}")
        save_rejection_json(args.output_dir, {"status": "ERROR", "reason": error_msg})
        return

    # 2-5. Gatekeeper审计 + 模型推理 + 后处理 + 报告导出。整体包一层 except
    # Exception：离线推理容器按提交产出 JSON 是硬契约，裸崩溃会让该次提交零
    # 输出（比预测错误更差，大概率直接判零分）。gatekeeper 审计此前在 try 外
    # 面——权重缺失等场景硬报错时会让 predict.py 整进程崩溃、零输出（本次
    # Gate2审查同步给 authenticity.py 加了硬报错，若不把这段纳入 try 会自相
    # 矛盾：越权威的失败越容易把容器打成裸崩溃）。只兜 build_model() 一处也
    # 不够——torch.load 遇损坏 checkpoint 会抛 pickle.UnpicklingError（不是
    # RuntimeError 子类，原 except RuntimeError 接不住）、CUDA OOM、
    # decode_predictions/cv2/报告生成任一步也都可能因边界数据崩溃。异常类型名
    # 随 str(e) 一起落盘，不靠拆分 except 分支也能区分"预期权重不匹配"和
    # "其它意外故障"（2026-07-10 Gate2审查发现）。
    print(f"Running Multi-task Diagnosis under {args.precision.upper()} precision...")
    try:
        # 2. Gatekeeper audit (predict only)
        passed, rejection_json = run_gatekeeper_audit(
            img_np, config, device, args.weights, args.output_dir)
        if not passed:
            save_rejection_json(args.output_dir, rejection_json)
            return

        model = build_model(config, args.weights, device)
        img_tensor = preprocess_image(img_np, config["data"]["img_size"], device)
        outputs = run_forward(model, img_tensor, args.precision, device)

        cal_path = config.get("calibration", {}).get("param_file", "checkpoints/calibration.json")
        calibrator = TemperatureScaler.from_file(cal_path)
        preds = decode_predictions(outputs, calibrator=calibrator)
        mask_resized = cv2.resize(
            preds["mask_binary"], (img_np.shape[1], img_np.shape[0]),
            interpolation=cv2.INTER_NEAREST)
        breakdown = build_tirads_breakdown(
            preds["comp_idx"], preds["echo_idx"],
            preds["shape_idx"], preds["margin_idx"], preds["foci_prob"])

        image_name = os.path.basename(args.input_image)
        report_txt = generate_report(
            image_id=image_name,
            patient_id="ANONYMOUS_PACS_PATIENT" if is_dicom else "ANONYMOUS",
            breakdown=breakdown, malignancy_prob=preds["mal_prob"])

        # Grad-CAM 可解释热图(可选)。必须在 run_forward 的 inference_mode 块之外,
        # 用 preprocess_image 产出的普通张量重跑前向+反向(grad_cam 内部自带前向),
        # 否则 inference tensor 无法参与 autograd。失败不阻断主流程(save_gradcam_overlay 已内部兜底)。
        gradcam_done = 0
        if args.enable_gradcam:
            gradcam_done, _ = maybe_save_gradcam(model, img_tensor, img_np, args.output_dir)
            if gradcam_done:
                report_txt += "\n\n  Grad-CAM 可视化解释: 见 gradcam_cls.png(恶性激活热图)"

        out_json = {
            "status": "PASSED", "input_type": "static_image",
            "malignancy_probability": preds["mal_prob"],
            "is_malignant": int(preds["mal_prob"] >= 0.5),
            "tirads_points": breakdown["total_points"],
            "tirads_level": breakdown["tr_level"], "risk_percentage": breakdown["risk"],
            "foci_probabilities": preds["foci_prob"].tolist(), "gradcam_generated": gradcam_done,
        }
        save_prediction_artifacts(args.output_dir, mask_resized, report_txt, out_json)
        print(f"--> Inference completed! Outputs saved to {args.output_dir}")
    except Exception as e:
        print(f"[FATAL] Inference pipeline failed ({type(e).__name__}): {e}")
        save_rejection_json(args.output_dir, {
            "status": "ERROR", "reason": f"{type(e).__name__}: {e}",
        })


if __name__ == "__main__":
    main()
