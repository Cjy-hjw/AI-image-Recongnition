import argparse
import os
import yaml
import cv2
import torch
from src.inference.io import load_input_image
from src.inference.gatekeeper_runner import run_gatekeeper_audit
from src.inference.model_runner import build_model, preprocess_image, run_forward
from src.inference.postprocess import decode_predictions, build_tirads_breakdown
from src.inference.export import save_prediction_artifacts, save_rejection_json
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
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    device = torch.device(
        config["environment"]["device"] if torch.cuda.is_available() else "cpu")

    # 1. Load & anonymize
    img_np, is_dicom, error_msg = load_input_image(args.input_image, args.output_dir)
    if img_np is None:
        print(f"Error: {error_msg}")
        return

    # 2. Gatekeeper audit (predict only)
    passed, rejection_json = run_gatekeeper_audit(
        img_np, config, device, args.weights, args.output_dir)
    if not passed:
        save_rejection_json(args.output_dir, rejection_json)
        return

    # 3. Model inference
    print(f"Running Multi-task Diagnosis under {args.precision.upper()} precision...")
    model = build_model(config, args.weights, device)
    img_tensor = preprocess_image(img_np, config["data"]["img_size"], device)
    outputs = run_forward(model, img_tensor, args.precision, device)

    # 4. Postprocess
    preds = decode_predictions(outputs)
    mask_resized = cv2.resize(
        preds["mask_binary"], (img_np.shape[1], img_np.shape[0]),
        interpolation=cv2.INTER_NEAREST)
    breakdown = build_tirads_breakdown(
        preds["comp_idx"], preds["echo_idx"],
        preds["shape_idx"], preds["margin_idx"], preds["foci_prob"])

    # 5. Report & export
    image_name = os.path.basename(args.input_image)
    report_txt = generate_report(
        image_id=image_name,
        patient_id="ANONYMOUS_PACS_PATIENT" if is_dicom else "ANONYMOUS",
        breakdown=breakdown, malignancy_prob=preds["mal_prob"])
    out_json = {
        "status": "PASSED", "malignancy_probability": preds["mal_prob"],
        "is_malignant": int(preds["mal_prob"] >= 0.5),
        "tirads_points": breakdown["total_points"],
        "tirads_level": breakdown["tr_level"], "risk_percentage": breakdown["risk"],
        "foci_probabilities": preds["foci_prob"].tolist(), "gradcam_generated": 0,
    }
    save_prediction_artifacts(args.output_dir, mask_resized, report_txt, out_json)
    print(f"--> Inference completed! Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
