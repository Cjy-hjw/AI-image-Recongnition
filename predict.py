import os
import argparse
import json
import yaml
import cv2
import torch
import numpy as np

from src.models.multi_task_model import MultiTaskThyroidModel
from src.gatekeeper.authenticity import AuthenticityModule
from src.gatekeeper.deduplication import Deduplicator
from src.gatekeeper.tampering import detect_copy_move, detect_splicing
from src.utils.grad_cam import GradCAM, overlay_heatmap
from src.utils.report_generator import calculate_tirads_points, generate_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True, help="Path to input B-mode ultrasound image (PNG/JPG)")
    parser.add_argument("--config", type=str, default="configs/thyroid_config.yaml")
    parser.add_argument("--weights", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="data/output")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config["environment"]["device"] if torch.cuda.is_available() else "cpu")
    
    # 1. Load input image
    img_np = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
    if img_np is None:
        print(f"Error: Could not read image from {args.input_image}")
        return
        
    # --- 2. Run Gatekeeper Pipeline ---
    print("Running Gatekeeper Audit...")
    auth_module = AuthenticityModule()
    auth_weights_path = os.path.join(os.path.dirname(args.weights), "authenticity_model.pth")
    auth_module.load_weights(auth_weights_path)
    is_phantom, auth_probs = auth_module.predict(img_np)
    
    deduplicator = Deduplicator(
        phash_threshold=config["gatekeeper"]["phash_threshold"],
        cosine_threshold=config["gatekeeper"].get("cosine_threshold", 0.95)
    )
    # Seed deduplicator with train database if it exists
    if os.path.exists(config["data"]["train_path"]):
        for sample_dir in os.listdir(config["data"]["train_path"]):
            p = os.path.join(config["data"]["train_path"], sample_dir, "image.png")
            if os.path.exists(p):
                train_img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if train_img is not None:
                    deduplicator.add_image(train_img)
                    
    is_duplicate, dup_reason = deduplicator.is_duplicate(img_np)
    is_copy_move, cm_mask, cm_bbox = detect_copy_move(img_np)
    is_spliced, sp_mask, sp_bbox = detect_splicing(img_np)
    
    audit_failed = (is_phantom > 0) or is_duplicate or is_copy_move or is_spliced
    
    if audit_failed:
        rejections = []
        if is_phantom == 1: rejections.append("Physical Phantom detected")
        if is_phantom == 2: rejections.append("GAN/Synthetic image detected")
        if is_duplicate: rejections.append(f"Duplicate image: {dup_reason}")
        if is_copy_move: rejections.append("Copy-move tampering detected")
        if is_spliced: rejections.append("Splicing/editing detected")
        
        reason_str = ", ".join(rejections)
        print(f"--> [REJECTED] Gatekeeper audit failed: {reason_str}")
        
        # Save rejection metadata
        out_json = {
            "status": "REJECTED",
            "reason": reason_str,
            "is_phantom": int(is_phantom == 1),
            "is_gan": int(is_phantom == 2),
            "is_duplicate": int(is_duplicate),
            "is_tampered": int(is_copy_move or is_spliced)
        }
        with open(os.path.join(args.output_dir, "result.json"), "w") as f:
            json.dump(out_json, f, indent=2)
        return
        
    print("--> [PASSED] Gatekeeper audit successfully verified image authenticity and normalization.")
    
    # --- 3. Run Multi-task Model Inference ---
    print("Running Multi-task Diagnosis...")
    model = MultiTaskThyroidModel(backbone_name=config["model"]["backbone"], pretrained=False).to(device)
    if os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print("[WARNING] Trained checkpoint not found. Using default initialized weights.")
        
    model.eval()
    
    # Preprocess image
    img_resized = cv2.resize(img_np, (config["data"]["img_size"], config["data"]["img_size"]))
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0
    
    with torch.no_grad():
        outputs = model(img_tensor)
        
    # Process outputs
    # 1. Nodule segmentation mask
    pred_mask = torch.sigmoid(outputs["mask"])[0, 0].cpu().numpy()
    mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask_binary, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 2. Category Predictions
    comp_idx = torch.argmax(outputs["composition"], dim=1)[0].item()
    echo_idx = torch.argmax(outputs["echogenicity"], dim=1)[0].item()
    shape_idx = torch.argmax(outputs["shape"], dim=1)[0].item()
    margin_idx = torch.argmax(outputs["margin"], dim=1)[0].item()
    foci_prob = torch.sigmoid(outputs["foci"])[0].cpu().numpy()
    
    # Calculate TI-RADS breakdown
    breakdown = calculate_tirads_points(comp_idx, echo_idx, shape_idx, margin_idx, foci_prob)
    
    # 3. Malignancy prediction
    mal_prob = torch.softmax(outputs["malignancy"], dim=1)[0, 1].item()
    
    # Generate structured clinical report
    image_name = os.path.basename(args.input_image)
    report_txt = generate_report(
        image_id=image_name,
        patient_id="ANONYMOUS",
        breakdown=breakdown,
        malignancy_prob=mal_prob
    )
    
    # 4. Generate Grad-CAM activations
    gradcam_saved = False
    gradcam = None
    try:
        target_layer = model.encoder.layer4[-1]
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate_heatmap(img_tensor, target_class=1)
        overlay = overlay_heatmap(img_np, heatmap)
        cv2.imwrite(os.path.join(args.output_dir, "gradcam_activation.png"), overlay)
        gradcam_saved = True
    except Exception as e:
        print(f"[Grad-CAM Warning] Could not generate heatmap: {e}")
    finally:
        if gradcam is not None:
            gradcam.remove_hooks()
        
    # Save files
    cv2.imwrite(os.path.join(args.output_dir, "mask_predicted.png"), mask_resized)
    
    with open(os.path.join(args.output_dir, "clinical_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)
        
    out_json = {
        "status": "PASSED",
        "malignancy_probability": mal_prob,
        "is_malignant": int(mal_prob >= 0.5),
        "tirads_points": breakdown["total_points"],
        "tirads_level": breakdown["tr_level"],
        "risk_percentage": breakdown["risk"],
        "foci_probabilities": foci_prob.tolist(),
        "gradcam_generated": int(gradcam_saved)
    }
    with open(os.path.join(args.output_dir, "result.json"), "w") as f:
        json.dump(out_json, f, indent=2)
        
    print(f"--> Inference completed! Outputs saved to {args.output_dir}")

if __name__ == "__main__":
    main()
