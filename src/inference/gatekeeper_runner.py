"""Gatekeeper audit pipeline — production inference only (NOT evaluate)."""

import os
import cv2

from src.gatekeeper.authenticity import AuthenticityModule
from src.gatekeeper.deduplication import Deduplicator
from src.gatekeeper.tampering import detect_copy_move, detect_splicing


def run_gatekeeper_audit(img_np, config, device, weights_path, output_dir):
    """Run full gatekeeper audit (authenticity, dedup, tampering).

    Returns:
        (passed: bool, rejection_json: dict|None)
    """
    print("Running Gatekeeper Audit...")
    auth_module = AuthenticityModule()
    auth_weights_path = os.path.join(
        os.path.dirname(weights_path), "authenticity_model.pth"
    )
    auth_module.load_weights(auth_weights_path)
    is_phantom, auth_probs = auth_module.predict(img_np)

    deduplicator = Deduplicator(
        device=device,
        phash_threshold=config["gatekeeper"]["phash_threshold"],
        cosine_threshold=config["gatekeeper"].get("cosine_threshold", 0.95),
    )
    # Seed deduplicator database with train database if it exists
    if os.path.exists(config["data"]["train_path"]):
        train_samples = os.listdir(config["data"]["train_path"])
        limit = min(len(train_samples), 200)
        for i in range(limit):
            p = os.path.join(
                config["data"]["train_path"], train_samples[i], "image.png"
            )
            if os.path.exists(p):
                train_img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if train_img is not None:
                    deduplicator.add_image(train_img)

    is_duplicate, dup_reason = deduplicator.is_duplicate(img_np)
    is_copy_move, cm_mask, cm_bbox = detect_copy_move(img_np)
    is_spliced, sp_edge, sp_bbox = detect_splicing(img_np)

    audit_failed = (is_phantom > 0) or is_duplicate or is_copy_move or is_spliced

    if audit_failed:
        rejections = []
        if is_phantom == 1:
            rejections.append("Physical Phantom detected")
        if is_phantom == 2:
            rejections.append("GAN/Synthetic image detected")
        if is_duplicate:
            rejections.append(f"Duplicate image: {dup_reason}")
        if is_copy_move:
            rejections.append("Copy-move tampering detected")
        if is_spliced:
            rejections.append("Splicing/editing detected")

        reason_str = ", ".join(rejections)
        print(f"--> [REJECTED] Gatekeeper audit failed: {reason_str}")

        rejection_json = {
            "status": "REJECTED",
            "reason": reason_str,
            "is_phantom": int(is_phantom == 1),
            "is_gan": int(is_phantom == 2),
            "is_duplicate": int(is_duplicate),
            "is_tampered": int(is_copy_move or is_spliced),
        }
        return False, rejection_json

    print(
        "--> [PASSED] Gatekeeper audit successfully verified image authenticity and normalization."
    )
    return True, None
