"""Gatekeeper audit pipeline — production inference only (NOT evaluate)."""

import os
import cv2

from src.gatekeeper.authenticity import AuthenticityModule
from src.gatekeeper.deduplication import Deduplicator
from src.gatekeeper.tampering import detect_copy_move, detect_splicing


def build_seeded_deduplicator(config, device):
    """构建并用训练集种子初始化 Deduplicator。

    抽出此函数是为了让视频/cine 推理**只播种一次**、跨帧复用——否则逐帧调用
    `run_gatekeeper_audit` 会对每帧重复读取并 ResNet 编码最多 200 张训练图,
    O(帧数×200) 的冗余严重拖慢延迟(赛道有推理时延自动评分)。
    """
    deduplicator = Deduplicator(
        device=device,
        phash_threshold=config["gatekeeper"]["phash_threshold"],
        cosine_threshold=config["gatekeeper"].get("cosine_threshold", 0.95),
    )
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
    return deduplicator


def run_gatekeeper_audit(img_np, config, device, weights_path, output_dir,
                         deduplicator=None):
    """Run full gatekeeper audit (authenticity, dedup, tampering).

    Args:
        deduplicator: 可选,预先播种好的 Deduplicator(视频逐帧复用以省去重复播种)。
            为 None 时按训练集即时构建(单图路径行为不变,向后兼容)。

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

    if deduplicator is None:
        deduplicator = build_seeded_deduplicator(config, device)

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
