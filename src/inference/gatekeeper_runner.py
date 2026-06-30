"""Gatekeeper audit pipeline — production inference only (NOT evaluate)."""

import os
import cv2

from src.gatekeeper.authenticity import AuthenticityModule
from src.gatekeeper.deduplication import Deduplicator
from src.gatekeeper.tampering_catnet import CatNetTamperingDetector


def build_seeded_deduplicator(config, device):
    """构建并用训练集种子初始化 Deduplicator。

    抽出此函数是为了让视频/cine 推理**只播种一次**、跨帧复用——否则逐帧调用
    `run_gatekeeper_audit` 会对每帧重复读取并 ResNet 编码最多 200 张训练图,
    O(帧数×200) 的冗余严重拖慢延迟(赛道有推理时延自动评分)。
    """
    # cosine_threshold 缺省传 None(而非 0.95)——否则 0.95 兜底会顶替 None 锁死
    # 按后端自适应/映射逻辑(reviewer 必修1)。config 可给标量(两后端同值)或
    # 映射 {sscd:0.86, resnet18:0.995};缺该键则交由 Deduplicator 取后端推荐值。
    deduplicator = Deduplicator(
        device=device,
        phash_threshold=config["gatekeeper"]["phash_threshold"],
        cosine_threshold=config["gatekeeper"].get("cosine_threshold"),
        sscd_weights=config["gatekeeper"].get("sscd_weights"),
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
    # CAT-Net v2 取证检测器:detect() = 深度CAT-Net ∪ BAG相位 ∪ 几何(copy-move/splice)
    # 三路并集,缺权重时降级 dct-bag(纯 NumPy)。函数内构造:无权重时仅 3 次
    # os.path.isfile 探测,廉价;视频逐帧若需复用可后续提为参数。
    tamper_detector = CatNetTamperingDetector(
        catnet_weights=config["gatekeeper"].get("catnet_weights")
    )
    is_tampered, _tam_mask, _tam_bbox = tamper_detector.detect(img_np)

    audit_failed = (is_phantom > 0) or is_duplicate or is_tampered

    if audit_failed:
        rejections = []
        if is_phantom == 1:
            rejections.append("Physical Phantom detected")
        if is_phantom == 2:
            rejections.append("GAN/Synthetic image detected")
        if is_duplicate:
            rejections.append(f"Duplicate image: {dup_reason}")
        if is_tampered:
            rejections.append("Tampering detected (copy-move/splicing/blend)")

        reason_str = ", ".join(rejections)
        print(f"--> [REJECTED] Gatekeeper audit failed: {reason_str}")

        rejection_json = {
            "status": "REJECTED",
            "reason": reason_str,
            "is_phantom": int(is_phantom == 1),
            "is_gan": int(is_phantom == 2),
            "is_duplicate": int(is_duplicate),
            "is_tampered": int(is_tampered),
        }
        return False, rejection_json

    print(
        "--> [PASSED] Gatekeeper audit successfully verified image authenticity and normalization."
    )
    return True, None
