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
        # sorted():os.listdir 顺序取决于文件系统/inode,跨容器 rebuild、跨平台
        # (Win 开发机 vs Linux 容器)会变——不排序则被选入去重种子库的前 200 张
        # 可能不同,导致同一张边界重复图 is_duplicate 判定跨环境不一致(踩「宁可漏检
        # 不可误杀」红线)。排序后种子集合确定,判定可复现。
        train_samples = sorted(os.listdir(config["data"]["train_path"]))
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


def build_auth_module(weights_path):
    """构建并加载 AuthenticityModule，供视频逐帧复用（只加载一次权重）。

    单图 predict.py 无需调用——run_gatekeeper_audit 内部按需懒建；
    视频 video_predict.py 在帧循环前调用此函数，传入 auth_module 参数。
    """
    auth_module = AuthenticityModule()
    auth_weights_path = os.path.join(
        os.path.dirname(weights_path), "authenticity_model.pth"
    )
    auth_module.load_weights(auth_weights_path)
    return auth_module


def build_tamper_detector(config):
    """构建 CatNetTamperingDetector，供视频逐帧复用（避免每帧重探 weights 文件）。

    单图 predict.py 无需调用——run_gatekeeper_audit 内部按需懒建；
    视频 video_predict.py 在帧循环前调用此函数，传入 tamper_detector 参数。
    """
    return CatNetTamperingDetector(
        catnet_weights=config["gatekeeper"].get("catnet_weights"),
        bag_hard_reject=config["gatekeeper"].get("bag_hard_reject", False),
    )


def run_gatekeeper_audit(img_np, config, device, weights_path, output_dir,
                         deduplicator=None, auth_module=None,
                         tamper_detector=None):
    """Run full gatekeeper audit (authenticity, dedup, tampering).

    Args:
        deduplicator:    可选，预建 Deduplicator（视频复用，省去重复播种）。
        auth_module:     可选，预建 AuthenticityModule（视频复用，省去重复加载权重）。
        tamper_detector: 可选，预建 CatNetTamperingDetector（视频复用，省去重复文件探测）。
        三者均为 None 时内部即时构建——单图路径行为不变，向后兼容。

    Returns:
        (passed: bool, rejection_json: dict|None)
    """
    print("Running Gatekeeper Audit...")
    if auth_module is None:
        auth_module = build_auth_module(weights_path)
    # 权重缺失/损坏仍在 AuthenticityModule.load_weights 内硬报错(未改动,红线保留)。
    is_phantom, auth_probs = auth_module.predict(img_np)

    if deduplicator is None:
        deduplicator = build_seeded_deduplicator(config, device)

    is_duplicate, dup_reason = deduplicator.is_duplicate(img_np)
    # CAT-Net v2 取证检测器:detect() = 深度CAT-Net ∪ BAG相位(默认软) ∪ 几何(恒软)。
    # 缺权重时降级 dct-bag(纯 NumPy)。
    if tamper_detector is None:
        tamper_detector = build_tamper_detector(config)
    is_tampered, _tam_mask, _tam_bbox = tamper_detector.detect(img_np)

    # authenticity 软 flag(2026-07-12,统一软flag制度):authenticity_trainer 的
    # phantom/synthetic 训练标签 100% 来自 synthesis.py 程序化合成代理,从未见过
    # 真实体模/真实AI合成图;代码自陈 Nakagami/TAI 判别方向"是合成训练域的特性,
    # 非真实物理结论"(authenticity.py 的 extract_nakagami_features docstring);
    # 且 argmax 无任何置信度门槛。全仓零真实图像 FPR 实测(对比 tampering 有 359
    # 张真实图 0%/27.3% 的明确数字)。故不再作一票否决,仅记录审计,不驱动 REJECT。
    # 恢复硬拒绝的唯一闸门:官方真实干净 holdout 上预注册、带置信区间的 FPR 验收
    # 达标(见 docs/P2红线修复实施计划.md B1、docs/P2启动定论_Opus复核.md §五.5)。
    if is_phantom == 1:
        print("--> [Gatekeeper 软 flag] authenticity: Physical Phantom 疑似(不拒绝,仅记录)", flush=True)
    elif is_phantom == 2:
        print("--> [Gatekeeper 软 flag] authenticity: GAN/Synthetic 疑似(不拒绝,仅记录)", flush=True)

    audit_failed = is_duplicate or is_tampered

    if audit_failed:
        rejections = []
        if is_duplicate:
            rejections.append(f"Duplicate image: {dup_reason}")
        if is_tampered:
            # 2026-07-12 code-review(Angle A)发现:此前坍缩成笼统的
            # "hard-reject path"丢失了具体命中源(CatNet深度模型 vs BAG,
            # 二者可信度/验证量级截然不同)。改用 CatNetTamperingDetector 新增的
            # last_hard_reject_source 拼更精确的文案;tamper_detector 为向后兼容
            # 保留的旧对象(无该属性)时退化为原通用文案。
            source = getattr(tamper_detector, "last_hard_reject_source", None)
            if source == "catnet":
                rejections.append("Tampering detected (CAT-Net deep model hard-reject)")
            elif source == "bag_phase":
                rejections.append("Tampering detected (BAG phase inconsistency hard-reject)")
            else:
                rejections.append("Tampering detected (hard-reject path)")
        # phantom/synthetic 已降级为软flag、不独立驱动 REJECTED,但若本次审计因
        # 去重/篡改被拒绝,仍把 phantom/synthetic 的软flag信息附加进 reason——
        # 否则(2026-07-12 code-review Angle B 发现)is_phantom/is_gan 字段虽仍
        # 保留在 JSON 里,但人类可读的 reason 文本会静默丢失这条并发信号。
        if is_phantom == 1:
            rejections.append("(soft-flag: Physical Phantom suspected)")
        elif is_phantom == 2:
            rejections.append("(soft-flag: GAN/Synthetic suspected)")

        reason_str = ", ".join(rejections)
        print(f"--> [REJECTED] Gatekeeper audit failed: {reason_str}")

        rejection_json = {
            "status": "REJECTED",
            "reason": reason_str,
            # is_phantom/is_gan 保留为审计字段(信息记录,不再驱动 REJECT)
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
