"""Online synthesis engine for gatekeeper training/validation.

公开数据集没有"医保骗保假人体/重复/拼接超声"样本,故在 DataLoader 端
按需合成,用于训练 AuthenticityClassifier 与验证 dedup/tampering 检测器。
所有函数接受 `rng`(np.random.RandomState)或 `seed`,确定性可复现。

类别约定(对齐 AuthenticityClassifier):
    0: Real Human Scan      (真人体,不在此合成)
    1: Physical Phantom     make_phantom_like
    2: Synthetic (GAN/扩散) make_synthetic_like
"""

import cv2
import numpy as np


def _as_rng(rng_or_seed):
    """统一接受 RandomState 或 int seed,返回 RandomState。"""
    if isinstance(rng_or_seed, np.random.RandomState):
        return rng_or_seed
    return np.random.RandomState(rng_or_seed)


def make_phantom_like(img_np, rng_or_seed=0):
    """模拟超声体模(PVA/琼脂):匀化散斑、削弱解剖纹理。

    真人组织有复杂解剖分层与声学衰减伪影;体模基质高度均匀。
    手段:强双边/高斯平滑压低 GLCM 对比度、抬高 homogeneity,再叠加
    低幅规则噪声(体模无生理散斑的随机性)。
    """
    rng = _as_rng(rng_or_seed)
    img = img_np.astype(np.float32)

    # 1. 大核高斯平滑抹掉精细解剖纹理
    smoothed = cv2.GaussianBlur(img, (0, 0), sigmaX=rng.uniform(2.5, 4.0))

    # 2. 双边滤波保留大边界但进一步匀化内部
    smoothed = cv2.bilateralFilter(
        smoothed.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75
    )

    # 3. 极低幅噪声(体模噪声远低于生理散斑,不能抵消平滑带来的高 homogeneity)
    reg_noise = rng.normal(0.0, 0.5, size=img.shape).astype(np.float32)
    out = smoothed + reg_noise

    return np.clip(out, 0, 255).astype(np.uint8)


def make_synthetic_like(img_np, rng_or_seed=0):
    """模拟 GAN/扩散合成图:注入频域周期伪影 + 重压缩残差异常。

    合成图在频域/高频噪声分布上遗留非自然周期性伪影。
    手段:叠加正弦周期纹理(频域规则峰)+ 一次激进 JPEG 重压(打乱 ELA 残差)。
    """
    rng = _as_rng(rng_or_seed)
    img = img_np.astype(np.float32)

    # 1. 频域周期伪影:低幅正弦栅格(GAN 上采样棋盘效应的代理)
    h, w = img.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    freq = rng.uniform(0.15, 0.35)
    phase = rng.uniform(0, 2 * np.pi)
    grating = 4.0 * np.sin(2 * np.pi * freq * xx + phase)
    grating += 4.0 * np.sin(2 * np.pi * freq * yy + phase)
    out = img + grating.astype(np.float32)

    # 2. 激进 JPEG 重压缩,制造异常 ELA 残差分布
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    quality = int(rng.randint(40, 65))
    _, enc = cv2.imencode(".jpg", out_u8, [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    if decoded is None:  # 重压缩理论上不会失败,守卫满足类型契约
        return out_u8
    return decoded


def make_near_duplicate(img_np, rng_or_seed=0):
    """生成近重复图:重采样/裁剪/水印/亮度偏移后回填原尺寸。

    医保骗保中同一影像经下采样、局部裁剪、加字符水印、调亮度后重报。
    检测器(dHash + ResNet 余弦)应仍能判为重复。
    """
    rng = _as_rng(rng_or_seed)
    h, w = img_np.shape
    out = img_np.copy()

    # 1. 随机轻裁剪后放大回原尺寸(重采样)
    crop = rng.uniform(0.0, 0.08)
    y0, x0 = int(h * crop), int(w * crop)
    y1, x1 = h - y0, w - x0
    if y1 - y0 > 16 and x1 - x0 > 16:
        out = cv2.resize(out[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. 亮度/对比度偏移
    alpha = rng.uniform(0.9, 1.1)
    beta = rng.uniform(-12, 12)
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # 3. 角落加字符水印(模拟医院/患者信息覆盖)
    if rng.rand() < 0.7:
        text = f"ID{int(rng.randint(1000, 9999))}"
        cv2.putText(
            out, text, (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, 255, 1, cv2.LINE_AA
        )

    return out


def make_copy_move(img_np, rng_or_seed=0, patch_frac=0.18):
    """复制-移动篡改:把一块区域复制粘贴到同图另一处(隔开足够距离)。

    Returns:
        (tampered_img, mask, bbox):
            mask  — uint8 二值,粘贴区域为 255
            bbox  — [x1,y1,x2,y2] 粘贴区域
    """
    rng = _as_rng(rng_or_seed)
    h, w = img_np.shape
    out = img_np.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    ph = max(16, int(h * patch_frac))
    pw = max(16, int(w * patch_frac))
    if ph >= h or pw >= w:
        return out, mask, None

    # 源块位置
    sy = int(rng.randint(0, h - ph))
    sx = int(rng.randint(0, w - pw))
    patch = out[sy:sy + ph, sx:sx + pw].copy()

    # 目标块位置,确保与源块中心距离 >= min_dist
    min_dist = max(ph, pw)
    dy = int(rng.randint(0, h - ph))
    dx = int(rng.randint(0, w - pw))
    for _ in range(20):
        if np.hypot((sy - dy), (sx - dx)) >= min_dist:
            break
        dy = int(rng.randint(0, h - ph))
        dx = int(rng.randint(0, w - pw))

    out[dy:dy + ph, dx:dx + pw] = patch
    mask[dy:dy + ph, dx:dx + pw] = 255
    bbox = [dx, dy, dx + pw, dy + ph]

    return out, mask, bbox


def make_splice(img_np, donor_np, rng_or_seed=0, patch_frac=0.22):
    """拼接篡改:把 donor 图的一块缝合到本图(制造声学边界突变)。

    Returns:
        (spliced_img, mask, bbox)
    """
    rng = _as_rng(rng_or_seed)
    h, w = img_np.shape
    out = img_np.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    # donor 缩放到与本图同尺寸,便于取块
    donor = cv2.resize(donor_np, (w, h), interpolation=cv2.INTER_LINEAR)

    ph = max(16, int(h * patch_frac))
    pw = max(16, int(w * patch_frac))
    if ph >= h or pw >= w:
        return out, mask, None

    sy = int(rng.randint(0, h - ph))
    sx = int(rng.randint(0, w - pw))
    dy = int(rng.randint(0, h - ph))
    dx = int(rng.randint(0, w - pw))

    # 直接硬粘贴(不羽化),刻意保留尖锐拼接边界供 splicing 检测器捕获
    out[dy:dy + ph, dx:dx + pw] = donor[sy:sy + ph, sx:sx + pw]
    mask[dy:dy + ph, dx:dx + pw] = 255
    bbox = [dx, dy, dx + pw, dy + ph]

    return out, mask, bbox
