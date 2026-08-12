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


def make_splice_poisson(img_np, donor_np, rng_or_seed=0, patch_frac=0.22):
    """泊松/羽化拼接:donor 块用 `cv2.seamlessClone` 无缝融入本图。

    与 `make_splice`(硬粘贴、留尖锐缝)互补:泊松融合**消除长直阶跃边界**,
    几何缝检测器(`detect_splicing`)对其无能为力——专供 CAT-Net DCT/BAG 取证
    (`tampering_catnet`)训练/验证:若 donor 与宿主块栅格相位错位,融合后 RGB
    边界看不见,但块伪影栅格仍被破坏。

    Returns:
        (spliced_img, mask, bbox);seamlessClone 不可用则退化为高斯羽化混合。
    """
    rng = _as_rng(rng_or_seed)
    h, w = img_np.shape
    out = img_np.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    donor = cv2.resize(donor_np, (w, h), interpolation=cv2.INTER_LINEAR)
    ph = max(24, int(h * patch_frac))
    pw = max(24, int(w * patch_frac))
    # 守卫:中心采样区间 randint(ph//2+1, h-ph//2-1) 须 low<high,否则 numpy 抛
    # ValueError(不被 except cv2.error 接住)。小图/极端 patch_frac 下 ph 逼近 h 时
    # 区间会塌缩,故要求两轴均留足边距(比单纯 ph>=h 更严)。
    if ph >= h - 2 or pw >= w - 2:
        return out, mask, None

    sy = int(rng.randint(0, h - ph))
    sx = int(rng.randint(0, w - pw))
    # 目标中心需留出半块边距,避免 seamlessClone ROI 越界
    cy = int(rng.randint(ph // 2 + 1, h - ph // 2 - 1))
    cx = int(rng.randint(pw // 2 + 1, w - pw // 2 - 1))

    src_rgb = cv2.cvtColor(donor[sy:sy + ph, sx:sx + pw], cv2.COLOR_GRAY2BGR)
    dst_rgb = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    patch_mask = np.full((ph, pw), 255, dtype=np.uint8)
    try:
        blended = cv2.seamlessClone(
            src_rgb, dst_rgb, patch_mask, (cx, cy), cv2.NORMAL_CLONE
        )
        out = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        # 退化:高斯羽化 alpha 混合(同样无硬缝),保证函数不崩
        y0, x0 = cy - ph // 2, cx - pw // 2
        alpha = cv2.GaussianBlur(patch_mask.astype(np.float32) / 255.0, (0, 0), 5)
        roi = out[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
        src = donor[sy:sy + ph, sx:sx + pw].astype(np.float32)
        out[y0:y0 + ph, x0:x0 + pw] = (
            alpha * src + (1 - alpha) * roi
        ).astype(np.uint8)

    y0, x0 = cy - ph // 2, cx - pw // 2
    mask[y0:y0 + ph, x0:x0 + pw] = 255
    bbox = [x0, y0, x0 + pw, y0 + ph]
    return out, mask, bbox


def inject_nakagami_speckle(img_np, rng_or_seed=0, m_range=(0.6, 1.4)):
    """注入一阶 Nakagami 散斑(乘性),模拟真实超声散斑统计。

    超声包络服从 Nakagami 分布(形状参数 m:m=1 为充分发育 Rayleigh 散斑)。
    Nakagami(m,Ω) 的平方 = Gamma(k=m, θ=Ω/m),故包络 = sqrt(Gamma 抽样)。
    乘性叠加到图像上,使域随机化样本的散斑统计贴近真实声学过程,而非加性高斯。

    Args:
        m_range: 形状参数采样区间,默认 [0.6,1.4] 覆盖欠/充分发育散斑。
    """
    rng = _as_rng(rng_or_seed)
    img = img_np.astype(np.float32)
    m = rng.uniform(*m_range)
    # Nakagami 包络:E[env]≈1 归一,作乘性增益(Ω=1 → Gamma(m,1/m),均值 1)
    gamma = rng.gamma(shape=m, scale=1.0 / m, size=img.shape)
    env = np.sqrt(gamma).astype(np.float32)
    env /= (env.mean() + 1e-6)  # 归一到均值 1,只调散斑纹理不改整体亮度
    out = img * env
    return np.clip(out, 0, 255).astype(np.uint8)


def domain_randomize(img_np, rng_or_seed=0):
    """域随机化:Nakagami 散斑 → 高斯模糊 k∈{3,5,7} → JPEG 重压 Q∈[15,75]。

    缩小合成训练样本与官方真实 JPG 的域差(不同探头/增益/压缩链)。三步按真实
    采集-传输链顺序施加:先生理散斑、再光学/重建模糊、最后有损压缩伪影。
    确定性可复现(同 seed 同输出)。
    """
    rng = _as_rng(rng_or_seed)
    out = inject_nakagami_speckle(img_np, rng)

    # 高斯模糊,核 ∈ {3,5,7}(奇数),sigma 由核自动推断
    k = int(rng.choice([3, 5, 7]))
    out = cv2.GaussianBlur(out, (k, k), 0)

    # JPEG 重压缩 Q∈[15,75],植入块伪影/量化残差
    quality = int(rng.randint(15, 76))
    ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        decoded = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        if decoded is not None:
            return decoded
    return out  # 编解码异常守卫:退回模糊结果,满足 uint8 类型契约
