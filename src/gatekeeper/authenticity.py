import cv2
import hashlib
import os
import numpy as np
import torch
import torch.nn as nn
from skimage.feature import graycomatrix, graycoprops

def compute_ela(img_np, quality=90):
    """
    Computes Error Level Analysis (JPEG re-compression error).
    Synthesized (GAN/Diffusion) images show different reconstruction error distributions.
    """
    # Compress image as JPEG
    _, encoded = cv2.imencode('.jpg', img_np, [cv2.IMWRITE_JPEG_QUALITY, quality])
    decompressed = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    
    # Calculate absolute difference
    ela_map = cv2.absdiff(img_np, decompressed)
    return ela_map

def extract_glcm_features(img_np):
    """
    Extracts Gray-Level Co-occurrence Matrix (GLCM) texture descriptors.
    Helps distinguish biological tissue from physical phantom materials (PVA, agar).
    """
    # Downsample gray levels to 16 for computational efficiency
    img_coarse = (img_np // 16).astype(np.uint8)
    
    # Compute GLCM
    glcm = graycomatrix(img_coarse, distances=[1, 2], angles=[0, np.pi/4], levels=16, symmetric=True, normed=True)
    
    # Extract properties
    contrast = graycoprops(glcm, 'contrast').mean()
    # 零方差(如理想 phantom 平滑图)时 GLCM correlation 除零→NaN,会沿特征向量
    # 传播污染标准化与 CrossEntropy,导致 authenticity 训练整批 NaN。此处守卫。
    correlation = np.nan_to_num(graycoprops(glcm, 'correlation').mean())
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    
    return [contrast, correlation, energy, homogeneity]

def extract_frequency_features(img_np):
    """Extracts high-frequency statistics using Laplacian filtering."""
    laplacian = cv2.Laplacian(img_np, cv2.CV_64F)
    mean = np.mean(laplacian)
    std = np.std(laplacian)
    return [mean, std]


def _naka_m(vals):
    """Nakagami 形状参数 m 的矩估计:m = E[x²]² / Var(x²)。

    标准定义,完全发展散斑(瑞利包络)时 x²~指数分布 → m=1(见单测锁定)。
    均质常数块 var(x²)→0 时 m→∞,clip(0,50) 防数值爆炸污染下游标准化。
    """
    if vals.size < 2:
        return 0.0
    x2 = vals * vals
    omega = float(x2.mean())
    var = float(((x2 - omega) ** 2).mean())
    m = (omega * omega) / (var + 1e-6)
    return float(np.clip(m, 0.0, 50.0))


def extract_nakagami_features(img_np):
    """一阶散斑统计:双纳卡甘米(Double Nakagami)包络参数。

    物理依据(预研报告 Q2):真人体组织是非随机、有限数量、有微观取向的散射体群;
    物理体模(PVA/琼脂)基质均质。Nakagami 形状参数 m 用矩估计闭式求解
    (仅一/二阶矩,开销极低):m = E[x²]² / Var(x²)。

    ⚠ 方向与适用域(经 Gate2 审查澄清,诚实标注):
      • m 定义在**线性回波包络**上;此处输入是 B-mode **对数压缩后的灰度**,
        故 m 为近似量,"m=1↔瑞利"的物理解释在灰度域只近似成立。真实数据
        (2026-08-01)到位后应评估是否做逆对数恢复包络域再估 m。
      • 理论:完全发展散斑(瑞利)m=1(`test_naka_m_rayleigh_is_one` 锁定)。
      • 本仓**合成**体模 `make_phantom_like` 经强平滑 → 灰度方差小 → m 偏大(趋 clip);
        真实组织有结构纹理 → m 适中且空间异质。故当前合成域判别信号为
        「体模 m 偏高、真实 m 适中」,与真实体模(有散斑、m≈1)不同——这是合成
        训练域的特性,非真实物理结论,真训时须用官方真假样本重标定方向。

    双纳卡甘米:按强度中位数分高/低两介质各估一个 m(m1/m2),捕获活体「正常实质 +
    胶原血管束」的双峰非均匀结构。注:不再返回 w_low —— 中位数分组使其恒≈0.5、
    零判别力(Gate2 认定死特征,已删)。

    Returns: [m_global, m1_low, m2_high] 3 维。
    """
    x = img_np.astype(np.float32).reshape(-1)
    m_global = _naka_m(x)
    med = float(np.median(x))
    m1 = _naka_m(x[x <= med])
    m2 = _naka_m(x[x > med])
    return [m_global, m1, m2]


def extract_tai_features(img_np, win=32):
    """二阶空间各向异性:局部结构张量各向异性指数(TAI)。

    物理依据(预研报告 Q2):真人体甲状腺实质/颈前肌群在不同方向上纹理呈强各向
    异性(轴向依赖),体模各切面均匀圆滑、各向同性。对结构张量
        J = [[ΣIx², ΣIxIy],[ΣIxIy, ΣIy²]]
    求特征值 λ1≥λ2,各向异性 TAI = (λ1-λ2)/(λ1+λ2);闭式等价
        TAI = √((Jxx-Jyy)² + 4·Jxy²) / (Jxx+Jyy)。
    真实组织 TAI 高且空间异质,体模全图平均 TAI≈0。TAI 对增益(亮度)线性变化不敏感
    (分子分母同阶),声学物理鲁棒性强。

    Returns: [global_tai, local_tai_mean] 2 维。
    """
    img = cv2.GaussianBlur(img_np.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = gx * gx, gy * gy, gx * gy

    # 全局结构张量(整图聚合)
    sxx, syy, sxy = float(jxx.sum()), float(jyy.sum()), float(jxy.sum())
    global_tai = np.sqrt((sxx - syy) ** 2 + 4.0 * sxy * sxy) / (sxx + syy + 1e-6)

    # 局部 TAI:box 滤波在 win×win 窗内聚合张量分量,逐点求各向异性后取均值
    k = (win, win)
    bxx = cv2.boxFilter(jxx, -1, k, normalize=False)
    byy = cv2.boxFilter(jyy, -1, k, normalize=False)
    bxy = cv2.boxFilter(jxy, -1, k, normalize=False)
    local_tai = np.sqrt((bxx - byy) ** 2 + 4.0 * bxy * bxy) / (bxx + byy + 1e-6)
    return [float(np.nan_to_num(global_tai)), float(np.nan_to_num(local_tai.mean()))]


def extract_authenticity_features(img_np):
    """聚合 13 维真实性特征:纹理 + 频域 + ELA + 散斑物理统计。

    维度构成(顺序固定,改动须同步 authenticity_trainer.FEAT_DIM 与测试断言):
      [0:4]   GLCM   contrast / correlation / energy / homogeneity
      [4:6]   freq   Laplacian mean / std
      [6:8]   ELA    JPEG 重压残差 mean / std
      [8:11]  DND    双纳卡甘米 m_global / m1 / m2          ← 体模 vs 真实(一阶散斑)
      [11:13] TAI    结构张量各向异性 global / local_mean    ← 体模 vs 真实(二阶各向异性)
    前 8 维区分 GAN/合成(频域+重压指纹),后 5 维区分物理体模(声学物理本质)。
    (原 DND w_low 维经 Gate2 认定为恒≈0.5 死特征已删,14→13。)
    """
    glcm_feat = extract_glcm_features(img_np)
    freq_feat = extract_frequency_features(img_np)

    ela = compute_ela(img_np)
    ela_mean = np.mean(ela)
    ela_std = np.std(ela)

    naka_feat = extract_nakagami_features(img_np)
    tai_feat = extract_tai_features(img_np)

    features = (
        glcm_feat + freq_feat + [ela_mean, ela_std] + naka_feat + tai_feat
    )
    # 显式限定 inf 替换值(默认 1.7e308 会在 trainer 标准化时爆掉)
    return np.nan_to_num(
        np.array(features, dtype=np.float32),
        nan=0.0, posinf=50.0, neginf=-50.0,
    )

class AuthenticityClassifier(nn.Module):
    """
    A simple, robust MLP classifier that categorizes images as:
    0: Real Human Scan
    1: Physical Phantom
    2: Synthetic (GAN/Diffusion)

    input_dim 默认 13:8 维取证特征(GLCM/freq/ELA)+ 5 维散斑物理(DND 3 + TAI 2)。
    改维度须同步 extract_authenticity_features 与 authenticity_trainer.FEAT_DIM。
    """
    def __init__(self, input_dim=13, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

class AuthenticityModule:
    """High-level class wrapping feature extraction and prediction."""
    # SHA256 of trusted authenticity_model.pth (更新训练后需同步此哈希)
    EXPECTED_CKPT_SHA256 = None  # 首次训练后设置;None 时跳过校验(开发态)

    def __init__(self):
        self.model = AuthenticityClassifier()
        self.model.eval()
        # 训练集特征标准化参数(随权重一起加载);None 时表示旧格式权重,跳过标准化
        self.feat_mean = None
        self.feat_std = None

    def load_weights(self, path):
        if not os.path.exists(path):
            # 2026-07-10 Gate2审查发现：此前无 else 分支，权重文件不存在时方法
            # 直接返回，模型停留在 __init__ 的随机初始化态，零日志零标志位。
            # predict() 之后照常对随机 logits 做 softmax+argmax，返回合法的
            # 0/1/2——与真实分类结果在接口上完全无法区分，真实性检测器悄悄
            # 变成随机分类器，一票否决门失效或随机误杀真实图像，且无人能发现。
            # 与本次训练管道"strict=False静默吞key→随机初始化"同源，触发点从
            # "key不匹配"换成"文件不存在"。gatekeeper 是安全关键路径，硬报错
            # 交给上层转 ERROR result.json，不允许静默退化为随机分类继续放行。
            raise RuntimeError(
                f"[FATAL] AuthenticityModule 权重文件不存在: {path}\n"
                "真实性检测是 gatekeeper 一票否决链路的一环，缺权重不能静默退化为"
                "随机初始化分类——继续跑会让虚假/合成图像有几率被误判为真实而放行，"
                "也可能把真实图像随机误杀。"
            )
        # 完整性校验:防御 pickle 任意代码执行攻击(weights_only=False 必须配套)
        if self.EXPECTED_CKPT_SHA256 is not None:
            with open(path, 'rb') as f:
                actual_sha = hashlib.sha256(f.read()).hexdigest()
            if actual_sha != self.EXPECTED_CKPT_SHA256:
                raise RuntimeError(
                    f"Checkpoint {path} integrity check failed.\n"
                    f"Expected SHA256: {self.EXPECTED_CKPT_SHA256}\n"
                    f"Actual SHA256:   {actual_sha}\n"
                    "Possible causes: file corruption, malicious replacement, or "
                    "EXPECTED_CKPT_SHA256 not updated after retraining."
                )
        # weights_only=False:ckpt 含 numpy 标准化统计(feat_mean/feat_std),
        # torch≥2.6 默认 weights_only=True 会拒绝反序列化 numpy 对象而报错。
        # 权重由本项目 authenticity_trainer 自产(可信来源),显式关闭安全限制。
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
            fm, fs = ckpt.get("feat_mean"), ckpt.get("feat_std")
            self.feat_mean = None if fm is None else np.asarray(fm, dtype=np.float32)
            self.feat_std = None if fs is None else np.asarray(fs, dtype=np.float32)
        else:
            # 兼容旧的纯 state_dict 格式(无标准化参数)
            self.model.load_state_dict(ckpt)

    def predict(self, img_np):
        features = extract_authenticity_features(img_np)
        # 应用与训练一致的特征标准化(否则训练/推理分布不一致,分类失效)
        if self.feat_mean is not None and self.feat_std is not None:
            features = (features - self.feat_mean) / (self.feat_std + 1e-6)
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(feat_tensor)
            prob = torch.softmax(logits, dim=1).numpy()[0]
            pred = np.argmax(prob)
        return int(pred), prob
