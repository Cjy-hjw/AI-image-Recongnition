# src/gatekeeper/AGENTS.md - Ultrasound Image Gatekeeping Specifications

This protocol defines the 3-layer gatekeeper pipeline for detecting fake, duplicated, or tampered ultrasound images.

---

## 0. SYNTHESIS ENGINE (`synthesis.py`)

在线合成引擎,为 gatekeeper 基础项(真实性/规范性)提供可训练/可测试数据。全部函数支持 `rng_or_seed` 参数确保确定性可复现。

### 导出函数

| 函数 | 功能 |
|:---|:---|
| `make_phantom_like(img, rng_or_seed)` | 模拟体模:平滑匀化 + 高 GLCM homogeneity |
| `make_synthetic_like(img, rng_or_seed)` | 模拟 GAN/扩散合成:频域周期伪影 + 激进 JPEG 重压 |
| `make_near_duplicate(img, rng_or_seed)` | 近重复:重采样/裁剪/水印/亮度偏移 |
| `make_copy_move(img, rng_or_seed, patch_frac)` | → `(tampered, mask, bbox)` |
| `make_splice(img, donor, rng_or_seed, patch_frac)` | 硬粘贴(留尖锐缝,供几何检测器)→ `(spliced, mask, bbox)` |
| `make_splice_poisson(img, donor, rng_or_seed, patch_frac)` | 泊松/羽化拼接(`cv2.seamlessClone`,无硬缝,供 CAT-Net DCT/BAG 取证;seamlessClone 不可用退化高斯羽化;小图区间守卫)→ `(spliced, mask, bbox)` |
| `inject_nakagami_speckle(img, rng_or_seed, m_range)` | 一阶 Nakagami 乘性散斑(env=√Gamma(m,1/m) 归一均值1),m_range 默认 (0.6,1.4) 覆盖欠/充分发育(m=1 即 Rayleigh) |
| `domain_randomize(img, rng_or_seed)` | 域随机化:Nakagami 散斑 → 高斯模糊 k∈{3,5,7} → JPEG 重压 Q∈[15,75],缩小合成样本与官方真实 JPG 域差 |

### 类别约定

| Label | 含义 |
|:---|:---|
| 0 | 真实图像 |
| 1 | 体模(phantom) |
| 2 | AI 合成(GAN/扩散) |

---

## 1. AUTHENTICITY CLASSIFIER (`authenticity.py`)

- **Feature extraction**: 13 维真实特征:GLCM(4)+ 频域(2)+ ELA(2)+ DND 双纳卡甘米散斑 m_global/m1/m2(3)+ TAI 结构张量各向异性 global/local(2)。前 8 维区分 GAN/合成,后 5 维(DND+TAI)区分物理体模(PVA/琼脂)的声学物理本质。改维度须同步 `authenticity.py`、`authenticity_trainer.FEAT_DIM`、test 断言。
- **训练数据**: 从真实训练图提取 label=0(真实)、`make_phantom_like` 生成 label=1(体模)、`make_synthetic_like` 生成 label=2(合成),特征标准化后 40 epoch 训练。
- **Model**: Shallow MLP 三分类器(`AuthenticityClassifier`)。
- **Weights**: Trained during `train.py` startup, saved to `checkpoints/authenticity_model.pth`. 推理端因 ckpt 含 numpy `feat_mean/feat_std`,torch≥2.6 必须 `torch.load(..., weights_only=False)`;该路径只允许加载本项目自产可信权重。正式训练出固定权重后须同步 `AuthenticityModule.EXPECTED_CKPT_SHA256`,否则 `weights_only=False` 缺少完整性约束会重新暴露 pickle 任意代码执行风险。
- **验证**: 训练结束在验证集报告三类准确率。
- **Red-line**: MLP weights MUST NOT be altered without retraining on physical phantom datasets.

> **历史说明**: 旧版用 `torch.randn` 随机噪声 + 随机标签做假训练(完全无用占位),已于本次重写为真实特征训练流程。

---

## 2. DEDUPLICATION PIPELINE (`deduplication.py`)

Two-stage duplicate detection:

| Stage | Method | Threshold |
|:---|:---|:---|
| **Primary** | 64-bit perceptual hash (pHash) + Hamming distance | Distance ≤ 12 = duplicate |
| **Secondary** | 可插拔深度描述符 + Cosine Similarity | 按后端取阈值(生产 config:SSCD 0.86 / ResNet18 0.995)|

- **后端可插拔(SSCD 优先,降级 ResNet18)**:`FeatureExtractor` 探测 SSCD 离线 TorchScript 权重(`checkpoints/sscd_disc_mixup.torchscript.pt` 或 `~/.cache/torch/hub/checkpoints/`,来源 facebookresearch/sscd-copy-detection,Apache-2.0/MIT,95MB,512 维 GeM,对 ≤80% 局部裁切/水印/色调偏移近似不变)。权重缺失则**降级** torchvision ResNet18,行为与历史一致——保证离线无权重环境下测试/推理零破坏。两后端均输出 **512 维 L2 归一**向量;`feat_dim` 由 init 跑 dummy 前向**实测探测**(reviewer 必修2,防误传 1024 维 disc_large 静默错算),FAISS `IndexFlatIP(feat_dim)` 与 NumPy 双路兼容。
- **按后端阈值映射(守红线)**:`Deduplicator` 的 `cosine_threshold` 接受三态——`None`→取后端推荐(SSCD 0.86 / ResNet18 0.95);`dict`→按 `backend_name` 取值(生产 config `{sscd:0.86, resnet18:0.995}`,缺键回退推荐);标量→两后端同值(向后兼容)。**单一标量无法同时服务两后端**:SSCD 治近重复漏检需 0.86,ResNet18 守一票否决红线「宁可漏检不可误杀真实影像」需 0.995。`gatekeeper_runner` 透传时缺省传 `None`(非 0.95,否则兜底值顶替 None 锁死映射逻辑)。
- **权重交付**:SSCD TorchScript 需赛前手动下载放入 `checkpoints/`;比赛交付为商用离线容器,SSCD 协议合规。

---

## 3. TAMPERING DETECTION (`tampering.py` + `tampering_catnet.py`)

| Attack Type | Detection Method | 当前能力边界 |
|:---|:---|:---|
| **Copy-Move** | Dense block-matching to find duplicate pixel regions | 召回约 20–40%(干净图零误报);合成引擎现可批量生成训练样本供后续迭代提升 |
| **Splicing(硬粘贴)** | 轴对齐长直缝几何判据:Sobel 强阶跃边 → HoughLinesP 找长直线(≥0.14·min(H,W))→ 按 ±10° 分水平/垂直缝 → **要求一条水平缝与一条垂直缝端点在 14px 内角点正交相交**(硬粘贴矩形特征) | 经两轮失败定案:像素/区域方差方案在含结节的干净图 20/20 误报(结节本身即方差异常区),触碰一票否决红线故弃用。现方案干净图 **0/80 零误报**、真实组织拼接召回 **~84%**,放弃纯噪声 donor 等极端对抗样本部分召回以换零误报 |
| **Splicing(羽化/泊松融合)** | `tampering_catnet.py` — CAT-Net v2 风格 DCT/BAG 取证:`block_artifact_energy` 估计 8×8 块伪影栅格(BAG)主导相位,滑窗找局部相位错位区(来自另一 JPEG 的错位粘贴会破坏全图一致栅格,即使 RGB 边界被融合) | 几何缝检测器抓不到「无长直阶跃」的羽化拼接;BAG 取证补此盲区。**红线守护**:① 全图 `grid_strength<阈值`(无 JPEG 栅格)→ 弃权返回未篡改,不对未压缩真实图取证;② `grid_strength` 分母加绝对能量地板,防真实平滑背景+结节图因 median 塌缩被放大成误报;③ 局部相位换算回全图坐标系 `(p+y)%block`,消除窗口原点偏移(否则非 block 整倍 stride 会让干净 JPEG 全图误报)。`min_flag_blocks` 聚类门压随机误报 |

**可插拔 + 降级**(沿用 dedup 范式):`CatNetTamperingDetector` 探测 `checkpoints/catnet_v2.torchscript.pt` 深度 HRNet 权重——存在则 `backend_name="catnet"` 加载 TorchScript;缺失/torch 不可用则安静降级 `backend_name="dct-bag"`(纯 NumPy/cv2 BAG 取证)。`detect()` 统一入口 = 深度CAT-Net ∪ BAG相位 ∪ 既有几何(copy-move/splice)三路并集。离线无权重环境零破坏既有检测器。

> **Gate2 跟进项(待真实数据验证)**:并集对误报率单调不减(P(任一命中)≥max各路),红线安全依赖每路各自保守。BAG 单路对干净单次 JPEG 由「全图弃权门 + 局部窗弃权 + `min_flag_blocks` 聚类」三重把守(全图同一栅格 → 局部相位恒等全图相位 → 零命中);但**重采样/缩放后的 JPEG** 会破坏 8×8 栅格对齐使局部相位散乱,属并集 FP 理论敞口(通常先被全图低 `grid_strength` 弃权门拦下)。**2026-08-01 官方真实 JPG 数据到位后须实测**:BAG 单路真图误报率 + copy-move/splice 两路几何检测器红线表现。

---

## 4. GATEKEEPER SCOPE

- **Production inference only**: Gatekeeper runs in `predict.py` for deployment safety.
- **Offline evaluation bypass**: Gatekeeper is NOT run during `evaluate.py` (holdout benchmarking) — all holdout samples are known-clean ground truth, and running GLCM+FAST/RANSAC on 614 samples would add ~15 minutes with no diagnostic value.
