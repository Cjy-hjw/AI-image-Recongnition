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
| `make_splice(img, donor, rng_or_seed, patch_frac)` | → `(spliced, mask, bbox)` |

### 类别约定

| Label | 含义 |
|:---|:---|
| 0 | 真实图像 |
| 1 | 体模(phantom) |
| 2 | AI 合成(GAN/扩散) |

---

## 1. AUTHENTICITY CLASSIFIER (`authenticity.py`)

- **Feature extraction**: 8 维真实特征:ELA + GLCM + 频域特征(非随机噪声占位)。
- **训练数据**: 从真实训练图提取 label=0(真实)、`make_phantom_like` 生成 label=1(体模)、`make_synthetic_like` 生成 label=2(合成),特征标准化后 40 epoch 训练。
- **Model**: Shallow MLP 三分类器(`AuthenticityClassifier`)。
- **Weights**: Trained during `train.py` startup, saved to `checkpoints/authenticity_model.pth`.
- **验证**: 训练结束在验证集报告三类准确率。
- **Red-line**: MLP weights MUST NOT be altered without retraining on physical phantom datasets.

> **历史说明**: 旧版用 `torch.randn` 随机噪声 + 随机标签做假训练(完全无用占位),已于本次重写为真实特征训练流程。

---

## 2. DEDUPLICATION PIPELINE (`deduplication.py`)

Two-stage duplicate detection:

| Stage | Method | Threshold |
|:---|:---|:---|
| **Primary** | 64-bit perceptual hash (pHash) + Hamming distance | Distance ≤ 12 = duplicate |
| **Secondary** | ResNet18 deep features + Cosine Similarity | Similarity > 0.95 = duplicate |

---

## 3. TAMPERING DETECTION (`tampering.py`)

| Attack Type | Detection Method | 当前能力边界 |
|:---|:---|:---|
| **Copy-Move** | Dense block-matching to find duplicate pixel regions | 召回约 20–40%(干净图零误报);合成引擎现可批量生成训练样本供后续迭代提升 |
| **Splicing** | 轴对齐长直缝几何判据:Sobel 强阶跃边 → HoughLinesP 找长直线(≥0.14·min(H,W))→ 按 ±10° 分水平/垂直缝 → **要求一条水平缝与一条垂直缝端点在 14px 内角点正交相交**(硬粘贴矩形特征) | 经两轮失败定案:像素/区域方差方案在含结节的干净图 20/20 误报(结节本身即方差异常区),触碰一票否决红线故弃用。现方案干净图 **0/80 零误报**、真实组织拼接召回 **~84%**,放弃纯噪声 donor 等极端对抗样本部分召回以换零误报 |

---

## 4. GATEKEEPER SCOPE

- **Production inference only**: Gatekeeper runs in `predict.py` for deployment safety.
- **Offline evaluation bypass**: Gatekeeper is NOT run during `evaluate.py` (holdout benchmarking) — all holdout samples are known-clean ground truth, and running GLCM+FAST/RANSAC on 614 samples would add ~15 minutes with no diagnostic value.
