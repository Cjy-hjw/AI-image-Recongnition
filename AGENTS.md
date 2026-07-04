# AGENTS.md - Global Project Architecture & Rules Directory

This document serves as the master coordinator for the thyroid multi-task diagnostic project, defining overall specifications and indexing module-specific rules.

---

| Directory / Module | Specific Rules File | Key Focus |
| :--- | :--- | :--- |
| **Global Rules & Gates** | `/rules.md` | Pre-flight check workflow, OOM bounds, abort policy, **code modularity**. |
| **Quality Verification** | `/QA_VP.md` | Non-negotiable testing standards, check lists. |
| **Model & Custom Loss** | `/src/models/AGENTS.md` | vCLUB detach path, BatchNorm freeze eval logic, AUC-Margin unscale sequence. |
| **Loss Functions** | `/src/models/losses/` | Decomposed loss modules: segmentation, classification, concepts, multitask, components. |
| **Training Pipeline** | `/src/training/AGENTS.md` | Three-stage progressive training, parameter freezing, NaN sentinels. |
| **Data Loading & Augmentation** | `/src/dataset/AGENTS.md` | Static normalization, MixUp/CutMix, patient ID grouping. |
| **Inference Pipeline** | `/src/AGENTS.md` §2 | Shared inference modules: io, model_runner, postprocess, calibration, export. |
| **Gatekeeper Pipeline** | `/src/gatekeeper/AGENTS.md` | Authenticity classification, deduplication, tampering detection. |
| **Scripts & Splits** | `/scripts/AGENTS.md` | 7-Gate verify_model.py tests, data splitting (纯随机图像级划分). |
| **CI / 架构铁律** | `/.github/workflows/ci.yml` + `/.importlinter` | PR/push 到 main 自动跑两 job:① import-linter 4 契约(inference⊄training、models 无反向依赖、gatekeeper⊄上层、utils 叶子层)② Gate 1 七门 + 全套 pytest。任一红则挡合并。 |
| **Tests** | `/tests/AGENTS.md` | 全套 **181 passed**:14 physical_prior + 15 cbm_head + 19 phase_refactor + 41 gatekeeper + 10 masked_loss + 21 video + 2 authenticity_trainer + 7 multisource + 2 smoke_entrypoints + 11 calibration + 19 entrypoint_dispatch + **20 pathology_dataset**(标签广播/患者分组/valid键契约). |
| **Source Packages** | `/src/AGENTS.md` | Package structure overview, TI-RADS score guidelines. |

---

## 1. MODULAR CODE ARCHITECTURE

模块文件结构与职责详见各子目录 AGENTS.md；此处仅列顶层导航。

- **`src/models/`** — 双主干模型 + CBM 头 + 物理先验 + 损失：见 `src/models/AGENTS.md`
- **`src/training/`** — 三阶段渐进训练 + GradScaler 协议 + vCLUB 隔离：见 `src/training/AGENTS.md`
- **`src/inference/`** — 推理管线（io / model_runner / postprocess / calibration / export / video）：见 `src/inference/AGENTS.md`
- **`src/dataset/`** — 数据加载 + 增强 + PathologyDataset 病理预训练：见 `src/dataset/AGENTS.md`
- **`src/gatekeeper/`** — 真实性 / 去重 / 篡改四层门控：见 `src/gatekeeper/AGENTS.md`
- **`scripts/`** — 7-Gate verify_model.py + 数据格式化 + 预训练管道：见 `scripts/AGENTS.md`
- **`tests/`** — 181 tests（含 20 pathology_dataset 新增）：见 `tests/AGENTS.md`

所有拆分模块通过 `__init__.py` re-export shim 保持旧导入路径兼容。

---

## 2. PRODUCTION INFERENCE & CONTAINER DEPLOYMENT

### 2.1 Inference Optimizations
见 `src/inference/AGENTS.md` §3（`torch.inference_mode()` + clamp ±15 规范）。

### 2.2 PACS DICOM Metadata Anonymization
21 PHI 字段脱敏（HIPAA Safe Harbor）。字段清单与规则见 `src/inference/AGENTS.md` §1 及 `QA_VP.md §1.6`。

### 2.3 Container Checksum Verification & Entrypoint Dispatch
- `scripts/entrypoint.sh` executes SHA-256 verification on `checkpoints/best_model.pth`, `predict.py`, `video_predict.py`, **and the entire `src/` tree** (核心推理逻辑,原仅校验入口脚本是防篡改盲区,现闭合)。`PRODUCTION=1` → hard exit on any hash mismatch.
- **打包密封**:`scripts/seal_container.py` 交付前算 predict/video_predict/**src 全树**/best_model 的 SHA-256,正则注入 entrypoint.sh 的 `EXPECTED_*` 占位位。src 树聚合指纹算法(find *.py → POSIX 相对路径 → LC_ALL=C 字节序排序 → 逐文件 `sha  path\n` 拼接 → SHA-256)由 `scripts/_src_tree_hash.sh`(bash)与 `seal_container.py`(Python)**字节级配对**,单一事实源避免 drift,测试反向验证改坏函数必 FAIL。`.gitattributes` 锁 `.py` 为 LF 防 Win 构建/Linux 运行行尾哈希假阳性。
- **防护边界**:此机制为「自包含篡改检测」非密码学防篡改——EXPECTED_* 写死在 entrypoint 自身且 src 树不含 entrypoint,能改 src 者可重算哈希绕过;真防篡改需信任锚外移(镜像签名 + cosign 外部验签)。详见 entrypoint.sh 顶部威胁模型注释。
- **入口按输入扩展名自动分发**:`.mp4/.avi/.mov/.mkv/.dcm/.dicom` → `video_predict.py --input_video`(DICOM 单帧亦安全,内部 `_load_dicom_cine` 视作 1 帧);其余静态图 → `predict.py --input_image`。第一个位置参数为输入路径,其余 flag 透传。
- Dockerfile `COPY` 同时拷入 `predict.py` 与 `video_predict.py`;`requirements.txt` 显式声明 torch/torchvision/numpy/safetensors + faiss-cpu(去重加速,缺失自动降级 numpy 检索),paramiko 仅训练需要不入容器(`scripts/deploy.py` SSH 凭证经 `DEPLOY_*` 环境变量注入,严禁写死入库)。

---

## 3. CLASSIFICATION AUC BREAKTHROUGH REFACTORING

### 3.1 Architecture Changes
- **Morphological Erosion Circularity**: Differentiable erosion in `physical_prior.py` aligns with `cv2.findContours` evaluation.
- **EMA Entropy Gating**: EMA-tracked median entropy in `cbm_head.py`, 5% leakage floor.
- **SupCon Projection Remapping**: 352-dim pooled backbone features (not 288-dim concept features).

### 3.2 Dual-Backbone Architecture
ConvNeXt-Nano（分割）+ EfficientNet-B2（分类）+ TaskAlignmentAdaptor 跨任务对齐。见 `src/models/AGENTS.md` §1。

### 3.3 Loss & Training Changes
5-Task FAMO + 3 阶段渐进 + Stage 1 梯度遮蔽 + AUCM 对偶参数。见 `src/models/AGENTS.md` §7 和 `src/training/AGENTS.md` §1。

### 3.4 Verification
`python scripts/verify_model.py` — 7 门全过（shapes / vCLUB / AUC-margin / FAMO / data-leakage / stage1-masking / optimizer-lr）。见 `scripts/AGENTS.md` §2。