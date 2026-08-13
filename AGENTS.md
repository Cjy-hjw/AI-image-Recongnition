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
| **Scripts & Splits** | `/scripts/AGENTS.md` | 8-Gate verify_model.py tests, data splitting (纯随机图像级划分). |
| **CI / 架构铁律** | `/.github/workflows/ci.yml` + `/.importlinter` | PR/push 到 main 自动跑三 job:① import-linter 4 契约(inference⊄training、models 无反向依赖、gatekeeper⊄上层、utils 叶子层)② `test` job 跑 Gate 1 八门(2026-07-11 补 Gate 8 cls→seg 梯度隔离审计)+ 全套 pytest ③ `docker-build-smoke`(2026-07-11 新增,host runner 上真实 `docker build` + 起容器冒烟——此前 CI 只在裸机镜像内 pip install「模拟」容器环境,从未真 build 过 Dockerfile,`entrypoint.sh` 缺 `_src_tree_hash.sh` 那次 bug 就是因此潜伏未被发现;同日首次真跑就抓到 CPU+fp16/fp32 autocast 假降级真 bug,详见 `src/inference/AGENTS.md`)。**输入覆盖(2026-07-11 P1 扩容,两轮)**:标准 PNG 静态图 + 标准 MP4 动态图 + DICOM 单帧 + DICOM 多帧 cine + **压缩 DICOM(JPEG baseline)** + **JPG 静态图**(官方确认交付格式)+ 非空损坏文件(必须 ERROR 不崩溃)+ 极端低分辨率(4x4)视频,共 8 类场景,均用真实容器验证机制层(能否构建/分发/落盘),不验证预测数值(该职责在 Gate1-8+pytest)。压缩 DICOM 一项是 Opus 战略审查发现的致命缺口补测——真实超声 DICOM 绝大多数是 JPEG/JPEG2000/RLE 压缩,`requirements.txt` 已补 `pylibjpeg`/`pylibjpeg-libjpeg`/`pylibjpeg-openjpeg`(RLE 由 pydicom 内置 `rle_handler` 覆盖,无需额外包),版本锁 `numpy<2` 兼容区间(见 `requirements.txt` 注释)。任一红则挡合并。 |
| **Tests** | `/tests/AGENTS.md` | 🔴 **测试计数的单一权威在 `tests/AGENTS.md`,本表不再重复罗列明细**(此前三处堆砌同一份明细且都自带"与实际有偏差"免责,维护成本高于价值)。当前口径:**814 collected / 813 passed + 1 skipped**(2026-08-13 `Demo_D` 主环境实测,backlog #124 TI-RADS 头补齐 + Gate2 审查后补回归测试对拍;含四头相关单测 142+1 项 = head_interfaces 14 + location 25 + extension 39 + clinical_t 27 + official_head_losses 10 + train_step_head_targets 12 + tirads_head 15 + **official_schedule 16**(Phase E A/B/C 调度 + JSONL) + official_loader 的防回归 2(hflip RNG + mask 命名 glob))。⚠️ **跨环境不可比**:torch2.8 预演环境的 TI-RADS 覆盖待下次云端对拍;此前 795/794+1 快照(2026-08-12)已过时。skip 数随依赖集与平台门浮动,**判回归必须同环境前后对拍**。精确值一律以 `pytest --collect-only tests/` 实跑为准。 |
| **Source Packages** | `/src/AGENTS.md` | Package structure overview, TI-RADS score guidelines. |
| **Claude Code Hooks** | `/.claude/hooks/AGENTS.md` | 非交付物,会话自动化护栏:Gate2 审查强制执行状态机(reminder→escalate_block→enforce→mark_pass) + 若干提醒式 hook。三次真实教训(高风险清单不全/绝对路径vs相对路径不一致/`.py`扩展名无差别放行非代码文件)记录在案,改动前必读。

---

## 1. MODULAR CODE ARCHITECTURE

模块文件结构与职责详见各子目录 AGENTS.md；此处仅列顶层导航。

- **`src/models/`** — 双主干模型 + CBM 头 + 物理先验 + 损失：见 `src/models/AGENTS.md`
- **`src/training/`** — 三阶段渐进训练 + GradScaler 协议 + vCLUB 隔离 + **官方三头 A/B/C 调度(`schedule_official.py`,禁复用 stages.py)**：见 `src/training/AGENTS.md`
- **`src/inference/`** — 推理管线(io / model_runner / postprocess / calibration / export / video / case_pipeline / v21_output / prediction_json / nifti_io / nodule_split):见 `src/inference/AGENTS.md`
- **`src/dataset/`** — 数据加载 + 增强 + PathologyDataset 病理预训练：见 `src/dataset/AGENTS.md`
- **`src/gatekeeper/`** — 真实性 / 去重 / 篡改四层门控：见 `src/gatekeeper/AGENTS.md`
- **`scripts/`** — 8-Gate verify_model.py + 数据格式化 + 预训练管道：见 `scripts/AGENTS.md`
- **`src/service/`** — V2.1 常驻 HTTP 推理服务(app/tasks/config/case_discovery/runner):见 `src/service/AGENTS.md`
- **`tests/`** — 测试计数与模块明细见上表「Tests」行 / `tests/AGENTS.md`(单一权威,不在此重复)。

所有拆分模块通过 `__init__.py` re-export shim 保持旧导入路径兼容。

---

## 2. PRODUCTION INFERENCE & CONTAINER DEPLOYMENT

### 2.1 Inference Optimizations
见 `src/inference/AGENTS.md` §3（`torch.inference_mode()` + clamp ±15 规范）。

### 2.2 PACS DICOM Metadata Anonymization
21 PHI 字段脱敏（HIPAA Safe Harbor）。字段清单与规则见 `src/inference/AGENTS.md` §1 及 `QA_VP.md §1.6`。

### 2.3 Container Checksum Verification & Entrypoint Dispatch
- `scripts/entrypoint.sh` executes SHA-256 verification on `checkpoints/thyroidxl_merged_v2.pth`, `predict.py`, `video_predict.py`, **and the entire `src/` tree** (核心推理逻辑,原仅校验入口脚本是防篡改盲区,现闭合)。`PRODUCTION=1` → hard exit on any hash mismatch.
- **打包密封**:`scripts/seal_container.py` 交付前算 predict/video_predict/**src 全树**/best_model 的 SHA-256,正则注入 entrypoint.sh 的 `EXPECTED_*` 占位位。src 树聚合指纹算法(find *.py → POSIX 相对路径 → LC_ALL=C 字节序排序 → 逐文件 `sha  path\n` 拼接 → SHA-256)由 `scripts/_src_tree_hash.sh`(bash)与 `seal_container.py`(Python)**字节级配对**,单一事实源避免 drift,测试反向验证改坏函数必 FAIL。`.gitattributes` 锁 `.py` 为 LF 防 Win 构建/Linux 运行行尾哈希假阳性。
- **防护边界**:此机制为「自包含篡改检测」非密码学防篡改——EXPECTED_* 写死在 entrypoint 自身且 src 树不含 entrypoint,能改 src 者可重算哈希绕过;真防篡改需信任锚外移(镜像签名 + cosign 外部验签)。详见 entrypoint.sh 顶部威胁模型注释。
- **入口按输入扩展名自动分发**:`.mp4/.avi/.mov/.mkv/.dcm/.dicom` → `video_predict.py --input_video`(DICOM 单帧亦安全,内部 `_load_dicom_cine` 视作 1 帧);其余静态图 → `predict.py --input_image`。第一个位置参数为输入路径,其余 flag 透传。
- **result.json 输出契约(2026-07-29 起,官方"分类必须输出概率分布否则计0分"硬约束)**:两入口共用 `build_probability_export` 产出 `concept_probabilities`/`concept_predicted_index`/`malignancy_probabilities` 完整分布;`gatekeeper` 段两入口对齐(含 `flagged`/`would_reject_in_hard_mode`/`soft_flags`,video 聚合层另有 `any_*` 别名);video 侧 `n_frames_sampled`/`n_frames_diagnosed`/`n_frames_failed`/`failed_frame_indices` 三分帧数不静默少算。**坏帧隔离**(2026-07-30):单帧概率/mask NaN 或形状非法只丢该帧不拖垮整段,`img_tensor` 仅存关键帧(防 32 帧常驻 14G 显存 OOM),关键帧错位已从结构上消除(tensor 存 per_frame 元素,不再用平行下标 dict)。契约细节见 `src/inference/AGENTS.md` §2/§4。
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