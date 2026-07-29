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
| **Tests** | `/tests/AGENTS.md` | 全套 **311 passed, 1 skipped**(312 collected,2026-07-29 用 `pytest --collect-only -q` 实测核对):18 physical_prior(含2 e_prior→seg_backbone端到端梯度耦合,2026-07-11新增) + 15 cbm_head + 19 phase_refactor + **49 gatekeeper**(2026-07-12 统一软flag制度+code-review补强,共5例:copy_move/splice与CatNet状态解耦回归 + BAG硬拒绝开关双向回归 + CatNet/BAG命中来源标签`last_hard_reject_source`回归) + 10 masked_loss + **37 video**(2026-07-12 `TestGatekeeperSoftFlagPolicy` 8例:authenticity不再一票否决的整合层回归 + REJECTED reason按命中源标注的3例回归;2026-07-29 新增 `TestEnforcementModeSoftDefault` 7例:soft 为默认、非法值降级 soft、hard 仍可拒、交付 yaml 必须是 soft) + **21 probability_output**(2026-07-29 新增,官方"分类必须输出概率分布否则计0分"硬约束的契约锁定:分布和为1/argmax与分布自洽/类别顺序对齐映射表与 yaml/JSON可序列化/两入口共用同一 exporter) + **7 gatekeeper_soft_enforcement**(2026-07-29 新增,入口层回归:审计命中后仍输出完整诊断,单帧命中不再让整段视频零诊断,hard 模式拒绝链路仍完整) + 3 authenticity_trainer + 7 multisource + **18 smoke_entrypoints**(2026-07-11 多轮扩容:输入/config 加载失败 ERROR 兜底 + P1 输入覆盖——DICOM单帧/多帧cine/压缩DICOM(单帧+多帧参数化) + JPG静态图 + 非空损坏文件) + 11 calibration + **30 entrypoint_dispatch**(2026-07-12 新增 `TestSubprocessCrashFallback` 4例:P2红线修复C级,entrypoint.sh去exec后子进程崩溃兜底回归;同日 Gate2 复审补丁再加 `TestSignalForwarding` 1例:真实 SIGTERM 驱动,锁定 wait 被 trap 打断提前返回的竞态 bug 已修复) + 20 pathology_dataset + 13 pathology_mil_dataset + 4 bn_freeze + **15 model_runner**(2026-07-11 两轮修复 CPU autocast 假降级 + load_config 契约新增) + 13 report_generator + **2 gate8_adaptor_isolation**(2026-07-11 新增，TaskAlignmentAdaptor cls→seg 梯度隔离正面+monkeypatch负面回归)。 |
| **Source Packages** | `/src/AGENTS.md` | Package structure overview, TI-RADS score guidelines. |
| **Claude Code Hooks** | `/.claude/hooks/AGENTS.md` | 非交付物,会话自动化护栏:Gate2 审查强制执行状态机(reminder→escalate_block→enforce→mark_pass) + 若干提醒式 hook。三次真实教训(高风险清单不全/绝对路径vs相对路径不一致/`.py`扩展名无差别放行非代码文件)记录在案,改动前必读。

---

## 1. MODULAR CODE ARCHITECTURE

模块文件结构与职责详见各子目录 AGENTS.md；此处仅列顶层导航。

- **`src/models/`** — 双主干模型 + CBM 头 + 物理先验 + 损失：见 `src/models/AGENTS.md`
- **`src/training/`** — 三阶段渐进训练 + GradScaler 协议 + vCLUB 隔离：见 `src/training/AGENTS.md`
- **`src/inference/`** — 推理管线（io / model_runner / postprocess / calibration / export / video）：见 `src/inference/AGENTS.md`
- **`src/dataset/`** — 数据加载 + 增强 + PathologyDataset 病理预训练：见 `src/dataset/AGENTS.md`
- **`src/gatekeeper/`** — 真实性 / 去重 / 篡改四层门控：见 `src/gatekeeper/AGENTS.md`
- **`scripts/`** — 8-Gate verify_model.py + 数据格式化 + 预训练管道：见 `scripts/AGENTS.md`
- **`tests/`** — 311 passed, 1 skipped（含 20 pathology_dataset + 13 pathology_mil_dataset + 4 bn_freeze + 15 model_runner + 2 gate8_adaptor_isolation + 49 gatekeeper + 37 video + 30 entrypoint_dispatch + 21 probability_output + 7 gatekeeper_soft_enforcement）：见 `tests/AGENTS.md`

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