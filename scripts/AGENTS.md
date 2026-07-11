# scripts/AGENTS.md - Verification & Partitioning Guidelines

## 1. DATA SPLITTING (Phase D: 纯随机图像级划分)

- **现状**: TN3K 原始数据无患者级元数据。Phase D 已废弃 `PAT_{img_id//2}` 伪分组,改为**纯随机图像级划分**(`_split_samples`),每张图独立身份,train/val 图像级互斥、固定 seed 可复现。
- **Holdout**: 来自 TN3K 独立 `test-image/` 目录,与 train/val 物理隔离,**无泄漏**。
- **Gate**: `gate5_data_leakage.py` 双检查 train↔val + holdout↔trainval 图像级 0 重叠断言。

### Package Structure (`scripts/data/`)

| File | Contents |
|:---|:---|
| `data/pseudo_labels.py` | `extract_features_from_mask_and_image()` — TI-RADS pseudo-label extraction from image+mask |
| `data/split_tn3k.py` | `format_split()` + `_split_samples()` — 纯随机图像级 train/val/holdout 划分(见 `scripts/data/AGENTS.md`) |

`format_tn3k.py` is a thin CLI that imports from `scripts/data/`.

---

## 2. verify_model.py EIGHT-GATE TEST SUITE

### Package Structure (`scripts/verify/`)

| File | Contents |
|:---|:---|
| `verify/runner.py` | `verify()` — orchestrates gate1→gate8, returns True/False |
| `verify/gates/gate1_shapes.py` | Output tensor shape assertions |
| `verify/gates/gate2_vclub.py` | vCLUB gradient isolation (encoder grad sum < 1e-7) |
| `verify/gates/gate3_auc_margin.py` | AUC-Margin dual params + BatchNorm freeze check |
| `verify/gates/gate4_famo.py` | FAMO 5-channel weight dynamics |
| `verify/gates/gate5_data_leakage.py` | 图像级泄漏双检查:train↔val + holdout↔trainval(目录缺失优雅跳过) |
| `verify/gates/gate6_stage1_masking.py` | Stage 1 gradient masking (loss_mal=0.0, zero leakage) |
| `verify/gates/gate7_optimizer_lr.py` | Optimizer LR alignment across 3 stages |
| `verify/gates/gate8_adaptor_isolation.py` | TaskAlignmentAdaptor cls→seg 梯度隔离审计(2026-07-11 新增)——`loss_mal.backward()` 后断言 `seg_backbone` 梯度精确 `==0.0`(非 `<1e-7` 容差);内部主动复位 `requires_grad`(消除 gate3 Rule 3.3 的"假冻结"掩盖)+ 强制 `adaptor_stage{3,4}.gamma` 非零(消除零初始化门控让正确/错误 detach 都测出 0 的伪 Gate 陷阱)+ `gamma_prior=0.0`(消除物理先验合法耦合通路的干扰)。详见该文件源码注释与 `src/models/AGENTS.md` §1.1、`QA_VP.md` §1.1。回归测试:`tests/test_gate8_adaptor_isolation.py`(正面通过 + monkeypatch detach 负面检测)。 |

`verify_model.py` is a thin CLI (~11 lines) importing from `scripts/verify/runner.py`.

### Backward Compatibility
- `from src.models.loss import MultiTaskLoss` resolves through `loss.py` shim → `losses/multitask.py`.

---

## 3. ZERO-HALLUCINATION DIAGNOSTIC LOOP

If any gate fails (Exit Code != 0), output a structured JSON diagnostic report: failed gate, line range, observed vs expected metrics, recovery guidance.

---

## 4. 部署与交付脚本

| 脚本 | 用途 | 关键参数/产出 | 注意事项 |
|:---|:---|:---|:---|
| `deploy.py` | 通过 SSH/SFTP 将本地代码增量同步到 A800 服务器（按文件大小差异跳过未改动文件） | 环境变量：`DEPLOY_HOST` `DEPLOY_PORT` `DEPLOY_USER` `DEPLOY_PASSWORD`；跳过 `data/` `.git/` `__pycache__` `.md` 等 | 凭证必须用环境变量注入，严禁写死；`.md` 文件不会被同步到服务器 |
| `seal_container.py` | 计算 `predict.py`、`video_predict.py`、`src/` 全树、`checkpoints/best_model.pth` 的 SHA-256 并注入 `entrypoint.sh`，构成防篡改签名链 | `--dry-run` 只打印不写入；产出：`entrypoint.sh` 中 `EXPECTED_*` 四个占位符被真实指纹替换 | **只在最终交付前跑一次**；提前封印会导致后续任何代码改动使容器启动校验失败；权重缺失时注入零值占位会使生产容器必然失败；**误跑了怎么办**：`git checkout scripts/entrypoint.sh` 回退占位哈希 |

---

## 5. 数据与权重下载脚本

| 脚本 | 用途 | 关键参数/产出 | 注意事项 |
|:---|:---|:---|:---|
| `download_thyroidxl.py` | 从 HuggingFace（或 hf-mirror）下载 ThyroidXL 公开甲状腺超声数据集，生成 SHA-256 manifest | `--output`（默认 `data/public_datasets/ThyroidXL`）；`--token`（或 `HF_TOKEN` 环境变量）；`--no-mirror` | ThyroidXL 是 Gated Dataset，需先在 HF 申请权限；访问外网须先设代理 `$env:http_proxy="http://127.0.0.1:1203"` |
| `download_pretrain_weights.py` | 离线打包下载四个预训练权重：ConvNeXt-Nano、EfficientNet-B2（timm）、SSCD-ResNet50、MedSAM ViT-B，生成 MD5/SHA-256 manifest | 产出目录：`.cache_weights/`；manifest：`.cache_weights/weights_manifest.json`；总计约 548 MB | 需要代理；CI/自动化环境须预设代理 |

---

## 6. 训练脚本

| 脚本 | 用途 | 关键参数/产出 | 注意事项 |
|:---|:---|:---|:---|
| `pretrain_malignancy.py` | MultiTaskThyroidModel malignancy 通路单帧 CE 预训练基线（绕 CBM 用临时线性头） | `--data-root` `--epochs`(30) `--batch-size`(32) `--smoke` 快速3epoch；产出 `checkpoints/malignancy_pretrain_best.pth` | 无泄漏 val AUC ≈ 0.50（患者级弱标签瓶颈，已确认），**勿据此 checkpoint 微调**；评估强制 `patient_grouped_split` |
| `pretrain_malignancy_mil.py` | 门控注意力 MIL 患者级 malignancy 预训练（一患者所有帧=一个 bag） | 阶段1（冻结backbone）：`--epochs --batch-size --max-frames-per-bag`；阶段2加 `--unfreeze-backbone --resume-from --backbone-lr-scale --llrd-eta`；产出 `checkpoints/malignancy_mil_best.pth` | Phase1 best AUC=0.5653，**低于 0.65 退出判据，当前已暂停**；Phase2 待 Phase1 突破后启动。2026-07-10 已用 `train_thyus2path_lora.py` 对拍证实卡点是数据本身而非 MIL 框架，见该行 |
| `retrain_authenticity.py` | 重训 AuthenticityClassifier（修复特征维度变更后的 gatekeeper 崩溃） | `--data-root` `--out`(默认 `checkpoints/authenticity_model.pth`)；产出含 `state_dict` + `feat_mean`/`feat_std` | **特征维度变更后必须重跑**；必须在 A800 上运行；输入维度断言==13 |
| `train_thyroidxl.py` | ThyroidXL LP-FT-LoRA 训练入口（LP 10epoch 只训头部 → FT-LoRA 40epoch 解冻 LoRA A/B）；患者级 GroupKFold(fold0) 评估 patient-AUC(Max聚合)+Dice | `--use-lora`(默认True) `--lora-r`(8) `--lora-alpha`(16) `--lp-epochs`(10) `--ft-epochs`(40) `--batch-size`(16)；产出 `checkpoints/thyroidxl_best.pth`（`{"epoch","stage","state_dict","patient_val_auc","dice"}` 包装格式，非裸 state_dict，供续训用） | **必须每 epoch 调用 `set_frozen_modules_eval()`**（2026-07-10 血泪教训：遗漏此调用导致冻结 backbone 的 BN running stats 漂移，50 epoch 后数值病态、checkpoint 报废，详见 `src/training/AGENTS.md`）；训练循环内部报告的 AUC **不能直接采信**，必须先跑 `merge_lora_checkpoint.py` 合并后用 `predict.py`/`evaluate.py` 真实推理链路重测；只用 ThyroidXL 单一数据源，不与 27021604/ThyUS2Path 混训 |
| `merge_lora_checkpoint.py` | 把 `train_thyroidxl.py` 产出的 LoRA 结构 checkpoint 合并为纯 Conv2d 部署权重（`disable_lora`/`merge_weights` 数学合并，fp32、非 autocast） | `--in`(源LoRA checkpoint) `--out`(合并后裸 state_dict路径) `--lora-r`/`--lora-alpha`(须与训练时一致，否则 `strict=True` 加载直接报错) | 产出可被 `model_runner.build_model()` 直接 strict 加载（无需推理侧感知 LoRA）；**合并脚本本身无罪**——若合并前后模型输出差异巨大，先怀疑 backbone 数值病态（退化 BatchNorm），不要怀疑合并数学（已用单层测试 diff~1e-7 验证正确性） |
| `train_thyus2path_lora.py` | 27021604(ThyUS2Path) LP-FT-LoRA 训练——**MIL vs LoRA 对拍实验**，同架构/同LP-FT-LoRA协议搬到该数据集，绕开 MIL bag 框架，验证 MIL 卡点是框架问题还是数据天花板 | `--data-root`(默认`data/27021604_extract`) `--lp-epochs`(10) `--ft-epochs`(40) `--batch-size`(16) `--val-frac`(0.2)；产出 `checkpoints/thyus2path_lora_best.pth` | 2026-07-10 实测：best patient-AUC=**0.5531**（FT-LoRA阶段train loss崩到0.02但val AUC全程0.45~0.55震荡，收官0.4998），与MIL的0.5653、随机猜0.50均在噪声范围内——**结论：瓶颈是数据本身（患者级标签广播到大量非靶区背景帧），非MIL框架问题**；不训分割头(该数据无mask标注)；只用27021604单一数据源 |

