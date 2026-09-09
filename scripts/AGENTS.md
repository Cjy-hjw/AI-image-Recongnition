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
| `data/preextract_dynamic.py` | **动态 .nii.gz 预抽取 PNG 缓存(2026-08-15 新增,首轮真训 `invalid block type` + 步耗时 22 倍抖动根因修复)**——把训练真正消费的那一帧一次性导成 512×512 uint8 PNG + `<cache-dir>/samples.pkl`,训练期 `__getitem__` 完全不再碰 `.nii.gz`。`--limit N` 先小规模验证管道(只处理前 N 个动态患者,约 1 分钟);幂等可续跑(已存在且非空的 PNG 跳过,挂掉重跑不重做);故障文件进 `preextract_report.txt`、对应样本从 samples.pkl 剔除(宁可少样本不可让训练在第 30 epoch 才炸)。⚠️ **`--img-size` 必须与训练时 `img_size` 一致**(两侧默认 512):不一致不会报错,loader 把缓存图再 resize 一次 = 降采样后又插值回去,**静默掉画质**(脚本把实际值写进 samples.pkl 的 `_meta`,`build_official_datasets(samples_pkl=...)` 加载时校验不符即 WARN)。产物被 `official_loader._read_cached_pair`/`_load_samples_pkl` 消费,详见 `src/dataset/AGENTS.md` 官方数据加载器节 |

`format_tn3k.py` is a thin CLI that imports from `scripts/data/`.

---

## 2. verify_model.py EIGHT-GATE TEST SUITE

### Package Structure (`scripts/verify/`)

| File | Contents |
|:---|:---|
| `verify/runner.py` | `verify()` — orchestrates gate1→gate8, returns True/False |
| `verify/gates/gate1_shapes.py` | Output tensor shape assertions |
| `verify/gates/gate2_vclub.py` | vCLUB gradient isolation (encoder grad sum < 1e-7) |
| `verify/gates/gate3_auc_margin.py` | AUC-Margin dual params + BatchNorm freeze check + **alpha 极性方向断言**(R6 2026-08-11 补,防"只验变化不验方向") |
| `verify/gates/gate4_famo.py` | FAMO 5-channel weight dynamics + **方向断言**(救最差任务:慢任务权重↑,R6 2026-08-11 补) |
| `verify/gates/gate5_data_leakage.py` | 图像级泄漏双检查:train↔val + holdout↔trainval(目录缺失优雅跳过) |
| `verify/gates/gate6_stage1_masking.py` | Stage 1 gradient masking (loss_mal=0.0, zero leakage) |
| `verify/gates/gate7_optimizer_lr.py` | Optimizer LR alignment across 3 stages |
| `verify/gates/gate8_adaptor_isolation.py` | TaskAlignmentAdaptor cls→seg 梯度隔离审计(2026-07-11 新增)——`loss_mal.backward()` 后断言 `seg_backbone` 梯度精确 `==0.0`(非 `<1e-7` 容差);内部主动复位 `requires_grad`(消除 gate3 Rule 3.3 的"假冻结"掩盖)+ 强制 `adaptor_stage{3,4}.gamma` 非零(消除零初始化门控让正确/错误 detach 都测出 0 的伪 Gate 陷阱)+ `gamma_prior=0.0`(消除物理先验合法耦合通路的干扰)。详见该文件源码注释与 `src/models/AGENTS.md` §1.1、`QA_VP.md` §1.1。回归测试:`tests/test_gate8_adaptor_isolation.py`(正面通过 + monkeypatch detach 负面检测)。 |
| `verify/gen_synthetic_dicom.py` | **不属于 gate1→gate8 训练验证链路**,是 CI `docker-build-smoke` job(2026-07-11 P1 输入覆盖扩容) + `tests/test_smoke_entrypoints.py::TestDicomInputPaths` 共用的合成 DICOM fixture 生成器——用 `pydicom.dataset.FileDataset` 构造真正能被 `pydicom.dcmread()` 解析(区别于 `test_entrypoint_dispatch.py::_write_dicom` 那种只有魔数字节、只供 bash 分发路由测试用的假文件)、`video_io._load_dicom_cine` 能读出 `pixel_array` 的最小合法 DICOM,支持单帧(`--frames 1`)与多帧 cine(`--frames >1`),及未压缩(默认,`ExplicitVRLittleEndian`)/压缩(`--compressed`,JPEG baseline 经 `cv2.imencode` + `pydicom.encaps.encapsulate` 封装,`TransferSyntaxUID=JPEGBaseline8Bit`)两种像素编码。CLI:`python scripts/verify/gen_synthetic_dicom.py --out PATH --frames N [--rows R --cols C] [--compressed]`。 |

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
| `deploy.py` | 通过 SSH/SFTP 将本地代码增量同步到 A800 服务器（按文件大小差异跳过未改动文件） | 凭证:环境变量优先,未设则 fallback 读 `账号密码/deploy.env`(该目录已被 `ignore_dirs` 整目录级排除,不会同步上去);跳过 `data/` `.git/` `__pycache__` `.md` `账号密码/` **`checkpoints/`** 等 | **已知局限**:按文件大小diff会漏掉"内容变但大小不变"的编辑(2026-07-13 实测复现:改 `predict.py`/`evaluate.py` 一行权重路径默认值,大小未变导致漏传,靠额外 SHA256 全量核对才发现)——涉及权重路径类的小改动后,建议額外核实关键文件内容,不能只信 deploy.py 的 SUCCESS。凭证文件后缀不属于 `.md` 时会绕过旧的后缀级过滤,新增敏感文件前先确认是否已被 `ignore_dirs` 目录级覆盖(2026-07-13 曾因此把 `deploy.env` 传到 A800,已删除+补目录级排除)。**2026-08-13 补 `checkpoints` 到 `ignore_dirs`**:此前只有形近的 `.ipynb_checkpoints`,导致本地权重会被推上 A800——实测一次 mock smoke 改写了本地 `official_heads_best.pth`,大小一变即触发 121MB 上传,而当时 A800 磁盘已用到 99%(仅剩 97G)。本脚本职责是**同步代码**,权重方向相反(A800 训练产出 → MCP download 拉回本地)。⚠️ **SFTP 登录用 `DEPLOY_USER`(root),而 `ssh-a800` MCP 执行命令用 cjy —— 两个不同用户**:deploy 传上去的文件 owner=root、模式 644,cjy 无法覆盖(目录可写也不行,见 §6 train_official.py 的可写性预检) |
| `seal_container.py` | 计算 `predict.py`、`video_predict.py`、`src/` 全树、`checkpoints/thyroidxl_merged_v2.pth` 的 SHA-256 并注入 `entrypoint.sh`,构成防篡改签名链 | `--dry-run` 只打印不写入;产出:`entrypoint.sh` 中 `EXPECTED_*` 四个占位符被真实指纹替换 | **只在最终交付前跑一次**;提前封印会导致后续任何代码改动使容器启动校验失败;权重缺失时注入零值占位会使生产容器必然失败;**误跑了怎么办**:`git checkout scripts/entrypoint.sh` 回退占位哈希 |
| `probe_heads_cuda.py` | **上云前置 GPU 契约探针**(2026-08-12 Phase2 阶段2.2 新增,阶段2.3 扩至三头,backlog #124 扩至四头)——四头(位置/侵犯/临床T/TI-RADS)在真实 CUDA 上的前向反向契约探针,查维度失配/显存/非零梯度,防"本地 CPU 过、上云 GPU 崩" | 探针项 15→21→**24**(backlog #124 扩展到 TI-RADS):本地 RTX4060 实测 **24/24 PASS、EXIT=0**(2026-08-13);batch=4 显存 allocated 45.2MB / reserved 68.0MB。四头参数量实测:ExtensionHead **0.063M** / LocationHead **0.409M** / ClinicalTHead **0.065M** / TiradsHead **0.065M**,合计约 0.602M(取代此前"三头合计 0.536M"的过时口径)。新增探针:fp16 空 mask 的 `log(clamp(s,1e-4))` 不产 -inf/NaN、fp16 下 T/TI-RADS 概率和为 1、T 头 train 模式 batch=1 不崩、T/TI-RADS 梯度不回流 ETE logits/nodule_scale/CBM 五概念(含"梯度确实到达 stage4_spatial"反证项)。**退出码三档**:0=全过无 SKIP / 2=有 SKIP(**不等于已验证**)/ 1=有 FAIL | 上云训练/交付前跑,确认四头在 GPU 上无维度/显存问题;更新头维度或新增头后必跑。⚠️ **该文件现 369 行、有意超 rules §6 的 300 行红线且不拆**——上云现场工具、云上传人工审核+约10文件限额、拆包多一层 import 失败点、切两半无认知边界;理由细节见其模块 docstring,勿当疏忽去拆 |

---

## 5. 数据与权重下载脚本

| 脚本 | 用途 | 关键参数/产出 | 注意事项 |
|:---|:---|:---|:---|
| `download_thyroidxl.py` | 从 HuggingFace（或 hf-mirror）下载 ThyroidXL 公开甲状腺超声数据集，生成 SHA-256 manifest | `--output`（默认 `data/public_datasets/ThyroidXL`）；`--token`（或 `HF_TOKEN` 环境变量）；`--no-mirror` | ThyroidXL 是 Gated Dataset，需先在 HF 申请权限；访问外网须先设代理 `$env:http_proxy="http://127.0.0.1:7897"` |
| `download_pretrain_weights.py` | 离线打包下载四个预训练权重：ConvNeXt-Nano、EfficientNet-B2（timm）、SSCD-ResNet50、MedSAM ViT-B，生成 MD5/SHA-256 manifest | 产出目录：`.cache_weights/`；manifest：`.cache_weights/weights_manifest.json`；总计约 548 MB | 需要代理；CI/自动化环境须预设代理 |

---

## 6. 训练脚本

| 脚本 | 用途 | 关键参数/产出 | 注意事项 |
|:---|:---|:---|:---|
| `train_official.py` | **官方四头(位置/ETE/临床T/TI-RADS)A/B/C 调度训练入口**(Phase E,2026-08-13;backlog #124 补 TI-RADS 后同一调度自动覆盖四头)。复用 `train_batch`(内建四头 targets 组装,归属闸门按 lam 阈值 `<=0.35` donor / `>=0.65` recipient 分派,详见 `src/training/AGENTS.md`),调度走 `src/training/schedule_official.py` | `--data-root --xlsx`(官方数据,云端)/`--mock-labels --smoke`(本地合成冒烟)/`--resume-from`(分割底座 warm-start)/`--hflip-prob`/`--freeze-legacy`/`--log-dir`/**`--samples-pkl <cache-dir>/samples.pkl`**(2026-08-15 新增:跳过 10-25 分钟样本重建,必与预抽取 `--img-size` 一致,见 `scripts/data/preextract_dynamic.py` 行);产出 `checkpoints/official_heads_best.pth` + JSONL 日志 | **不动 `train.py`**(legacy 交付路径);warm-start 用 `strict=False` 但**断言 missing_keys ⊆ 四头前缀**(多缺一个非头键即 fail-loud,杜绝静默半载);**Dice 护栏**相对 warm-start 基线跌 >1 点警、>3 点 abort(防重演 Stage2 崩塌);默认**保留 legacy 多任务损失**守护旧任务,`--freeze-legacy` 才纯头训练;JSONL 默认落 V2.1 强制的容器绝对路径,本地必须传 `--log-dir logs`。🔴 **`--out` 可写性预检(2026-08-13 加,位置在 argparse 之后、建数据集/建模型/warm-start 之前)**:`torch.save` 只在每轮 val 后执行,35 epoch 跑到第 30 轮才发现写不进去 = 前面全废。**`os.makedirs` 测不出这个** —— A800 实测复现:`checkpoints/` 目录 owner 是当前用户可写(makedirs 通过),但目标文件 owner 是 root、模式 644,`torch.save` 走 open(w) 覆盖时 Permission denied,**目录可写 ≠ 目标文件可写**。探针用 `open(path,"ab")`(append 不截断,已有权重零风险)+ 仅当 `size==0` 才删(绝不误删已有权重),失败即 `SystemExit`。三环境对拍实测:不可写时 `EXIT=1 / FATAL=1 / [smoke]=0 / Epoch=0`,正常时 `EXIT=0 / SMOKE_OK=1 / Epoch=3`。**`torch.amp.GradScaler(device.type, ...)`** 用新 API(`torch.cuda.amp.GradScaler` 在 2.8 已 deprecated,见 `docs/交付与部署/torch2.8迁移方案.md:80`;传 `device.type` 而非硬编码 `'cuda'`,CPU 冒烟不刷无谓警告;上游未把它放进 `torch.amp.__all__`,故带 `# type: ignore`,2.5.1/2.6.0/2.8.0 三版**运行时均实测可构造**)。warm-start 的 `torch.load` 显式 `weights_only=False` 是**防御性**非修 bug(A800 torch2.6 实测 17 个 checkpoint:分割底座均为纯 OrderedDict、默认参数下正常;唯一失败的 `authenticity_model.pth` 由 `gatekeeper/authenticity.py:229` 自己以 `weights_only=False` 加载,不走这里) |
| `pretrain_malignancy.py` | MultiTaskThyroidModel malignancy 通路单帧 CE 预训练基线（绕 CBM 用临时线性头） | `--data-root` `--epochs`(30) `--batch-size`(32) `--smoke` 快速3epoch；产出 `checkpoints/malignancy_pretrain_best.pth` | 无泄漏 val AUC ≈ 0.50（患者级弱标签瓶颈，已确认），**勿据此 checkpoint 微调**；评估强制 `patient_grouped_split` |
| `pretrain_malignancy_mil.py` | 门控注意力 MIL 患者级 malignancy 预训练（一患者所有帧=一个 bag） | 阶段1（冻结backbone）：`--epochs --batch-size --max-frames-per-bag`；阶段2加 `--unfreeze-backbone --resume-from --backbone-lr-scale --llrd-eta`；产出 `checkpoints/malignancy_mil_best.pth` | Phase1 best AUC=0.5653，**低于 0.65 退出判据，当前已暂停**；Phase2 待 Phase1 突破后启动。2026-07-10 已用 `train_thyus2path_lora.py` 对拍证实卡点是数据本身而非 MIL 框架，见该行 |
| `retrain_authenticity.py` | 重训 AuthenticityClassifier（修复特征维度变更后的 gatekeeper 崩溃） | `--data-root` `--out`(默认 `checkpoints/authenticity_model.pth`)；产出含 `state_dict` + `feat_mean`/`feat_std` | **特征维度变更后必须重跑**；必须在 A800 上运行；输入维度断言==13 |
| `train_thyroidxl.py` | ThyroidXL LP-FT-LoRA 训练入口（LP 10epoch 只训头部 → FT-LoRA 40epoch 解冻 LoRA A/B）；患者级 GroupKFold(fold0) 评估 patient-AUC(Max聚合)+Dice | `--use-lora`(默认True) `--lora-r`(8) `--lora-alpha`(16) `--lp-epochs`(10) `--ft-epochs`(40) `--batch-size`(16)；产出 `checkpoints/thyroidxl_best.pth`（`{"epoch","stage","state_dict","patient_val_auc","dice"}` 包装格式，非裸 state_dict，供续训用） | **必须每 epoch 调用 `set_frozen_modules_eval()`**（2026-07-10 血泪教训：遗漏此调用导致冻结 backbone 的 BN running stats 漂移，50 epoch 后数值病态、checkpoint 报废，详见 `src/training/AGENTS.md`）；训练循环内部报告的 AUC **不能直接采信**，必须先跑 `merge_lora_checkpoint.py` 合并后用 `predict.py`/`evaluate.py` 真实推理链路重测；只用 ThyroidXL 单一数据源，不与 27021604/ThyUS2Path 混训 |
| `merge_lora_checkpoint.py` | 把 `train_thyroidxl.py` 产出的 LoRA 结构 checkpoint 合并为纯 Conv2d 部署权重（`disable_lora`/`merge_weights` 数学合并，fp32、非 autocast） | `--in`(源LoRA checkpoint) `--out`(合并后裸 state_dict路径) `--lora-r`/`--lora-alpha`(须与训练时一致，否则 `strict=True` 加载直接报错) | 产出可被 `model_runner.build_model()` 直接 strict 加载（无需推理侧感知 LoRA）；**合并脚本本身无罪**——若合并前后模型输出差异巨大，先怀疑 backbone 数值病态（退化 BatchNorm），不要怀疑合并数学（已用单层测试 diff~1e-7 验证正确性） |
| `train_thyus2path_lora.py` | 27021604(ThyUS2Path) LP-FT-LoRA 训练——**MIL vs LoRA 对拍实验**，同架构/同LP-FT-LoRA协议搬到该数据集，绕开 MIL bag 框架，验证 MIL 卡点是框架问题还是数据天花板 | `--data-root`(默认`data/27021604_extract`) `--lp-epochs`(10) `--ft-epochs`(40) `--batch-size`(16) `--val-frac`(0.2)；产出 `checkpoints/thyus2path_lora_best.pth` | 2026-07-10 实测：best patient-AUC=**0.5531**（FT-LoRA阶段train loss崩到0.02但val AUC全程0.45~0.55震荡，收官0.4998），与MIL的0.5653、随机猜0.50均在噪声范围内——**结论：瓶颈是数据本身（患者级标签广播到大量非靶区背景帧），非MIL框架问题**；不训分割头(该数据无mask标注)；只用27021604单一数据源 |

---

## 7. `entrypoint.sh` — 交付入口:扩展名分发 + 内容嗅探兜底(2026-07-11 加固)

> **容器入口定位(2026-08-10 A2 起)**:entrypoint.sh 已从容器**生产入口**降级为**本地/CI 单文件调试旁路**——生产容器 `ENTRYPOINT = python -m src.service.app`(常驻 HTTP 服务,uvicorn 8000,见 Dockerfile)。本脚本的分发/嗅探/崩溃兜底逻辑仍是有价值的调试资产并被 `tests/test_entrypoint_dispatch.py` 回归守护,**但不再是 `docker run image <input>` 的默认入口**。
>
> **CI 调试旁路的正确调法(2026-08-10 CI 修复,踩过的坑)**:`docker run image <input>` 会把位置参数交给默认 HTTP ENTRYPOINT(uvicorn)直接忽略 → 起成不退出的服务挂起。要在容器里走 CLI 分发,**必须显式覆盖入口**:
> `docker run --rm --entrypoint /workspace/scripts/entrypoint.sh <image> <input>`
> (`.github/workflows/ci.yml` 的 `entrypoint-dispatch-smoke` job 即按此法,9 个 docker run 全部带 `--entrypoint`;R3 起静态 PNG/JPG 用例断言 status=PASSED——authenticity 权重缺失已降级 skipped 不再硬 ERROR;corrupted 用例仍断言 ERROR,来自 cv2.imread 解码失败分支。)

**背景**:官方"动态图"具体交付格式从未在答疑中明确过(见 `docs/历史存档/比赛对齐与风险.md`「动态图交付格式未知风险」)。纯扩展名白名单(`mp4|avi|mov|mkv|dcm|dicom` → `video_predict.py`,其余 → `predict.py`)是工程假设,格式超出预期时会静默误判成静态图,导致该样本 100% 丢失预测。

**分发逻辑(`entrypoint.sh` 第 5 节)**:
1. **扩展名判断**(`ext_kind`,与此前行为一致,是路由默认值)。
2. **内容 magic bytes 嗅探**(`sniff_kind`,交叉校验/兜底,仅当文件存在且可读时启用):
   - **DICOM**:显式检查 offset 128 起紧跟的 4 字节 `"DICM"` 魔数(`dd if=... bs=1 skip=128 count=4`),不依赖 `file --mime-type`——不同环境/精简容器的 magic db 是否收录 DICOM 签名不保证,显式 offset 判定更可靠、更可移植。
   - 其余用 `file --mime-type -b`(Linux/Git-Bash 均自带,无新依赖)取 `image/*`、`video/*` 前缀判定。
   - 文件不存在/不可读/`file` 命令不可用 → `sniff_kind="skip"`,无嗅探依据,**退化为纯扩展名路由**(不引入新失败模式;是否存在由下游 `load_input_image`/`load_video_frames` 的加载失败分支负责报错,见 `src/inference/AGENTS.md` §1)。
3. **裁决**(`sniff_kind` 有效时才生效):
   - `sniff_kind` 与 `ext_kind` **一致** → 正常路由,无额外输出。
   - **不一致** → **以内容嗅探为准重新路由**,打印 `[WARNING] Extension-based routing (...) mismatches content sniff (...) -- rerouting by content, not extension.`(不静默按可能错误的扩展名走)。
   - **两者都判不出**(`sniff_kind="unknown"`:文件可读,但既非 DICOM 也非 `image/*`、`video/*`)→ **不猜测路由**,`echo ... CRITICAL ERROR ... >&2`、从透传参数里解析 `--output_dir`(`--output_dir value` 与 `--output_dir=value` 两种写法都认,未传则退化 `predict.py` 同款默认值 `data/output`)、落 `{"status": "ERROR", "reason": "..."}` 的 `result.json`(与 `src/inference/export.py::save_rejection_json` 同构 schema)、`exit 1`,**不 `exec` 任何 python 入口**——ERROR 是诚实的失败状态,比误判成静态图导致整段动态图样本丢失预测更安全。

**实现层选择**:嗅探逻辑放在 `entrypoint.sh`(bash 层)而非 `src/inference/io.py`/`video_io.py`(Python 层)——因为分发决策(选 `predict.py` 还是 `video_predict.py`)本身就发生在 Python 进程启动之前,若把嗅探下沉到 Python 层,届时已经"选错入口"、为时已晚(比如误判成静态图启动了 `predict.py`,`predict.py` 内部再嗅探出其实是视频也无法回头去启动 `video_predict.py`)。因此嗅探必须在 dispatch 之前、bash 层完成。

**测试**:`tests/test_entrypoint_dispatch.py::TestContentSniffFallback`(6 个用例):扩展名/内容不一致双向重路由(image↔video)、一致时不误报警告、DICOM 无白名单扩展名仍靠内容识别(offset-128 preamble 特殊性的回归验证)、完全无法识别内容硬失败(exit 1 + ERROR result.json + 不分发到任何入口)、文件不存在时退化为纯扩展名路由(不因为读不到文件就误判"无法识别")。既有 `TestDispatch`/`TestProductionIntegrity`/`TestSrcTreeHashParity` 19 个用例全部沿用旧行为(用不存在的占位路径如 `/tmp/case.mp4`,天然命中"跳过嗅探"分支),未受影响。

**已知边界(不在本次范围内)**:内容嗅探只能区分"这个文件是图片还是视频/DICOM",**不能**支持全新输入形态——若官方实际交付是"JPG 帧序列"(一组按帧编号的独立文件而非单文件),对单个 JPG 帧嗅探只会得到 `image/jpeg`,依然会被路由到 `predict.py` 做单帧预测,不会自动聚合成 cine 送 `video_predict.py`。这类形态若被 8-1 官方数据证实存在,需要在入参层面新增"目录路径"这一形态,现在不凭空实现(`rules.md`「先证后改」)。

---

## 7.1 `entrypoint.sh` — 子进程崩溃兜底(2026-07-12,P2红线修复C级)

**背景**:此前分发阶段(第 5 节路由裁决完成后)用 `exec python <script> ...`,把当前 bash 进程直接替换成 python 进程。若 python 侧发生**原生崩溃**(segfault)或被 **OOM-killer 发 SIGKILL**,替换掉的进程已不存在,**没有任何存活的进程能补写 `result.json`**——评测方拿到的是空目录/零输出,直接判零分(比任何数值误判都严重,`result.json` 缺失是比 `status=ERROR` 更差的失败模式)。

**改法(`entrypoint.sh` 第 6 节)**:
1. **去掉 `exec`**,改为 `python ... &` 后台启动子进程、记录 `child_pid`、`wait "$child_pid"` 父 shell 前台等待,取真实退出码。
2. **信号转发**:不再 `exec` 替换进程后,bash 收到的 `SIGTERM`/`SIGINT`(容器编排器优雅关闭时常发)不会自动传给子进程——显式 `trap forward_signal TERM INT`,`forward_signal()` 里 `kill -TERM "$child_pid"` 转发,避免子进程变孤儿、容器优雅关闭超时后被强杀。
3. **崩溃兜底**:`wait` 返回后,若 `$out_dir/result.json` **不存在**且 `exit_code` 非零,父 shell 补写一份最小 `{"status": "ERROR", "reason": "Inference subprocess exited with code N without producing result.json ..."}`,`reason` 里带上具体退出码辅助诊断。
4. **陈旧文件规避**:分发前若 `$out_dir/result.json` 已存在(可能是上一次调用残留),先 `mv` 到 `.stale` 后缀移开——否则子进程本次崩溃但未产出新文件时,残留的旧 `result.json`(可能是上次的 `PASSED`)会被误判成"这次成功了",比空目录更危险(空目录至少诚实地是"没有结果")。流程结束(无论成功/兜底)统一清理 `.stale` 文件。
5. **out_dir 解析前置**:`--output_dir` 解析逻辑从"仅未知格式分支使用"提升为脚本级、分发前统一解析,供未知格式 ERROR 分支与崩溃兜底分支共用同一份逻辑,避免重复解析 drift。

**诚实边界(不得夸大)**:此修法能接住 **(a) python 自身 segfault** 与 **(b) python 子进程单独被 OOM-killer 选中 SIGKILL** 这两类主要情形(父 shell 在这两种情况下仍存活,能观察到子进程退出并兜底)。**接不住** **(c) 整个 cgroup/容器被外部一并杀掉**的情形——此时父 shell 自身也不存在了,任何用户态兜底都无能为力,需要容器编排层(如 k8s 存活探针 + 重启策略)兜底,不是本脚本能解决的问题,脚本注释与本文档均不得虚假宣称"已解决 OOM 问题"。

**测试**:`tests/test_entrypoint_dispatch.py::TestSubprocessCrashFallback`(4 个用例,用专用 shim `_make_crashing_shim`/`_make_success_shim` 驱动,不依赖真实 segfault):子进程崩溃且未产出 result.json → 父 shell 补写 ERROR JSON、退出码透传;子进程正常产出 result.json → 兜底逻辑不覆盖/不触碰;out_dir 预先存在陈旧 result.json + 本次崩溃 → 最终结果是本次的 ERROR 而非被误判为陈旧的成功、`.stale` 标记文件清理干净;去掉 `exec` 后分发路由本身不受影响(回归锁定)。既有 `TestDispatch`/`TestProductionIntegrity`/`TestContentSniffFallback` 全部改为显式传 `tmp_path` 隔离 `--output_dir`(改动后所有分发路径都会 mkdir/移动/清理 `$out_dir/result.json`,不再仅限未知格式分支,不隔离会污染/误删仓库真实的 `data/output/`)。

### 7.1.1 Gate2 复审补丁(2026-07-12,实测复现的真实竞态 bug)

**背景**:上面 §7.1 的初版改动落地时,C 级(entrypoint.sh)按计划走的是"主会话自审 + 专门崩溃测试代替审查",没有走 B 级那套完整 `/code-review` skill + sonnet reviewer 子agent 的 Gate2 流程——这是低估了"容器唯一入口"这个文件风险等级的判断失误。事后补做完整 Gate2(两个独立子agent 分别做逐行 diff 扫描 + 跨文件调用追踪/shell 语言陷阱扫描)后,抓到一个**在真实 bash 环境下实测复现**的竞态 bug,不是理论假设。

**bug 本体**:`wait "$child_pid"` 在被 `trap` 捕获的信号(TERM/INT)打断时,bash 会在 trap handler 执行完毕后**立即返回**一个 `128+信号值` 的合成退出码——但子进程此时可能仍未真正退出(还在优雅关闭窗口内)。旧版代码只 `wait` 一次就把返回值当成子进程的真实退出码,若信号转发的目标进程需要哪怕零点几秒完成收尾写盘,旧版会:① 把合成的 `143` 误判成"子进程崩溃退出码";② 此时 `result.json` 还没被子进程写出(它还在忙),旧版的存在性检查判定为"未产出" → 补写一份**错误的 ERROR JSON**,而实际推理本可以在零点几秒后成功完成。子进程随即变成孤儿(重新挂靠 PID 1),原本正确的结果被静默丢弃。

**A800(真实 Linux bash 4.3.48,与容器 Ubuntu 基础镜像同族)实测对照**(用一个"收到 TERM 后 `sleep 0.3` 优雅关闭、写 `status=PASSED` 后 `exit 7`"的 shim 驱动):
| | 修复前(commit `6cb60a6`) | 修复后 |
|:---|:---|:---|
| entrypoint 最终 exit code | `143`(合成值,非子进程真实退出码) | `7`(子进程真实退出码) |
| `result.json` 内容 | `{"status":"ERROR", "reason":"...without producing result.json..."}`(错误结论) | `{"status":"PASSED","source":"slow-shutdown-shim"}`(子进程真实产出) |
| 子进程状态 | 变成孤儿(`PPID=1`),继续在后台跑完优雅关闭 | 正常被等到、无孤儿残留 |

**修复(`scripts/entrypoint.sh` 第6节)**:
1. **循环 wait**:`wait "$child_pid"` 返回后用 `kill -0 "$child_pid"` 探活,若子进程仍存活就重新 `wait`,直到它真正消失才采信 `exit_code`。
2. **窄窗口信号补偿**:`trap forward_signal TERM INT` 注册时机早于 `child_pid=$!` 真正赋值,这段窄窗口内到达的信号此前会被 `forward_signal` 里的 `[ -n "$child_pid" ]` 判假静默丢弃。改用 `signal_received` 标记位记录,`child_pid` 一旦可用立即核实是否要补发。
3. **收尾窗口延长保护**:`trap - TERM INT` 从"wait 后立即清除"挪到 `exit` 前最后一刻——期间(补写崩溃兜底 JSON、清理 `.stale`)若收到二次信号,trap 仍在场,bash 不会因未设 trap 的信号默认终止 shell 本身,避免兜底逻辑写到一半被打断。
4. **陈旧文件 mv 校验**:`mv -f "$result_file" "$result_file.stale"` 静默失败(只读文件系统等)时补一条 `[WARNING]` 到 stderr,不再假装陈旧文件规避一定生效。
5. **崩溃兜底 JSON 写入原子化**:`printf ... > "$result_file"` 改为先写 `.tmp` 再 `mv -f` 落位,避免这条兜底路径自己也留下半写文件。
6. **`src/inference/export.py` 同步加固**(python 侧真正的数据源头):`save_prediction_artifacts`/`save_rejection_json` 的 `result.json` 写入改经 `_write_json_atomic`(temp 文件 + `os.replace`),防止 SIGTERM 恰好落在 `json.dump` 写盘期间导致磁盘上出现"存在但截断"的 `result.json`——这类文件比完全不存在更危险,存在性检查对它没有免疫力。

**测试**:`tests/test_entrypoint_dispatch.py::TestSignalForwarding::test_signal_forwarded_and_wait_captures_real_exit_code` —— 用真实 `SIGTERM`(非 shim 自行退出的固定码)驱动上面同款"慢速优雅关闭" shim,断言 entrypoint 最终退出码是子进程的真实值(非合成 128+信号)、`result.json` 是子进程自己写出的那份而非兜底 ERROR。`os.name == "nt"` 时 skip(POSIX 信号语义在 Git Bash/Windows 进程模型下不保证一致,真实验证以 Linux CI 为准;本机验证额外走 A800 真实 Linux bash 环境人工复现,见上表)。

**教训沉淀**:① entrypoint.sh 这类"改错即整个容器交付失败"的关键基础设施文件,即使不涉及 ML 数学管道,也不能用"有专门测试就等于走过 Gate2"来替代真正的 `/code-review` + reviewer 子agent 审查——本次两者本该同时具备,上一轮只做了前者;② shell 脚本的信号/进程语义类 bug,本地开发机复现不了不代表不存在,必须在与生产容器同族的真实 Linux bash 环境验证(本次用 A800 服务器,而非仅信任本地 Windows Git Bash 或纯理论推导)。

