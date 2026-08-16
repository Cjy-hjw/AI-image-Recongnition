# src/training/AGENTS.md - Training Pipeline Specifications

## 0. PHASE 4 新增组件

### PCGrad优化器 (`pcgrad.py`) — 梯度冲突消解 ⚠️实验性,未接入训练主循环
- **⚠️ 边界约束(激活前必读)**:
  - **未接入 `train.py`**: 当前训练主循环用 FAMO 动态加权,不调用本类。
  - **与 FAMO 互斥**: PCGrad/FAMO 二选一：启用 PCGrad 前须先从 train.py 移除 FAMO 调用，并过 Gate2 复审。
  - **不兼容 AMP/GradScaler**: `pc_backward` 手工 `backward()`+`grad.copy_()`,GradScaler 感知不到这些
    手工梯度,会与 `scaler.step()` 冲突(inf/nan 静默跳步)。启用需 fp32 或额外适配。
  - **激活前必须过 Gate2 复审**(`rules.md §2`)。
- **背景**: 分割(几何驱动) vs 分类(纹理驱动)对共享编码器的梯度更新方向可能冲突(余弦<0),导致负迁移
- **原理**:
  - 计算各任务梯度g_i, g_j
  - 若g_i·g_j<0(冲突),投影g_i到g_j正交超平面: `g_i' = g_i - (g_i·g_j/||g_j||²)*g_j`
  - 消除冲突方向,保留一致方向
- **实现**:
  - `PCGrad(optimizer, shared_parameters=None)`: 包装Adam/SGD,在step()前拦截梯度投影
  - **`shared_parameters` 参数(54f0302 修复)**: PCGrad 理论只应对**共享编码器**参数做冲突投影;
    对各任务私有 head 投影是错的(head 梯度本就不该跨任务冲突,投影反而损害各 head 学习)。
    传入共享参数列表则只对其投影;不传则回退全参数(旧行为)并 docstring 警告。
  - `pc_backward(losses)`: 内部逐任务backward+清零提取独立梯度,两两投影,聚合后赋值参数梯度
  - **关键修复(d4c00ff)**: 旧版假设外部已backward导致梯度累加无法投影,新版内部自动处理
- **使用**:
  ```python
  # 仅对共享编码器投影(正确用法),且需 fp32、与 FAMO 二选一
  pcgrad = PCGrad(optimizer, shared_parameters=model.encoder.parameters())
  pcgrad.pc_backward([loss_seg, loss_cls])  # 自动backward+投影
  pcgrad.step()
  ```
- **显存**: <5%额外开销,适合官方 32G 显存(2026-08-12 实测,见总纲 §3;2026-08-11 的"14~16G"误判已作废)约束

---

## 1. THREE-STAGE PROGRESSIVE TRAINING (`stages.py` + `MultiTaskThyroidModel`)

| Stage | Epochs | Strategy | Learning Rate |
|:---|:---|:---|:---|
| **Stage 1** | 1–40 | Segmentation + Concepts (classification frozen, loss_mal/supcon masked to 0.0) | `lr_base` (0.0024) |
| **Stage 2** | 41–100 | Full model unfrozen, 5-task FAMO scheduling | `lr_base` (0.0024) |
| **Stage 3** | 101–120 | Backbone + seg frozen, AUC-Margin fine-tuning only | 5e-6 |

Stage freeze rules live on the model via `MultiTaskThyroidModel.set_training_stage(stage)` and `is_param_trainable(name, stage)`. `configure_stage_parameters()` delegates freezing to the model, then builds optimizer param groups and LR schedule.

### Critical Freezing Rules

- **vCLUB isolation**: vCLUB 使用专属 vclub_optimizer(lr=1e-4) 独立优化；主 optimizer 参数组显式排除 vclub 参数。
- **Stage 1**: `loss_mal` and `loss_supcon` multiplied by 0.0 to prevent classification gradient leakage.
- **Stage 3**: FAMO disabled (`update_famo_weights=False`).
- **BatchNorm auto-eval**: After `model.train()`, modules with all-frozen params are set to `.eval()` (see `loop/bn_freeze.py`).

### Checkpoint & Early Stopping

- **Stage 1 best**: `val_dice` only.
- **Stage 2–3 best**: `checkpoint_w_dice * val_dice + checkpoint_w_auc * val_auc` (defaults 0.30 / 0.70).
- **Stage 3 ES**: `stage3_early_stopping_patience=0` disables early stopping; tracker resets at epoch 101.
- **`validate_epoch` return**: 5-tuple `(avg_loss, avg_dice, avg_acc, avg_loss_l2, val_auc)`; `val_auc` is `None` in Stage 1 or single-class val sets.
- 阶段 2.4 新增可选出参 `official_metrics_out`(dict):传入时 `validate_epoch` 收集每批 `validate_batch` 返回的 `official_records`,epoch 末调 `metrics_official.compute_official_metrics` **原地写入**官方口径指标,返回值**不加宽**(`train.py:337` 按 5 元组解包,唯一调用点)。三头键缺席的数据集 → 写入全 None。选模判据与 Dice 护栏在 `scripts/train_official.py`(Phase E 已实现,2026-08-13:composite = 可算官方指标均值;Dice 相对 warm-start 基线跌 >1 点警、>3 点 abort)。

---

## 2. TRAINING & VALIDATION LOOPS

### Package Structure (`training/loop/`)

| File | Responsibility |
|:---|:---|
| `train_step.py` | `train_batch()` — single batch forward/backward with MixUp, vCLUB, SLR, AMP;模块级 `build_official_head_targets()` 组装四头 targets(见下) |
| `validate_step.py` | `validate_batch()` — single batch validation with Val Diag print;返回 **8 元组**(新增 `official_records`);`collect_official_records()`/`_official_head_targets()` |
| `metrics_official.py` | **官方口径验证指标(阶段 2.4 新增)**:病灶级聚合(同 `lesion_id` 多帧概率取均值)+ 官方筛选(见下) |
| `slr.py` | `compute_slr_weights()` — Supervised Label Rectification (1000-sample history, mean+2σ anomaly);把 `mix_info` 的 `lam` 当**软加权系数**消费(`slr.py:26-36` 按 `lam*raw_orig + (1-lam)*raw_perm` 插值),故四头监督闸门改动不得在 mixup 侧硬化 `lam`(见 `build_official_head_targets` 下方红线) |
| `vclub.py` | `vclub_learning_step()`, `compute_vclub_mi_bound()` — mutual information decoupling |
| `bn_freeze.py` | `set_frozen_modules_eval()` — automatic BatchNorm eval for frozen modules |

### 官方四头训练模块(Phase E,2026-08-13 新增;backlog #124 补 TI-RADS 头后同一套调度自动覆盖四头,均在 `src/training/` 顶层)

| File | Responsibility | 硬约束 |
|:---|:---|:---|
| `schedule_official.py` | 官方四头 A/B/C 阶段调度:`get_official_stage()`/`official_head_lr()`/`configure_official_stage()` | 🔴 **相邻 epoch lr 比值 ≤5**(`MAX_LR_JUMP_RATIO`),把 `stages.py` 的 100 倍跳变事故钉成不可能;返回契约与 `configure_stage_parameters` 同型 `(param_groups, lr, stage_name)`;**绝不复用/修改 `stages.py`**。🔴 **2026-08-16 契约变更:分割前向链(`seg_backbone`/`up_stages`/`dec_stages`/`seg_head`)改为 A/B/C 全程可训**——原设计冻结它是为「保护**已有**分割能力」(`训练架构方案.md:100`),而那个 warm-start 权重已三路不可达;冻结导致 `pred_mask` 恒全零 → 官方 IoU 匹配零命中 → 五项分类/分期零样本 → Sclin≈0(总纲 §6③)。解码器挂 `head_lr`(随机初始化)、ImageNet 主干挂 `head_lr×backbone_lr_scale`,**lr 档位与 Stage B 现有两档重合、不新增档次**。⚠️ **必须整块解冻**(部分解冻会触发 `bn_freeze` 的兄弟 BN 病态);⚠️ **解冻即须进 param_groups**(`requires_grad` 与 optimizer 集合必须严格对齐,否则梯度跨 batch 永不清零,见 `src/models/AGENTS.md:62` LoRA 同形事故) |
| `jsonl_logger.py` | V2.1 强制 JSONL 训练日志(`JsonlLogger`) | 云端路径写死 `/2026aicompetition/workspace/logs`(`OFFICIAL_LOG_DIR`,Windows 本地不可写,冒烟须 `--log-dir logs`);**逐行 flush**(审查随时读,缓冲区里的行读不到 = 判罚口径下自曝);`log(extra={...})` 收额外诊断字段(子损失分解/`n_valid_*`),官方原文「**应包含**以下字段」= 最小必填集未禁额外键,但 ① `_RESERVED` assert 拦同名覆盖(静默改写 `loss` 比崩溃阴)② `_drop_nonfinite` 丢 NaN/Inf(`json.dumps` 默认写裸 `NaN`,非标准 JSON,严格解析器拒收整行;**丢弃不填 0** —— 填 0 会把"这项炸了"伪装成"收敛得好") |
| `reproducibility.py` | seed 四件套共享助手:`seed_worker()`/`configure_reproducibility()` | sampler / loader **两个独立 generator**(共用会让改 `num_workers` 连带改变采样序列);`train.py` 内那份是 legacy 已验证路径,**不动**,两份漂移各自的测试会红 |

**阶段划分与 lr**(`lr_base` 取 `config.train.lr`):A=epoch 1-10(冻 cls_backbone/CBM,训四头 + **分割前向链**,线性 warmup 至 lr_base)、
B=11-30(额外解冻 `cls_backbone.blocks[5]/[6]`,LLRD:blocks[6]=头 lr×0.1、blocks[5] 再乘 η=0.75,余弦退火至 ×0.1)、
C=31-35(回到 A 的可训集,头部固定 lr_base×0.05;温度标定在推理侧 `calibration.py` 不在此)。
🔴 **分割前向链(`seg_backbone`/`up_stages`/`dec_stages`/`seg_head`)A/B/C 全程可训**(2026-08-16 契约变更,理由见上表 `schedule_official.py` 行):
解码器(随机初始化)挂 `head_lr`、与四头同组;ImageNet 主干挂 `head_lr×backbone_lr_scale`,与 Stage B 的 blocks[6] **同档不新增**。
param_groups 组数实测:**A/C 各 5 组、B 9 组**(stage 边界重建 optimizer,stage 内按位置 zip 更 lr —— 同 stage 内组数与各组参数数恒定,已实测)。
四头(位置/侵犯/临床T/TI-RADS)在**所有** legacy stage 恒冻结(`OFFICIAL_HEAD_PREFIXES`,含 `tirads_head`),两条调度线互不干扰。
入口是 `scripts/train_official.py`(**不动 `train.py`**),数值锁定见 `tests/test_official_schedule.py`(18 项)。官方四头 JSONL 计数另逐文件钉死于 `tests/AGENTS.md` 各头小节。


### `build_official_head_targets()`(train_step.py,阶段 2.4 提取为可单测模块级函数)

**必须在 mixup/cutmix 之后调用**——三头是硬标签 masked CE、无软标签路径,图像混了而标签不跟着换 = 静默错监督。混合后归属按阈值分派(D 段 CutMix 的 `lam` 本由 DR-CutMix 算成 0.0/1.0;M 段 MixUp 的软 `lam` 在 `[0,1]` 内):`lam<=0.35` 继承 donor(`index`)标签须 gather、`lam>=0.65` 保留 recipient、其余(两分支中间段)整批屏蔽。validity 走 **targets 顶层直传键**(`valid_lesion_loc` 等,float [B]),**不进 `label_validity`**——后者只被 `malignancy.py:25`/`concepts.py:155` 消费,三头损失各自显式断言 [B] 一维 validity、根本不读它。🔴 **三头只训主病灶**:`is_main==0` 的次病灶(约 13% 带标注)validity 置 0(评审规则:三新维度仅对主病灶标注/评估);`is_main` 键缺席(legacy)全当主病灶,零回归。测试见 `tests/test_train_step_head_targets.py`(12 条)+ `tests/test_mixup_label_dominance.py`(12 项,2026-08-15)。

🔴 **监督闸门阈值 2 行修复(2026-08-15,loss 双峰 ~5/~0.8 的根因)**:`donor = lam == 0.0` → **`lam <= 0.35`**;`hard = donor | (lam == 1.0)` → **`donor | (lam >= 0.65)`**。根因:CutMix 分支的 `lams` 确由 DR-CutMix 判成 0.0/1.0(精确相等成立),但 **MixUp 分支的 `lam ~ Beta(0.2,0.2)` 严格落在 (0,1)** → 旧判据下 `hard` 全 False → 四头 valid 被整批乘 0。`prob=0.3` 且 CutMix/MixUp 各半 → 约 **15% 的 batch 四头零监督**,不报错、只是没学,表现为 loss 在两个值域间跳。**刻意只改 train_step.py 的闸门、不动 `src/dataset/augmentation/mixup.py`**——`lam` 还被 CBM 概念损失(`src/models/losses/components/concepts.py:39-105`,`resolve_mix_params` 后 SLR 软插值、`mask_weight` 门控)与 SLR(`slr.py:26-36` 按 `lam*orig + (1-lam)*perm` 插值)当**软加权系数**消费,在 mixup 侧硬化成 0/1 会连带改掉旧任务的数学;`mask_weight` 同理不动(它还门控概念损失)。对 CutMix **完全等价**(其 lams 已是 0.0/1.0,`0.0<=0.35`/`1.0>=0.65` 恒成立)。回归测试 `tests/test_mixup_label_dominance.py`(12 项,含反向保护"mixup.py 必须仍产出软 lam")。

### `metrics_official.py`(官方口径验证指标,阶段 2.4)

病灶级聚合(同 `lesion_id` 多帧概率取均值、标签取首)+ 官方筛选(金标准侧,非预测侧):**全部指标只取 `is_main==1 且 valid_*==1`**;ETE/T 再加金标准 `malignancy==1`;位置/良恶性/TI-RADS 不限恶性。指标:良恶意 `average_precision_score`(AUC-PR)、位置 `roc_auc_score(multi_class='ovr', average='weighted')`(剔除缺席类+重归一化,**2 类时走二分类分支**——sklearn 对 2 类拒收 [N,2] 矩阵)、ETE `cohen_kappa_score(weights='quadratic')`(官方只说加权 Kappa 未指线性/二次,取 quadratic 并标注假设)、T `roc_auc_score`(t_conf 当 score)、TI-RADS `cohen_kappa_score(weights='quadratic')`(不限恶性,与位置同、与ETE/T不同)。单类/空集返回 None 不抛异常。**训练侧代理性质**:无 IoU 匈牙利匹配(训练用 GT mask 等价"匹配必成功"),数字与官方线上分不可直接对齐,只用于相对比较与选模。sklearn 2 类分支与加权方向均由 `tests/test_metrics_official.py`(15 条)钉死。

### `trainer.py` (Thin Orchestrator)

`trainer.py` contains `train_epoch()` and `validate_epoch()` which delegate to `loop/train_step.py` and `loop/validate_step.py`.

### NaN Sentinel Diagnostics

The validation loop includes a **Val Diag** print at `batch_idx == 0`:
```python
print(f"  [Val Diag] pred_prob: mean={pred_mean:.4f}, std={pred_std:.4f}, min={...}, max={...}", flush=True)
```

### GradScaler Protocol & Optimizer Encapsulation

- `scaler.unscale_(optimizer)` before alpha gradient inversion
- `clip_grad_norm_(model.parameters(), max_norm=1.0)` after every backward
- Dual optimization is encapsulated: `criterion.mal_loss_fn.step_dual_variable("pre")` before `optimizer.step()`, and `"post"` afterwards.

**AMP 精度选择（fp16+GradScaler vs bf16 无 scaler）**：主训练管道（`train.py`）用 fp16 + GradScaler（上述协议）。但 **LoRA 微调（`scripts/train_thyroidxl.py`）与 malignancy 预训练（`pretrain_malignancy.py`）用 bf16 AMP、无 GradScaler**——bf16 指数范围同 fp32，无需 loss scaling。**血泪教训**：`train_thyroidxl.py` 曾用 fp16 默认 autocast，FT-LoRA 阶段 backbone LoRA 扰动把激活顶穿 fp16 的 65504 天花板，GradScaler scale 从 65536 一路减半仍失守，loss 爆到 1e99（epoch 13）。改 bf16 后 1300 步诊断无溢出。**凡 backbone 参与微调/存在深层扰动的训练，优先用 bf16**；纯冻结 backbone 的浅层训练 fp16 尚可。bf16 路径下仍保留 `clip_grad_norm_(max_norm=1.0)`。

**冻结 backbone 必须同时冻结其 BatchNorm running stats（2026-07-10 血泪教训）**：`train_thyroidxl.py` 的 LP-FT-LoRA 训练曾**遗漏**调用 `set_frozen_modules_eval()`——虽然 backbone 原始权重 `requires_grad=False`，但 `model.train()` 递归下去后其内部 `BatchNorm` 仍处于训练模式，`running_mean`/`running_var` **只受 `module.training` 控制、与 `requires_grad` 无关**，会持续随每个 batch 的真实激活值漂移（本地 A800 实测：单个 batch 就能让某层 `running_var` 漂移 22.98）。50 epoch 下来把一个训练前就已临界（`running_var` 低至 3.5e-42，EfficientNet-B2 预训练权重自带）的 BN 通道冲向数值病态区间，经 backbone 逐层复合放大，导致 CPU/GPU、LoRA-结构/合并-结构两种等价 forward 之间相对误差炸到 70%~1240%，checkpoint 因此报废重训。**规则**：任何训练脚本只要有"参数冻结但仍 `model.train()`"的 backbone/子模块，必须在 `train_one_epoch`（或等价逐 epoch 入口）里紧跟 `model.train()` 之后调用 `from src.training.loop.bn_freeze import set_frozen_modules_eval; set_frozen_modules_eval(model)`——`train.py` 主管线的 `trainer.py:35` 早已采用这个模式，新训练脚本必须照抄，不能假设"冻结 = 完全不变"。验收方法：跑几个训练 step，`assert torch.equal(bn.running_var_before, bn.running_var_after)` 直接断言冻结子模块的 BN buffer 零漂移。

---

## 3. AUTHENTICITY CLASSIFIER (`authenticity_trainer.py`)

Lightweight MLP trained on 13-dim features (GLCM + freq + ELA + DND speckle + TAI anisotropy). Runs **before** main training by default. Saves to `checkpoints/authenticity_model.pth`. Main model weights MUST NOT include these parameters.

Use `python train.py --skip-auth` to skip when `checkpoints/authenticity_model.pth` already exists. If the flag is set but weights are missing, training logs a warning and runs authenticity training anyway.

Note: Renamed from `authenticity.py` to `authenticity_trainer.py` to avoid confusion with `src/gatekeeper/authenticity.py` (feature extraction + inference module).
