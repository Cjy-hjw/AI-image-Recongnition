# 重构前后对比 (2026-06-23)

## 一句话总结

把 300+ 行的上帝函数拆成了 11 个独立模块、清了死代码、修了数据泄漏、校准了评估尺度、补齐了文档覆盖。全面通过 43 项测试 + 7 道验证门。**服务器已正常训练中(Epoch 4, Val Dice 0.921)**。

---

## 之前哪里离谱

| 问题 | 之前 | 现在 |
|:---|:---|:---|
| **上帝函数** | `trainer.py` 含混杂 BN 冻结/SLR/vCLUB;`losses/multitask.py` 含全部子损失混杂 | 拆为 `training/loop/`(5文件) + `losses/components/`(6文件),单一职责 |
| **死代码** | `AdaptiveVSAndAsymmetricLoss`/`get_patient_groups`/`create_optimizer`/`log_vars` 等残留;`augmentation.py` 和 `augmentation/` 包名冲突 | 全部删除,`augmentation/` 独立包 |
| **评估路径不统一** | `evaluate.py` 直接 `model(img)`,缺少 `run_forward(pipeline)`,与生产预测(`predict.py`)不一致 | 统一经 `run_forward(model, img, precision, device)`,fp32 数值与裸 forward 一致 |
| **validation gamma 漂移** | `validate_step` 用 `(epoch-20)/20` 动态算 `gamma_prior`,validation 指标随 epoch 漂 | 固定 `gamma_prior=1.0`,validation 稳定 |
| **Dice 计算口径** | 全局平均,空 mask 时除零 | 逐样本 `dim=(2,3)` + epsilon 守卫,双空自然=1.0 |
| **PHI 字段残缺** | DICOM 脱敏仅 5 字段 | 扩展至 21 字段(HIPAA Safe Harbor 全量) |
| **伪患者分组** | `PAT_{img_id//2}` 把相邻两张图硬归为同一"患者",制造假隔离 | 纯随机图像级划分,`patient_id = {source}_{image_id}`(tv_/ho_ 前缀防跨集碰撞) |
| **gate5 空洞检查** | 仅查 train↔val(合成 ID 恒 PASS),holdout 未守卫 | 双检查:train↔val + holdout↔trainval 图像级 0 重叠 |
| **AGENTS.md 缺失** | `scripts/data/`,`src/utils/`,`tests/` 无文档;多处内容过时 | 11 个 AGENTS.md 全覆盖,架构/训练/推理/数据/测试均有独立文档 |
| **Windows 编码** | `verify_model.py` 在 Windows GBK 控制台输出 ✔/✘ 报 `UnicodeEncodeError` | 入口强制 `stdout.reconfigure(encoding='utf-8')` |
| **ruff 警告** | 20+ F541/E401/F401/F841/E741 散落 | 全项目零警告 |
| **deploy 缓存泄漏** | `deploy.py` 上传 `.ruff_cache`/`.pytest_cache`,且误过滤 `scripts/data/` | 缓存过滤 + `scripts/data/` 保留 |
| **FutureWarning** | `torch.cuda.amp.autocast` 已废弃 | 迁移至 `torch.amp.autocast('cuda')` |

---

## 重构做了什么

### Phase A — 解耦 + 死代码清理
- 拆分 `training/loop/{train_step,validate_step,bn_freeze,slr,vclub}.py`
- 拆分 `losses/components/{segmentation,concepts,malignancy,famo_mix,regularizers}.py`
- 拆分 `inference/{model_runner,export,io,postprocess,gatekeeper_runner}.py`
- 拆分 `dataset/augmentation/{transforms,mixup}.py`
- 拆分 `scripts/verify/{runner,gates/gate1-7}.py`
- CBMHead 返回 `CBMHeadOutput` dataclass(不再裸 tuple)
- 删除 `AdaptiveVSAndAsymmetricLoss`/`get_patient_groups`/`create_optimizer`/`log_vars`/`ortho` loss 项

### Phase B — 评估尺子校准
- `evaluate.py` 统一经 `run_forward(fp32)` 前向
- Dice 逐样本 `dim=(2,3)` + epsilon
- `validate_step` 固定 `gamma_prior=1.0`
- Grad-CAM 双主干分支(cls/seg 目标层互异)
- PHI 字段 5→21

### Phase D — 数据划分去伪分组
- `split_tn3k.py`: 抽出 `_split_samples` 纯函数,废弃 `PAT_{//2}` 伪分组
- `patient_id = f"{prefix}_{image_id}"` (tv_=trainval, ho_=holdout)
- `gate5`: 补 holdout↔trainval 图像级重叠检查

---

## 验证结果

```
43 passed in 8.03s                    # 完整测试套件
7/7 验证门通过                          # QA-VP §3 合规
ruff: 全项目零警告                      # 代码规范
pyright: 仅预存问题(第三方stub滞后)       # 类型安全
```

---

## 服务器状态

```
训练进行中 (2026-06-23 14:34 启动)
Stage 1: Seg+Concepts Warmup (Epoch 1-40)
Epoch 04: Val Dice 0.921, Train Loss 0.435
GPU: NVIDIA A800 80GB (100% util, 47.5GB)
环境: Python 3.10.18 + PyTorch 2.6.0 + CUDA 12.4
```

---

## 新增文件清单

| 位置 | 用途 |
|:---|:---|
| `src/training/loop/` (6 文件) | 训练/验证/bn/slr/vclub 独立模块 |
| `src/models/losses/components/` (6 文件) | 子损失解耦 |
| `src/inference/` (6 文件) | 推理管线 |
| `src/dataset/augmentation/` (3 文件) | 增强独立包 |
| `scripts/verify/gates/` (7 文件) | 7 道验证门 |
| `tests/test_phase_refactor.py` | Phase A/B/D 回归守卫(14 项) |
| `scripts/data/AGENTS.md` | 数据预处理文档 |
| `src/utils/AGENTS.md` | 工具函数文档 |
| `tests/AGENTS.md` | 测试套件文档 |
| `QA_VP.md` | 质量验证协议 |
