# src/models/losses/AGENTS.md - Loss Functions Specifications

## 1. 多任务损失架构

### 主损失函数 (`multitask.py`)
- `MultiTaskLoss`: 组合分割BCE+Dice、TI-RADS概念SLR、良恶性AUCM+SupCon
- 使用Homoscedastic Uncertainty Weighting自动平衡任务权重
- 支持FAMO动态权重调度

### 组件化设计 (`components/`)
- `segmentation.py`: `compute_segmentation_loss()` — BCE+Dice,支持`valid_mask`
- `concepts.py`: `compute_concept_losses()` — TI-RADS五项SLR+临床先验正则
- `malignancy.py`: `compute_malignancy_losses()` — AUCM+SupCon
- `famo_mix.py`: `apply_famo_weighting()` — FAMO权重应用与更新
- `regularizers.py`: `compute_regularizers()` — Latent L2+SPIN L1

---

## 2. PHASE 4 新增：掩膜多任务损失

### 背景
- 多源数据标注异质化：ThyroidXL全标注，ThyUS2Path仅病理，DDTI仅分割
- 训练时需自动屏蔽无标注任务的梯度，避免负梯度惩罚

### 实现1：组件内部支持（唯一正确路径）
**分割损失** (`components/segmentation.py`):
- `targets['valid_mask']`: Tensor[B]，1=有分割mask，0=无
- per-sample计算BCE/Dice后`sum() / valid_count`
- 无效样本**真正零梯度**（未进入计算图）

**概念损失** (`components/concepts.py`):
- 暂未实现per-sample mask（P1待办）
- 当前全batch计算，异构源暂靠 adapter 层 `task_mask` valid=0 从源头不产假标签

**良恶性损失** (`components/malignancy.py`):
- 已支持`label_validity`参数（line 31-57）
- 无效样本不进入AUCM dual variables更新

### 已废弃：MaskedMultiTaskLoss 后处理包装器（commit 54f0302 删除）
- 原 `masked_multitask.py` 用后处理缩放损失权重实现 task_mask，但 `base_loss` 返回时
  计算图已构建、per-task loss 已 reduce 成标量，**再乘 valid_ratio 只缩小损失量级，
  无标注样本的梯度仍经标量 loss 回传污染 encoder**——不是真屏蔽。且它重算 total_loss
  时绕过了 FAMO 动态加权。故整体删除。
- 唯一保留：任务名→loss key 映射表 `TASK_TO_LOSS_KEYS`，已搬入 `multitask.py` 作模块常量
  （仅文档性/调试用，非屏蔽逻辑）。
- 正确做法统一走「实现1」：`multitask.py` forward 把 `task_mask['mask']` 注入
  `targets['valid_mask']`，由 components 在 per-sample 层真正零梯度屏蔽。

### 使用方式
```python
# 唯一路径：利用组件内 valid_mask / label_validity 支持
targets['valid_mask'] = task_mask['mask'].float()        # 分割 per-sample 屏蔽
targets['label_validity'] = task_mask['malignancy']      # 良恶性 AUCM 屏蔽

loss_fn = MultiTaskLoss(...)
total_loss, loss_dict = loss_fn(outputs, targets, task_mask=task_mask)
# multitask.py forward 内部自动把 task_mask['mask'] 注入 targets['valid_mask']
```

### Gate2审查结果（commit d4c00ff → 54f0302 清理）
- **P0-1修复**：multitask.py利用现有`valid_mask`支持，避免后处理梯度泄漏
- **54f0302 清理**：删除劣质的 MaskedMultiTaskLoss 后处理路径（已被上述真屏蔽取代）
- **待办**：概念损失per-sample mask（P1）

---

## 3. 关键约束（继承自原AGENTS.md）

### MixUp目标插值保护
- AUCM dual variables (`a`, `b`, `alpha`)必须使用**未插值**的目标数组
- `multitask.py`已在`compute_malignancy_losses`前解析mixup参数

### 数值稳定
- 所有除法有`eps=1e-8`保护
- NaN Sentinel打印到stdout（flush=True）

### 优化器隔离
- vCLUB使用独立optimizer，禁止主optimizer更新vCLUB参数
- 见`src/training/AGENTS.md §1`

---

## 4. 测试覆盖

- `tests/test_masked_loss.py`: 10个测试验证valid_mask零梯度隔离
  - `test_all_invalid_zero_gradient`: 确保无效样本无梯度流
  - `test_partial_valid_excludes_invalid_samples`: 部分有效样本正确屏蔽
  - `test_all_valid_legacy_compat`: 向后兼容性

执行: `pytest tests/test_masked_loss.py -v`（161 passed total suite）
