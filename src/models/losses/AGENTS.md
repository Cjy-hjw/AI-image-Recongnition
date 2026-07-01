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

### 实现1：组件内部支持（推荐）
**分割损失** (`components/segmentation.py`):
- `targets['valid_mask']`: Tensor[B]，1=有分割mask，0=无
- per-sample计算BCE/Dice后`sum() / valid_count`
- 无效样本**真正零梯度**（未进入计算图）

**概念损失** (`components/concepts.py`):
- 暂未实现per-sample mask（P1待办）
- 当前全batch计算，异构源需wrapper缩放

**良恶性损失** (`components/malignancy.py`):
- 已支持`label_validity`参数（line 31-57）
- 无效样本不进入AUCM dual variables更新

### 实现2：包装器（过渡方案）
**MaskedMultiTaskLoss** (`masked_multitask.py`):
- 包装`MultiTaskLoss`，后处理缩放损失权重
- **局限**：后处理无法真正零梯度（梯度已流入encoder）
- 用途：快速验证task_mask逻辑，等待组件内部mask完善

### 使用方式
```python
# 方式1：利用现有valid_mask支持（分割+良恶性）
targets['valid_mask'] = task_mask['mask'].float()  # 分割
targets['label_validity'] = task_mask['malignancy']  # 良恶性

loss_fn = MultiTaskLoss(...)
total_loss, loss_dict = loss_fn(outputs, targets)

# 方式2：包装器（过渡）
base_loss = MultiTaskLoss(...)
masked_loss = MaskedMultiTaskLoss(base_loss)
total_loss, loss_dict = masked_loss(outputs, targets, task_mask=task_mask)
```

### Gate2审查结果（commit d4c00ff）
- **P0-1修复**：multitask.py利用现有`valid_mask`支持，避免后处理梯度泄漏
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
