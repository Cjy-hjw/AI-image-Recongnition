# tests/AGENTS.md — 回归测试套件

## Test Files

| File | Scope | Tests | 对应 Phase |
|:---|:---|:---|:---|
| `test_physical_prior.py` | `DifferentiablePhysicalPriorExtractor` — 形态学腐蚀、软圆形度、EMA 熵门控 | 18 | — |
| `test_cbm_head.py` | `ConceptEmbeddingCBMHead` — 输出 dataclass 字段、vCLUB forward、latent 维度 | 16 | — |
| `test_phase_refactor.py` | 重构回归守卫 — 解耦/死代码/评估尺子/数据划分 | 14 | Phase A/B/D |

---

## `test_phase_refactor.py` — 重构不变式(43 tests total, 此文件 14)

### Phase A: 解耦 + 死代码清理
| Test | 验证内容 |
|:---|:---|
| `test_cbm_head_returns_keyed_dataclass` | CBMHeadOutput 字段齐全 |
| `test_dead_code_removed` | AdaptiveVSAndAsymmetricLoss / get_patient_groups / create_optimizer / log_vars / ortho 已删除 |
| `test_forward_output_matches_contract` | forward 输出键 == ModelOutputs 注解集合 |
| `test_forward_deterministic` | 同输入两次 forward 完全一致(eval 下) |
| `test_loss_forward_finite_no_ortho` | 损失有限、ortho key 已移除 |

### Phase B: 评估尺子校准
| Test | 验证内容 |
|:---|:---|
| `test_run_forward_fp32_equals_bare_forward` | evaluate 路径数值 == 裸 forward(gamma=1.0) |
| `test_dice_double_empty_is_one_via_epsilon` | Dice 双空经 epsilon 自然=1.0 |
| `test_gradcam_dual_backbone_branches` | cls/seg 目标层相异 |
| `test_phi_fields_expanded` | PHI 字段 ≥ 15 |
| `test_validate_step_fixed_gamma_prior` | validate_step 源码中 gamma_prior=1.0、无 epoch 漂移 |

### Phase D: 数据划分去伪分组
| Test | 验证内容 |
|:---|:---|
| `test_split_samples_disjoint` | train/val 图像 ID 互斥 |
| `test_split_samples_ratio` | 划分比例 ≈ split_ratio |
| `test_split_samples_reproducible` | 固定 seed 可复现 |
| `test_split_samples_patient_id_is_image_id` | 每张图独立身份,全集覆盖 |

---

## 运行

```bash
# 完整套件
python -m pytest tests/ -q

# 仅重构回归
python -m pytest tests/test_phase_refactor.py -q
```

所有 test 函数既可加入 `pytest`,也可作为 `python tests/test_xxx.py` 独立运行(每个文件底部有 `if __name__ == "__main__": pytest.main(...)`)。
