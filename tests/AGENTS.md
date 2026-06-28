# tests/AGENTS.md — 回归测试套件

## Test Files

| File | Scope | Tests | 对应 Phase |
|:---|:---|:---|:---|
| `test_physical_prior.py` | `DifferentiablePhysicalPriorExtractor` — 形态学腐蚀、软圆形度、EMA 熵门控 | 14 | — |
| `test_cbm_head.py` | `ConceptEmbeddingCBMHead` — 输出 dataclass 字段、vCLUB forward、latent 维度 | 15 | — |
| `test_phase_refactor.py` | 重构回归守卫 — 解耦/死代码/评估尺子/数据划分/续训态持久化/Grad-CAM 端到端 | 19 | Phase A/B/C/D |
| `test_gatekeeper.py` | 合成引擎确定性+输出契约、dedup 近重复检测、copy-move 检测(召回~20-40%、干净图零误报)、authenticity 特征三类可分;splicing 检测器「轴对齐长直缝+角点正交相交」几何判据(干净图 0/80 零误报、真实组织拼接召回~84%、返回契约) | 15 | — |
| `test_masked_loss.py` | Masked Task-Specific Loss — 无效样本零梯度贡献、全有效与旧路径数值一致(向后兼容) | 10 | — |
| `test_video.py` | 动态图/cine 管线 — 抽样下标边界(含 max_frames==1 除零守卫)、帧级聚合(最大帧判定/关键帧映射/空安全)、灰度转换、视频解码往返(codec 缺失则 skip)、gatekeeper 去重库注入复用 | 17 | — |

---

## `test_phase_refactor.py` — 重构不变式(全套 90 passed,此文件 19)

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
| `test_gradcam_generates_valid_heatmap_end_to_end` | 端到端:前向+反向产出 [0,1] 归一化、512×512、无 NaN 热图(堵 backward 路径集成盲区) |
| `test_predict_passes_gradcam_flag_outside_inference_mode` | 静态守卫:predict.py 有 `--enable_gradcam` 开关,且 Grad-CAM 调用在 run_forward(inference_mode)之后 |
| `test_phi_fields_expanded` | PHI 字段 ≥ 15 |
| `test_validate_step_fixed_gamma_prior` | validate_step 源码中 gamma_prior=1.0、无 epoch 漂移 |

### Phase D: 数据划分去伪分组
| Test | 验证内容 |
|:---|:---|
| `test_split_samples_disjoint` | train/val 图像 ID 互斥 |
| `test_split_samples_ratio` | 划分比例 ≈ split_ratio |
| `test_split_samples_reproducible` | 固定 seed 可复现 |
| `test_split_samples_patient_id_is_image_id` | 每张图独立身份,全集覆盖 |

### Phase C: 损失正确性 + 续训态持久化
| Test | 验证内容 |
|:---|:---|
| `test_famo_state_dict_round_trip` | FAMO w_logits/prev_loss 序列化往返精确恢复 |
| `test_famo_load_rejects_task_count_mismatch` | 任务数不匹配的 checkpoint 显式报错,不静默广播 |
| `test_criterion_persists_dual_params_and_lambda` | criterion.state_dict 覆盖 AUCM 对偶参数 + lambda_latent,往返恢复 |

---

## 运行

```bash
# 完整套件
python -m pytest tests/ -q

# 仅重构回归
python -m pytest tests/test_phase_refactor.py -q
```

所有 test 函数既可加入 `pytest`,也可作为 `python tests/test_xxx.py` 独立运行(每个文件底部有 `if __name__ == "__main__": pytest.main(...)`)。
