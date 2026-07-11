# tests/AGENTS.md — 回归测试套件

## Test Files

| File | Scope | Tests | 对应 Phase |
|:---|:---|:---|:---|
| `test_physical_prior.py` | `DifferentiablePhysicalPriorExtractor` — 形态学腐蚀、软圆形度、EMA 熵门控；**buffer contiguity + state_dict 往返回归**(2026-07-10：`x_coords`/`y_coords` 非 contiguous 曾致 `load_state_dict()` 崩溃，任何 checkpoint 均无法被 `predict.py`/`evaluate.py` 真正加载)；**e_prior→seg_backbone 端到端梯度耦合**(2026-07-11 Gate2 review 非阻塞建议补齐,2例:全模型前向`gamma_prior=0.0`→seg_backbone精确0梯度、`gamma_prior=1.0`→非零有限梯度且量级健全；此前该合法耦合通路零端到端测试覆盖,只测过独立`DifferentiablePhysicalPriorExtractor`模块合成leaf张量,测不到是否真的按`gamma_prior`预期工作) | 18 | — |
| `test_cbm_head.py` | `ConceptEmbeddingCBMHead` — 输出 dataclass 字段、vCLUB forward、latent 维度 | 15 | — |
| `test_phase_refactor.py` | 重构回归守卫 — 解耦/死代码/评估尺子/数据划分/续训态持久化/Grad-CAM 端到端 | 19 | Phase A/B/C/D |
| `test_gatekeeper.py` | 合成引擎确定性+输出契约、dedup 近重复检测、copy-move 检测(召回~20-40%、干净图零误报)、authenticity 特征三类可分(含 DND/TAI 物理散斑 13 维 + Rayleigh m≈1 方向锁定);splicing 检测器「轴对齐长直缝+角点正交相交」几何判据(干净图 0/80 零误报、真实组织拼接召回~84%、返回契约);**dedup 后端可插拔**(SSCD 权重探测降级 + 输出维度 fail-fast + 按后端余弦阈值映射 sscd 0.86 / resnet18 0.995,守一票否决红线);**CAT-Net v2 DCT/BAG 取证**(块级 DCT 可逆性、JPEG 栅格显著度>未压缩、BAG 相位不一致定位、可插拔降级 dct-bag、干净 JPEG 8 seed 零误报 + 未压缩弃权 + 非整倍 stride 红线回归、三路并集契约、**catnet本次forward失败不误判block_level_hard回归**2026-07-10);**合成引擎升级**(泊松/羽化拼接 `make_splice_poisson` 无硬缝+小图守卫;域随机化 `domain_randomize` = Nakagami 散斑+模糊 k∈{3,5,7}+JPEG Q∈[15,75],确定性+JPEG 栅格植入) | 42 | — |
| `test_masked_loss.py` | Masked Task-Specific Loss — 无效样本零梯度贡献、全有效与旧路径数值一致(向后兼容) | 10 | — |
| `test_video.py` | 动态图/cine 管线 — 抽样下标边界(含 max_frames==1 除零守卫)、帧级聚合(最大帧判定/关键帧映射/空安全)、灰度转换、视频解码往返(codec 缺失则 skip)、gatekeeper 去重库注入复用(注入stub auth_module隔离真实性检测依赖);**`_to_gray` 通道鲁棒性**((H,W)/(H,W,1)/(H,W,3)/RGBA(H,W,4)/channel-first(3,H,W)/多光谱 全收敛 2D,杜绝 3D 漏过致 5 维崩溃;RGB vs BGR 通道序加权区分;16-bit 归一化);**彩色单图 DICOM 回归**(monkeypatch 伪造 pydicom,守 io.py 单图路径经 `_to_gray` 收敛 (H,W,3)→2D) | 21 | — |
| `test_authenticity_trainer.py` | 真实性分类器训练器 smoke — 合成伪超声喂训练器跑通,产出 dict(state_dict+feat_mean+feat_std);推理端 AuthenticityModule 加载并应用归一化、predict 返回 softmax(守卫训练/推理归一化口径一致);**权重缺失硬报错回归**(2026-07-10:此前静默退化为随机初始化分类,一票否决门形同虚设) | 3 | — |
| `test_report_generator.py` | `report_generator.py`(2026-07-10 新增,此前零测试覆盖) — TI-RADS评分表边界(ACR白皮书对照、TR1-5分级区间)、**越界idx硬报错回归**(comp/echo/shape/margin/foci越界此前静默按0分处理,实测margin_idx=5可致TR3→TR2静默跨越穿刺决策边界)、**malignancy_prob=None/nan/inf硬报错回归**(此前nan>=0.5恒False导致无报错输出"良性倾向",癌症筛查假阴性风险) | 13 | — |
| `test_multisource.py` | 多源数据集骨架 — 源级 label_mask(默认全有效向后兼容、源 mask 标注缺失任务、meta label_mask 覆盖源 mask);`MultiSourceThyroidDataset` 拼接长度、源级 mask 生效、采样权重反映源 ratio、空源不崩 | 7 | — |
| `test_smoke_entrypoints.py` | CLI 入口端到端冒烟 — 把 `predict.py`/`video_predict.py` 当**子进程实跑**(随机权重,build_model 缺 checkpoint 降级随机+warning 不报错):predict 噪声图路径(进程正常退出、落 result.json、status∈{REJECTED,PASSED,ERROR}——本地/CI无真实authenticity权重时ERROR是2026-07-10修复后的预期新行为,非回归)+ video `--skip_gatekeeper` 路径(推理/后处理/报告/JSON/Grad-CAM 四产物齐全、JSON 中文可还原)+ **权重不匹配 ERROR 兜底回归**(predict.py + video_predict.py 各一例;2026-07-10：build_model 硬报错后若无兜底会致容器裸崩溃零输出，测试构造 mismatched checkpoint 断言 exit 0 + `result.json.status=="ERROR"`；此前只测了video_predict.py一侧,同轮补齐predict.py侧因二者gatekeeper/build_model调用顺序不同,分支可达性此前从未验证过)+ **输入加载失败 ERROR 兜底回归**(2026-07-11 新增,4例:predict.py 图片不存在/空文件、video_predict.py 视频不存在/空文件——此前 `load_input_image`/`load_video_frames` 返回 `None` 时两脚本在 try 块外直接 `return`,只 print 不落 result.json,exit code 仍 0 但 output_dir 空目录,比 REJECTED 还差且绕开 except Exception 安全网,现已在 return 前补 `save_rejection_json`;video侧空文件用例2026-07-11 Gate2 review 非阻塞建议补齐,此前只测missing未测empty,覆盖度不对称)+ **`--config` 加载失败 ERROR 兜底回归**(2026-07-11 新增,`TestConfigLoadFailureErrorFallback` 4例:predict.py config缺失/格式错误、video_predict.py config缺失/格式错误——同一类 try 外裸操作问题,这次补配置加载这一处,见 `model_runner.load_config`;video侧格式错误用例2026-07-11 Gate2 review 非阻塞建议补齐,此前只测missing未测malformed,覆盖度不对称)。**只兜「管线连通+产物落盘+JSON 可解析」,不把 gatekeeper 在随机噪声上的判定当硬断言**——该判定依赖 GLCM/频域统计浮点细节,随平台(numpy 1.x 容器 vs 2.x 本地+不同 BLAS)翻转,本地拒、容器可能放行;TODO 真实数据复核其平台稳定性。补 99 组件级单测对入口「接线」的集成盲区;codec 缺失则 skip | 12 | — |
| `test_calibration.py` | Post-hoc 温度缩放骨架 — **降级正确性**(缺参数/损坏文件/全 T=1.0 → `decode_predictions` 输出与无校准逐位相等)、**单调性**(T>1 压低 max 概率向 0.5 收缩但不改 argmax)、**ECE 数值**(完美校准≈0、过度自信全错=1.0、一维正类概率自动补两列)、**拟合**(`fit_temperature` 返正温度、`from_file` roundtrip);非法温度(≤0)拒绝降级 T=1.0 | 11 | — |
| `test_pathology_dataset.py` | PathologyDataset — 标签广播(患者级→图像级)、患者分组无泄漏、batch列顺序容错(path,patient_name反列)、分层划分、valid键契约(`valid_malignancy=1`/`valid_mask=0`/其余`valid_*=0`)、输出形状dtype | 20 | — |
| `test_pathology_mil_dataset.py` | PathologyMILDataset + MIL模型 — bag聚合(同patient同bag)、label一致性、顺序确定性、形状`[N,1,H,W]`、max_frames截断、患者级class_counts、train/val无泄漏、GatedAttentionPooling输出`[1,1408]`+weights和≈1、ThyroidMILNet前向`[N,1,512,512]→[1,2]`、冻结后backbone无梯度 | 13 | — |
| `test_entrypoint_dispatch.py` | **扩展名分发**(mp4/avi/mov/mkv/dcm/dicom 含大写→video_predict、png/jpg/… →predict、flag 透传,用假 python shim 拦 exec 不真跑推理)、**PRODUCTION=1 防篡改阻断**(predict 哈希不符 / src 树哈希不符均在 `exec python` 前 `exit 1`,`SHIM_ARGS not in stdout` 实证「exec 前退出」;开发模式正确绕过)、**src 树指纹 parity**(source `_src_tree_hash.sh` 真函数 vs `seal_container.py` Python 实现字节级一致,反向验证改坏函数必 FAIL 防 drift 假绿) | 19 | — |
| `test_bn_freeze.py` | `set_frozen_modules_eval`(2026-07-10 新增，回归测试) — 冻结子模块 BN running_var/mean 零漂移、不冻结时确实漂移(反向验证测试有效性)、可训练子模块不被误冻、LoRA 场景下同级独立 BN 子模块仍正确冻结 | 4 | — |
| `test_model_runner.py` | `build_model`(2026-07-10 新增，回归测试) — 裸 state_dict 正常加载、`{"state_dict":...}` 包装格式正确解包、unexpected/missing keys 硬报错(拒绝静默部分加载)、checkpoint 缺失时仍降级随机初始化(向后兼容)；`run_forward`(2026-07-11 新增，两轮修复，`TestRunForwardCpuFp16Fallback`+`TestRunForwardGpuPrecisionRegression`) — CPU+fp16 autocast "假降级"修复回归、**CPU+fp32 不崩溃**(第一轮修复误把此分支 dtype 从 bfloat16 改成 float16 引入新回归，CI test job 实测抓到，第二轮改为 CPU 上 dtype 恒 bfloat16)、bf16-on-cpu 未被连带破坏、GPU fp32/fp16/bf16 三值回归(skipif 无 CUDA)；`load_config`(2026-07-11 新增，`TestLoadConfig`) — 合法配置正常解析、文件不存在抛 `FileNotFoundError`、YAML 格式错误抛异常、缺 `environment.device` 键抛 `KeyError`(供 predict.py/video_predict.py 的 try/except Exception 统一兜底) | 15 | — |
| `test_gate8_adaptor_isolation.py` | `gate8_adaptor_isolation`(2026-07-11 新增) — TaskAlignmentAdaptor cls→seg 梯度隔离审计的正面/负面回归；正面:真实代码(`.detach()` 正确)+ 模拟前序 Gate 遗留的 `requires_grad=False` 污染状态下仍判定通过、`seg_backbone` 梯度精确 `0.0`、`gamma`/`requires_grad` 复位无副作用(2026-07-11 Gate2 review 非阻塞建议补齐:此前 finally 只复原 gamma 没复原 requires_grad,依赖"Gate8 是最后一个 Gate"这个隐性顺序假设,现改为逐参数显式记录+复原);负面:monkeypatch `torch.Tensor.detach` 为恒等函数模拟误删,断言 Gate 检测到泄漏并返回 False(踩坑记录:`TaskAlignmentAdaptor.gamma` 零初始化会让正确/错误 detach 都测出梯度 0,Gate 内部强制拨到非零才能让负面测试真正生效,详见该 Gate 源码注释) | 2 | — |

---

## `test_phase_refactor.py` — 重构不变式(全套 **230 passed**，此文件 19)

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
