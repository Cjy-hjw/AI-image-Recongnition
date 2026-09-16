# tests/AGENTS.md — 回归测试套件

## Test Files

| File | Scope | Tests | 对应 Phase |
|:---|:---|:---|:---|
| `test_physical_prior.py` | `DifferentiablePhysicalPriorExtractor` — 形态学腐蚀、软圆形度、EMA 熵门控；**buffer contiguity + state_dict 往返回归**(2026-07-10：`x_coords`/`y_coords` 非 contiguous 曾致 `load_state_dict()` 崩溃，任何 checkpoint 均无法被 `predict.py`/`evaluate.py` 真正加载)；**e_prior→seg_backbone 端到端梯度耦合**(2026-07-11 Gate2 review 非阻塞建议补齐,2例:全模型前向`gamma_prior=0.0`→seg_backbone精确0梯度、`gamma_prior=1.0`→非零有限梯度且量级健全；此前该合法耦合通路零端到端测试覆盖,只测过独立`DifferentiablePhysicalPriorExtractor`模块合成leaf张量,测不到是否真的按`gamma_prior`预期工作) | 18 | — |
| `test_cbm_head.py` | `ConceptEmbeddingCBMHead` — 输出 dataclass 字段、vCLUB forward、latent 维度 | 15 | — |
| `test_phase_refactor.py` | 重构回归守卫 — 解耦/死代码/评估尺子/数据划分/续训态持久化/Grad-CAM 端到端;2026-08-11 R6 新增 `test_famo_directional_correctness`(FAMO 权重更新方向回归锁:救最差任务,慢任务 logit/weight 应更高) | 22 | Phase A/B/C/D |
| `test_gatekeeper.py` | 合成引擎确定性+输出契约、dedup 近重复检测、copy-move 检测(召回~20-40%、干净图零误报)、authenticity 特征三类可分(含 DND/TAI 物理散斑 13 维 + Rayleigh m≈1 方向锁定);splicing 检测器「轴对齐长直缝+角点正交相交」几何判据(干净图 0/80 零误报、真实组织拼接召回~84%、返回契约);**dedup 后端可插拔**(SSCD 权重探测降级 + 输出维度 fail-fast + 按后端余弦阈值映射 sscd 0.86 / resnet18 0.995,守一票否决红线;resnet18 降级后端权重缺失时 try 联网下载,只有联网本身失败才转 `RuntimeError`——联网成功/失败两条路径各一例回归,防止离线容器要求与 CI/开发机联网能力的回归);**CAT-Net v2 DCT/BAG 取证**(块级 DCT 可逆性、JPEG 栅格显著度>未压缩、BAG 相位不一致定位、可插拔降级 dct-bag、干净 JPEG 8 seed 零误报 + 未压缩弃权 + 非整倍 stride 红线回归、三路并集契约、**catnet本次forward失败不误判block_level_hard回归**2026-07-10);**合成引擎升级**(泊松/羽化拼接 `make_splice_poisson` 无硬缝+小图守卫;域随机化 `domain_randomize` = Nakagami 散斑+模糊 k∈{3,5,7}+JPEG Q∈[15,75],确定性+JPEG 栅格植入);**2026-07-12 统一软flag制度(B2/B3)+code-review补强**——`test_geometric_never_hard_rejects_even_when_catnet_ok`(copy_move/splice 与 CatNet 状态解耦回归,monkeypatch强制命中确定性验证,取代早期依赖合成图自然触发的pytest.skip脆弱写法)+ `TestBagHardRejectToggle`(BAG 默认软flag、`bag_hard_reject=True` 时恢复硬拒绝,同样改为monkeypatch确定性触发)+ `test_catnet_own_hit_sets_hard_reject_source`/`test_no_hard_reject_leaves_source_none`(新增 `last_hard_reject_source` 字段回归,供 REJECTED reason 精确标注命中源)。**QA_VP §2.3 补课**(2026-07-12):`test_catnet_forward_failure_this_call_does_not_promote_geometric_to_hard` 也补做同样处理——原依赖合成图自然触发 `detect_splicing`、不触发时 `pytest.skip` 的脆弱写法,同样改为 monkeypatch 强制命中,消除该测试最后一处"检测器行为被意外破坏时可能静默 skip 而非 fail"的风险 | 49 | — |
| `test_masked_loss.py` | Masked Task-Specific Loss — 无效样本零梯度贡献、全有效与旧路径数值一致(向后兼容) | 10 | — |
| `test_video.py` | 动态图/cine 管线 — 抽样下标边界(含 max_frames==1 除零守卫)、帧级聚合(最大帧判定/关键帧映射/空安全)、灰度转换、视频解码往返(codec 缺失则 skip)、gatekeeper 去重库注入复用(注入stub auth_module隔离真实性检测依赖);**`_to_gray` 通道鲁棒性**((H,W)/(H,W,1)/(H,W,3)/RGBA(H,W,4)/channel-first(3,H,W)/多光谱 全收敛 2D,杜绝 3D 漏过致 5 维崩溃;RGB vs BGR 通道序加权区分;16-bit 归一化);**彩色单图 DICOM 回归**(monkeypatch 伪造 pydicom,守 io.py 单图路径经 `_to_gray` 收敛 (H,W,3)→2D);**2026-07-12 统一软flag制度(B1)+code-review补强**——`TestGatekeeperSoftFlagPolicy`(8例,整合层测试 `run_gatekeeper_audit`):authenticity 判 phantom/synthetic 均不再触发 REJECTED(仅软flag)、去重/篡改命中仍硬拒绝不受影响、REJECTED 产物里 `is_phantom`/`is_gan` 字段仍作审计保留且以"soft-flag:"标注形式出现在人类可读 reason 里(而非完全消失)、reason 按 `tamper_detector.last_hard_reject_source` 精确标注 CatNet/BAG 命中源(3例:无该属性时向后兼容退化为通用文案 + CatNet/BAG 各一例);**2026-07-29 enforcement_mode 软化**——`TestGatekeeperSoftFlagPolicy` 改为显式 `enforcement_mode="hard"` 以继续测硬拒绝语义(生产默认已是 soft),phantom/synthetic 两例改断言 `would_reject_in_hard_mode is False`(比原来只断言"不拒绝"更强);新增 `TestEnforcementModeSoftDefault`(7例):缺 key 默认 soft、去重/篡改命中 soft 下不拒但 `would_reject_in_hard_mode` 仍为真(不许假装没命中)、soft 产物不带 `status` 键、干净图 `flagged=False` 且 `reason==""`、非法值(`"HARD_typo"`)降级 soft、`" Hard "` 大小写/空格容错、**交付 `configs/thyroid_config.yaml` 必须解析为 soft**(防止改回 hard 后忘了,用 `__file__` 定位仓库根不依赖 cwd)、**`gatekeeper` 段写歪成标量/列表/None 时降级 soft 而非抛异常**(Gate2 P2-1:抛异常会被入口兜成 ERROR = 整例零诊断,恰是软化要避免的后果);**2026-07-29 篡改软flag接入 audit**——新增 `TestSoftFlagsLandInAudit`(6例):`last_soft_flags` 落进 audit 的 `soft_flags` 键并计入 `flagged`(此前生产零读取点,仅 copy_move/splice/BAG 命中的图 flagged=False、result.json 完全沉默,copy_move 真实图误报 27.3% 信号全丢)、**红线软flag恒不进 `would_reject_in_hard_mode`**(soft 与 hard 模式各锁一例)、无命中空列表、旧 detector 无该属性 getattr 兜 `[]` 不 AttributeError、畸形元素归一化不崩、**畸形配置 cosine_threshold 落交付值 0.995 而非库默认 0.95**(防静默放松去重判据踩宁可漏检不可误杀红线) | 48 | — |
| `test_authenticity_trainer.py` | 真实性分类器训练器 smoke — 合成伪超声喂训练器跑通,产出 dict(state_dict+feat_mean+feat_std);推理端 AuthenticityModule 加载并应用归一化、predict 返回 softmax(守卫训练/推理归一化口径一致);**权重缺失硬报错回归**(2026-07-10:此前静默退化为随机初始化分类,一票否决门形同虚设;**该回归针对 `AuthenticityModule.load_weights` 内部硬报错,仍成立**;其调用方 `build_auth_module` 自 2026-08-10 R3 起降级 skipped,见 `test_gatekeeper.py::TestAuthWeightFailureDegrades`);**EXPECTED_CKPT_SHA256 完整性校验**(2026-07-30 交付封印启用,`TestExpectedCkptSha256` 3例:匹配通过/不匹配硬报错/None 跳过,配套 `weights_only=False` 防 pickle 任意代码执行) | 6 | — |
| `test_report_generator.py` | `report_generator.py`(2026-07-10 新增,此前零测试覆盖) — TI-RADS评分表边界(ACR白皮书对照、TR1-5分级区间)、**越界idx硬报错回归**(comp/echo/shape/margin/foci越界此前静默按0分处理,实测margin_idx=5可致TR3→TR2静默跨越穿刺决策边界)、**malignancy_prob=None/nan/inf硬报错回归**(此前nan>=0.5恒False导致无报错输出"良性倾向",癌症筛查假阴性风险) | 13 | — |
| `test_multisource.py` | 多源数据集骨架 — 源级 label_mask(默认全有效向后兼容、源 mask 标注缺失任务、meta label_mask 覆盖源 mask);`MultiSourceThyroidDataset` 拼接长度、源级 mask 生效、采样权重反映源 ratio、空源不崩 | 7 | — |
| `test_multisource_seg_smoke.py` | **TN3K+ThyroidXL 分割混训 loader 契约**(2026-08-11 新增,A0 诊断后)— 无 mask 样本 valid_mask=0 且分割零梯度(`compute_segmentation_loss` 直测)、有 mask valid_mask=1、TN3K 源级屏蔽 5概念+malignancy、ThyroidXL 逐样本 meta 6键屏蔽概念、get_sample_weights 按 ratio 均摊 | 10 | — |
| `test_format_thyroidxl.py` | **`scripts/data/format_thyroidxl.py` 脚本逻辑单测**(2026-08-11 新增,补 smoke 够不到的脚本层)— **【阻塞1】`_build_meta` 的 label_mask.malignancy 是有效标志非标签值**(良性 has_conclusion=True→malignancy=1,label_malignancy=0 独立;7键锁定)、【高2】全黑 mask 判 has_mask=False 不写 mask.png、【footgun】dest 指向活数据根 raise ValueError、【撞名】image_id 撞名 raise、【_self_check】合法通过/缺键 raise。用 monkeypatch 假 adapter 纯合成不读盘 | 9 | — |
| `test_stage_schedule.py` | **三阶段训练调度的关键数值锁定**(2026-08-11 新增,Stage2 崩塌事故后补)—— 此前 `tests/` 下**无任何**测试断言 `configure_stage_parameters` 的 lr 返回值或 stage 边界,`gate7_optimizer_lr.py` 只验 epoch=1/41/101 三个单点"lr 算得出且 optimizer 对齐"、**不验跨阶段跳变幅度**,正是 100 倍 lr 跳变从门下穿过的原因(与 #119 同类:只验有变化不验方向/幅度)。锁定内容:**stage 边界 lr 跳变**(lr40=2.400e-5 / lr41=2.39852e-3 / **ratio≈99.94**,🔴 断言的是"当前实测值"非"理想值"——钉住这个已知缺陷是为了逼将来改调度的人先看到它和它的出处) + Stage2→3 反向骤降(2.4e-4→5e-6) + Stage3 恒定 lr + 全 120 epoch 与独立参考式逐点一致(刻意不 import 实现,任一方漂移即红) + warmup 段严格上升 + 各 stage 内余弦严格单调下降 + **γ_prior 退火表**(epoch 20→0 / 30→0.5 / **40→1.0 恰在 Stage2 起点前一个 epoch**)+ γ 全程 clamp 且单调不减 + `train_batch` 必须调用 `compute_gamma_prior`(AST 级,防内联公式复活致两处漂移) + **各 stage 冻结集合**(Stage1 只冻 CBM 直连头、主干可训;Stage2 零冻结;Stage3 按 `STAGE3_FROZEN_PREFIXES` 且反向验"该冻的真冻上",防前缀拼错静默失效)+ **optimizer 覆盖**(所有 requires_grad 参数入组=防漏进静默不更新;vclub 恒不入主组=防与独立 vclub_optimizer 双重更新)+ **AUCM 对偶变量绑定**(`QA_VP.md §1.2` 领域红线,Gate2 第二步补:上一条只遍历 `model.named_parameters()` 看不见 `criterion` 上的 `a`/`b`/`alpha`;Stage1 冻结且不入组、Stage2/3 解冻且必须入 weight_decay=0 的组——漏绑不崩溃只静默退化,minimax 内层失效而日志正常)+ param_group lr 与 stage_lr 一致(gate7 同款,补进 CI 侧)+ **混训采样器期望抽中次数**(解析式与蒙特卡洛双向对照,断言不变量而非硬编码;生产配比下 TN3K≈2.88/XL≈0.48/比值 6.0/XL 去重可见≈3760 张——修正 `train.py` 原注释漏除 Σw=1.6 的 "1.2/0.8" 错算)。纯 CPU/无网络/无权重,module 级 fixture 建一次模型 | 22 | — |
| `test_smoke_entrypoints.py` | CLI 入口端到端冒烟 — 把 `predict.py`/`video_predict.py` 当**子进程实跑**(随机权重,build_model 缺 checkpoint 降级随机+warning 不报错):predict 噪声图路径(进程正常退出、落 result.json、status∈{REJECTED,PASSED}——**2026-08-10 R3 起** authenticity 权重缺失在 build_auth_module 降级 skipped、**绝不走整例 ERROR**,故缺权重环境下 status 必须是 PASSED(soft)或 REJECTED(dedup/tamper 硬判据,平台浮点敏感);正向断言「降级确实生效」,不耦合任何错误文案字面量。此前的旧预期"无 authenticity 权重→ERROR 是预期行为"已作废)+ video `--skip_gatekeeper` 路径(推理/后处理/报告/JSON/Grad-CAM 四产物齐全、JSON 中文可还原)+ **权重不匹配 ERROR 兜底回归**(predict.py + video_predict.py 各一例;2026-07-10：build_model 硬报错后若无兜底会致容器裸崩溃零输出，测试构造 mismatched checkpoint 断言 exit 0 + `result.json.status=="ERROR"`；此前只测了video_predict.py一侧,同轮补齐predict.py侧因二者gatekeeper/build_model调用顺序不同,分支可达性此前从未验证过)+ **输入加载失败 ERROR 兜底回归**(2026-07-11 新增,4例:predict.py 图片不存在/空文件、video_predict.py 视频不存在/空文件——此前 `load_input_image`/`load_video_frames` 返回 `None` 时两脚本在 try 块外直接 `return`,只 print 不落 result.json,exit code 仍 0 但 output_dir 空目录,比 REJECTED 还差且绕开 except Exception 安全网,现已在 return 前补 `save_rejection_json`;video侧空文件用例2026-07-11 Gate2 review 非阻塞建议补齐,此前只测missing未测empty,覆盖度不对称)+ **`--config` 加载失败 ERROR 兜底回归**(2026-07-11 新增,`TestConfigLoadFailureErrorFallback` 4例:predict.py config缺失/格式错误、video_predict.py config缺失/格式错误——同一类 try 外裸操作问题,这次补配置加载这一处,见 `model_runner.load_config`;video侧格式错误用例2026-07-11 Gate2 review 非阻塞建议补齐,此前只测missing未测malformed,覆盖度不对称)+ **DICOM 输入路径**(2026-07-11 P1 新增,`TestDicomInputPaths` 4例:单帧 DICOM + 多帧 cine + **压缩 DICOM(JPEG baseline,单帧+多帧参数化)**,用 `scripts.verify.gen_synthetic_dicom.build_dicom` 生成真正可被 `pydicom.dcmread()` 解析的合法 DICOM(区别于 `test_entrypoint_dispatch.py::_write_dicom` 那种只有魔数字节、只供 bash 分发路由测试用的假文件),经 `video_predict.py --skip_gatekeeper` 全链路验证 `status=="PASSED"`;pydicom/pylibjpeg 未安装时各自 skipif。压缩 DICOM 一例是 Opus 战略审查发现的致命缺口补测——真实超声 DICOM 绝大多数是 JPEG/JPEG2000/RLE 压缩,此前 fixture 只测未压缩格式,若压缩解码插件缺失会导致该动态图样本 100% 丢失预测,`requirements.txt` 已补 `pylibjpeg`/`pylibjpeg-libjpeg`/`pylibjpeg-openjpeg`,版本锁 `numpy<2` 兼容区间;压缩+多帧组合 2026-07-11 Gate2 review 非阻塞建议补齐,此前只测压缩+单帧)+ **JPG 静态图路径**(2026-07-11 P1 新增,`TestJpegInputPath` 1例:官方确认静态图交付格式是 JPG,此前所有测试全用 PNG 构造从未验证过 JPG 路径)+ **非空损坏文件**(2026-07-11 P1 新增,`TestCorruptedNonEmptyFile` 1例:与已有空文件用例区分,写入 2048 字节随机 garbage 内容(非 0 字节)验证 `cv2.imread` 失败路径同样被正确兜住,`status=="ERROR"`)。**只兜「管线连通+产物落盘+JSON 可解析」,不把 gatekeeper 在随机噪声上的判定当硬断言**——该判定依赖 GLCM/频域统计浮点细节,随平台(numpy 1.x 容器 vs 2.x 本地+不同 BLAS)翻转,本地拒、容器可能放行;TODO 真实数据复核其平台稳定性。补 99 组件级单测对入口「接线」的集成盲区;codec 缺失则 skip | 18 | — |
| `test_probability_output.py` | **概率分布输出契约**(2026-07-29 新增)——官方硬约束"所有分类/分级/分型任务必须输出功率分布或置信度,**否则计0分**"(赛前动员会 2026-07-28)。此前 `decode_predictions` 对 comp/echo/shape/margin 只取 argmax,softmax 分布在 decode 层就丢失、从不落 `result.json`(`cbm_head.py` 内部其实已算出,属**导出遗漏**非能力缺失)。锁定:四个 softmax 头分布维度与 `report_generator` 映射表一致且和为 1、`*_idx` 恒等于分布 argmax(8 个随机 seed 循环验证,防索引与概率各说各话)、`mal_probs` 二类和为 1 且 `mal_prob == mal_probs[1]`(同源,禁两条计算路径)、`foci_prob` 是**多标签 sigmoid 不求和**(易被误当分布)、旧 `*_idx`/`foci_prob`/`pred_mask` 等字段未被删(下游 `build_tirads_breakdown` 依赖)、温度校准对概率与 argmax 走**同一条** `_cal` 路径(T=4 时最大概率被压低但 argmax 不变且未塌成均匀)、导出 `json.dumps` 可序列化(numpy float32 不 `.tolist()` 会直接抛)、类别数同时对齐 `report_generator` 映射表**与** `configs/thyroid_config.yaml` 的 `num_classes_*`、**AST 断言两个入口都调用 `build_probability_export`**(禁止任一侧手写概率字段导致静态图/动态图 schema 分叉)、**fp16 输入对拍**(Gate2 P2-6:半精度 logits 必须先 `.float()` 再校准/softmax,纯 fp32 测试删掉 `.float()` 也全绿抓不到) | 21 | — |
| `test_gatekeeper_soft_enforcement.py` | **gatekeeper 软化的入口层回归**(2026-07-29 新增)——`enforcement_mode` 默认 soft 后,审计命中必须仍输出完整诊断。在 `predict.py`/`video_predict.py` 的 `main()` 层实跑(monkeypatch `run_gatekeeper_audit` 注入 flagged 审计,权重缺失走随机初始化降级):静态图命中后 `status=="PASSED"` 且 7 个诊断字段齐全、`gatekeeper.is_duplicate==1` 事实照记(不许因不拒绝就抹掉审计)、hard 模式仍写 REJECTED 且不产出诊断;**动态图是重点**——3 帧中第 1 帧命中时断言 `run_gatekeeper_audit` 被调满 3 次(旧逻辑在此 `return`,后续帧根本不跑)、整段仍出完整诊断与 `per_frame_malignancy`、`flagged_frame_indices==[1]`;hard 模式下命中即停(只调 2 次)且带 `rejected_frame_index`;`--skip_gatekeeper` 时 `gatekeeper=={"skipped": true}`;**旧键 `foci_probabilities` 与新 `concept_probabilities.foci` 同值、`malignancy_probability` 与分布恶性分量同值**(Gate2 P2-6:同源但若只改一侧会静默分叉);静态图侧也须带 `skipped` 键(两链路共用解析代码);`keyframe_flagged` 与集合运算结果一致;**2026-07-29 坏帧隔离修复回归**(新增 `TestBadFrameIsolation` 5例,锁一轮自引入回归——Phase 1 的 `_assert_finite` NaN/Inf 硬校验在逐帧循环无 try/except 时,单帧异常冒泡成整段 ERROR,性质与「一帧命中拖垮整段」相同):单帧 ValueError 不杀整段、`n_frames_sampled`/`n_frames_diagnosed`/`n_frames_failed`/`failed_frame_indices` 如实分别记录不静默少算;**关键帧错位指纹回归**(核心——第 0 帧坏 + 第 2 帧最恶性时,stub Grad-CAM 截获 mask/热图实际入参的帧指纹,断言出自真正的帧 2 而非旧 `key_tensors` 平行 dict 下会错位取到的帧 1,已实测复现错位);全帧失败走显式守卫落 ERROR + 明细而非 AssertionError/IndexError(assert 在 `python -O` 下会被剥离);无失败时新字段退化为原行为;**RuntimeError(CUDA OOM/权重维度不符)仍冒泡成 ERROR 不被逐帧 except ValueError 吞掉**(否则真故障被伪装成"少几帧"的 PASSED) | 12 | — |
| `test_calibration.py` | Post-hoc 温度缩放骨架 — **降级正确性**(缺参数/损坏文件/全 T=1.0 → `decode_predictions` 输出与无校准逐位相等)、**单调性**(T>1 压低 max 概率向 0.5 收缩但不改 argmax)、**ECE 数值**(完美校准≈0、过度自信全错=1.0、一维正类概率自动补两列)、**拟合**(`fit_temperature` 返正温度、`from_file` roundtrip);非法温度(≤0)拒绝降级 T=1.0 | 11 | — |
| `test_pathology_dataset.py` | PathologyDataset — 标签广播(患者级→图像级)、患者分组无泄漏、batch列顺序容错(path,patient_name反列)、分层划分、valid键契约(`valid_malignancy=1`/`valid_mask=0`/其余`valid_*=0`)、输出形状dtype | 20 | — |
| `test_pathology_mil_dataset.py` | PathologyMILDataset + MIL模型 — bag聚合(同patient同bag)、label一致性、顺序确定性、形状`[N,1,H,W]`、max_frames截断、患者级class_counts、train/val无泄漏、GatedAttentionPooling输出`[1,1408]`+weights和≈1、ThyroidMILNet前向`[N,1,512,512]→[1,2]`、冻结后backbone无梯度 | 13 | — |
| `test_entrypoint_dispatch.py` | **扩展名分发**(mp4/avi/mov/mkv/dcm/dicom 含大写→video_predict、png/jpg/… →predict、flag 透传,用假 python shim 拦截不真跑推理)、**PRODUCTION=1 防篡改阻断**(predict 哈希不符 / src 树哈希不符均在分发到 python 之前 `exit 1`,`SHIM_ARGS not in stdout` 实证「阻断先于分发」;开发模式正确绕过)、**src 树指纹 parity**(source `_src_tree_hash.sh` 真函数 vs `seal_container.py` Python 实现字节级一致,反向验证改坏函数必 FAIL 防 drift 假绿)、**内容嗅探兜底**(`TestContentSniffFallback`:扩展名↔内容不一致双向重路由、一致时不误报、DICOM 靠内容识别、完全无法识别硬失败 exit 1、文件不存在退化为纯扩展名路由)、**2026-07-12 子进程崩溃兜底(P2红线修复C级)**(`TestSubprocessCrashFallback` 4例:去掉 `exec` 后子进程崩溃/非零退出且未产出 result.json → 父shell补写ERROR JSON+透传退出码;真实产出不被兜底覆盖;out_dir 预存陈旧 result.json 时不被误判为本次成功、`.stale` 标记清理干净;去 exec 后分发路由本身不受影响的回归锁定。既有分发/PRODUCTION/内容嗅探测试改为显式传 tmp_path 隔离 --output_dir,避免污染仓库真实 data/output/)、**2026-07-12 Gate2 复审补丁**(`TestSignalForwarding::test_signal_forwarded_and_wait_captures_real_exit_code`:用真实 `SIGTERM` 驱动一个"收到信号后需 0.3s 优雅关闭"的 shim,锁定 `wait` 被 trap 打断提前返回合成退出码这一实测复现的竞态 bug 已修复——entrypoint 最终退出码必须是子进程的真实值而非 128+信号,`result.json` 必须是子进程自己写出的那份而非误判的兜底 ERROR;`os.name=="nt"` 时 skip,POSIX 信号语义以 Linux CI 为准) | 30 | — |
| `test_bn_freeze.py` | `set_frozen_modules_eval`(2026-07-10 新增，回归测试) — 冻结子模块 BN running_var/mean 零漂移、不冻结时确实漂移(反向验证测试有效性)、可训练子模块不被误冻、LoRA 场景下同级独立 BN 子模块仍正确冻结 | 4 | — |
| `test_model_runner.py` | `build_model`(2026-07-10 新增，回归测试) — 裸 state_dict 正常加载、`{"state_dict":...}` 包装格式正确解包、unexpected/missing keys 硬报错(拒绝静默部分加载)、checkpoint 缺失时仍降级随机初始化(向后兼容)；`run_forward`(2026-07-11 新增，两轮修复，`TestRunForwardCpuFp16Fallback`+`TestRunForwardGpuPrecisionRegression`) — CPU+fp16 autocast "假降级"修复回归、**CPU+fp32 不崩溃**(第一轮修复误把此分支 dtype 从 bfloat16 改成 float16 引入新回归，CI test job 实测抓到，第二轮改为 CPU 上 dtype 恒 bfloat16)、bf16-on-cpu 未被连带破坏、GPU fp32/fp16/bf16 三值回归(skipif 无 CUDA)；`load_config`(2026-07-11 新增，`TestLoadConfig`) — 合法配置正常解析、文件不存在抛 `FileNotFoundError`、YAML 格式错误抛异常、缺 `environment.device` 键抛 `KeyError`(供 predict.py/video_predict.py 的 try/except Exception 统一兜底) | 15 | — |
| `test_gate8_adaptor_isolation.py` | `gate8_adaptor_isolation`(2026-07-11 新增) — TaskAlignmentAdaptor cls→seg 梯度隔离审计的正面/负面回归；正面:真实代码(`.detach()` 正确)+ 模拟前序 Gate 遗留的 `requires_grad=False` 污染状态下仍判定通过、`seg_backbone` 梯度精确 `0.0`、`gamma`/`requires_grad` 复位无副作用(2026-07-11 Gate2 review 非阻塞建议补齐:此前 finally 只复原 gamma 没复原 requires_grad,依赖"Gate8 是最后一个 Gate"这个隐性顺序假设,现改为逐参数显式记录+复原);负面:monkeypatch `torch.Tensor.detach` 为恒等函数模拟误删,断言 Gate 检测到泄漏并返回 False(踩坑记录:`TaskAlignmentAdaptor.gamma` 零初始化会让正确/错误 detach 都测出梯度 0,Gate 内部强制拨到非零才能让负面测试真正生效,详见该 Gate 源码注释) | 2 | — |
| `test_prediction_json.py` | prediction.json schema 构造与校验(2026-08-01 新增;**2026-08-10 A3 起 v21_output 经它组装/校验,并入 V2.1 交付链路**;**2026-08-17 #116a 加对冲键 `TI_RADS_Probs`/`ETE_Probs`**:可选参数、`None` 时不写该键、长度错抛 `PredictionSchemaError`) | 72 | Phase 5 |
| `test_v21_output.py` | **V2.1 单病例输出写盘入口**(2026-08-10 A3 新增)——`write_case_v21` 静/动两链路:静态把 case_pipeline 结果落成 `{case_dir}/{subdir_uid}/nodule_NNN.nii.gz` + `prediction.json`(StudyUID=PatientId、MaskFile 相对路径正斜杠、原子写、阴性空数组合法、缺 orig_hw 报 V21 错误);动态逐帧 resize 回原图 + scatter 回填全帧 3D mask + 3D 连通域跨帧关联。**🔴 2026-08-17 #116b 接线后语义变更**:原 `TestPlaceholderDims` 只断言"落在合法范围内"——接线后**仍会绿**但守的是已消失的语义(占位结构合规),对 off-by-one **零区分力**,已重写为 `TestOfficialHeadWiring`(逐项断言精确期望码,参数化**全部避开 idx0** —— 用 idx0 测等于没测:漏写 `+1` 时 idx0 恰好也"对")+ `TestHedgeKeys`(**从 `write_case_v21` 入口测到落盘 JSON**,只测单函数会漏掉"调用处没透传")+ `TestDynamicLesionLevelMean`(构造成"三帧均值 argmax ≠ 关键帧 argmax",否则沿用关键帧也能过) | 31 | Phase 5 |
| `test_service_runner.py` | **runner 批量编排**(2026-08-10 A2 新增)——mock 掉 heavy 依赖只留编排逻辑:逐病例调 write_case_v21 且 case_dir={output}/{patient_id}、静/动 kind 分派、processing_ms 非负 int、gatekeeper 每病例照跑、**单病例失败隔离**(一例 diagnose 崩溃不影响其余 + 落结构合规 ERROR 占位)、loader 失败隔离、**模型只 build 一次**(跨病例 + 跨多次 build_service_runner 缓存)、空/无效/全非影像目录不崩 | 12 | Phase 5 |
| `test_head_interfaces.py` | **三头接入接口契约**(2026-08-12 Phase2 阶段2.0 新增)——守四条「删掉就静默出错」的契约:① `stage4_spatial` 通道数是 `CLS_POOLED_DIM`(352) 而非 `CLS_STAGE4_CHANNELS`(208)，且自带 352≠208 反向锚点防断言失去区分力;② 命名即契约(`stage4_spatial` 有梯度、两个 `*_detached` 无梯度);③ 🔴 从 `stage4_spatial` 反传 `seg_backbone` 梯度**精确 0**(用 `!=0.0` 而非容差,对 NaN 敏感,同 Gate 8 论证)——保住 Gate 8 前提、防三头重开 Stage2 崩塌的物理先验回流;④ fp16 求和溢出反证(自带「反证失效」保护:先断言朴素 `.sum()==inf`,否则说明该测试已对 fp32 累加器失去区分力) | 14 | Phase 2 |
| `test_location_head.py` | **位置头契约**(2026-08-12 Phase2 阶段2.1 新增)——守「错了不崩、只静默学错」那几条:① 因子分解**拼接顺序**必须对齐 `LOCATION_LABELS`(用非对称确定性输入逐格核对——顺序错了概率和仍为 1、JSON 仍合法,只是左叶预测填进右叶槽);② 7 维**严格**和为 1(数学恒等式非再归一化);③ 峡部 level 标签是 `-1`,须 clamp+validity 双保险(附反证:直接喂 `cross_entropy` 确实越界);④ 标签越界**正负不对称**(`idx=7` IndexError 而 `idx=-1` 走负索引静默取末元素)双向拦截;⑤ `[B,1]` validity 隐式广播成 `[B,B]` 静默放大 loss(附反证坐实该坑真实存在);⑥ 🔴 **CoordConv 必须先过非线性再池化**——三段反证(普通卷积不可分 → **裸 CoordConv 仍不可分** → 加 ReLU 才可分),卷积线性可拆使坐标贡献退化为常数偏置被 GAP 平均掉;⑦ 水平翻转联动翻标签 | 25 | Phase 2 |
| `test_extension_head.py` | **ETE 侵犯头契约**(2026-08-12 Phase2 阶段2.2 新增)——守「稀有类静默学错 + CB 保护失效」那几条:① 4 类 ETE 概率**严格**和为 1、argmax 与分布自洽;② 类别顺序对齐 `ET4 码`(无/甲状腺包膜/毗邻软组织/血管,计数 3906/267/67/5);③ `class_balanced_weights` 乘性加权数值(β=0.99 默认 → 血管:无=20.4:1,β=0.9→2.4:1 / 0.999→196.4:1 / 0.9999→646.9:1 全表锁定);④ CB-Focal γ 聚焦 + Logit-Adjustment 先验补偿存在且不翻转方向;⑤ `boundary_band_from_mask` 带宽 = `kernel_size−1`(k=3→2px / 5→4px / 7→6px 实测,实现取 k=5 → 4px);⑥ 头/损失同名不同职责(`heads/extension.py` vs `losses/components/extension.py`)不致误导;⑦ 参数量 0.063M。**2026-08-17 backlog #128 +10 = LA-SoftQWK-OD**(`quadratic_weight_matrix` 与 sklearn quadratic 权重定义对齐 / SoftQWK 常数预测恒 1.0 与完美预测 <0.5 / QWK 分支梯度有限 / **validity 先过滤不掺 O-E 统计**(masked-mean 范式在此不适用) / 全无效返 0 无 NaN / 有序距离随距离单调 / 🔴 **`qwk_weight>0` 必须 `la_tau>0`** 的双向断言(拦只开 QWK、放行只开 LA 的既有配置) / LA 边界方向的 DR 数值实例。**后两条已做变异测试**:LA 符号 `+`→`-` 后方向断言变红(3.5098 vs 0.0794),证明有区分力、非自证式)。**2026-08-30 +2 = 变异审计补测**(ordinal 距离 `/(C-1)²` 归一化的逐项手算数值断言 + `total = qwkW·qwk + distW·dist` 的数值分解断言——系统变异审计证明此前这两处无区分力:丢 dist 项 / 归一化指数 `**2`→`**1` 两个变异在旧测试下全绿,补测后均被杀死) | 51 | Phase 2 |
| `test_clinical_t_head.py` | **T 临床显著性头契约**(2026-08-12 Phase2 阶段2.3 新增)——守「统计门控 + 尺度 log 通道 + 梯度隔离」那几条:① `t_conf` 是 **P(T=1) 不是 max prob**(填成 max prob 强阴性样本置信度≈1.0 即失败,反构造强 T=0/T=1 常数 logits 双向验证);② `T_conf` 一维 `[B]`(`[B,1]` 隐式广播 QA_VP §2.2);③ 🔴 ETE 旁路 `detach`+softmax——ETE logits 平移不变、**不反灌 ETE/侵犯头梯度**(ETA 是官方独立计分维度,含反证:改 ETE 分布必须改输出);④ `nodule_scale` 同样 detach 不回流分割主干;⑤ scale=0(空 mask 阴性)→log0=-inf 经 `clamp(min=1e-4)` 兜住不 NaN;⑥ 🔴 scale 的 **log 通道白盒 hook 验证**(等倍数 scale 对在 log 通道上间距恒 ln2≈0.6931,线性通道 40 倍差作反证——黑盒"输出不同"零区分力已实证,把 log 通道换第二份线性 s 照样全绿);⑦ 禁 BN1d(train+batch=1 实测 ValueError,结构检查+实跑双断言);⑧ 类权重:临床 T 千级计数下**抄侵犯头的 β=0.99 = 权重比恒 1.0 静默失效**(钉死实测防"统一一下超参",β=0.9999→1.28 仍弱于逆频率 1.3162,反证同函数在 ETE 长尾 3906/267/67/5 确实 20.4:1);⑨ 损失:validity 一维拒广播、标签越界正负双向拒、全无效零损失零梯度、分母只数有效样本、类权重按**标签**索引非 argmax、无权重时退化为裸 CE、label_smoothing 越界拒 + **LS 把 P(T=1) 上界钳死在 1−ls/2**(ls=0.05→0.975/0.1→0.95,训到收敛实测钉住)、±1000 logits 下损失梯度有限、bf16 autocast 有限值;⑩ 🔴 端到端参数级梯度隔离(真 `MultiTaskThyroidModel`+真 `ExtensionHead`):seg_backbone/cbm_head/**ExtensionHead 参数** 均零梯度,cls_backbone **必须**有梯度作反证 | 27 | Phase 2 |
| `test_case_discovery.py` | **病例发现**(2026-08-10 A2 新增)——`discover_cases` 递归遍历 + 静/动分派(.dcm→static,.nii.gz/.nii→dynamic)、静态读 DICOM 头取 PatientID/SeriesUID、动态按组委会口径取直接父目录名(PatientId=AccessionNumber)、路径推断兜底、空/不存在目录返回 []、**跨文件系统确定性顺序契约**(monkeypatch 反转 `os.scandir` 返回序,断言发现顺序仍为字典序 DFS——2026-08-12 CI run 31512732898 因 `os.walk` 未排子目录、Linux 走 hash 序而 flaky,`dirs.sort()` 修复后由本用例钉死);依赖 pydicom/nibabel 缺则 skip | 40 | Phase 5 |
| `test_mixup_label_dominance.py` | **四头监督闸门 lam 阈值分派 + 软 lam 反向保护**(2026-08-15 新增)——详见下方专节 | 12 | Phase 2 |
| `test_dynamic_frame_cache.py` | **预抽取 PNG 缓存契约**(2026-08-15 新增)——详见下方专节 | 9 | Phase 1 |
| `test_detection_match.py` | **检测诊断仪表盘·纯逻辑层**(2026-09-05 新增,检测 AP 3.05 主战场第一步)——官方匹配口径逐条:iou_2d 已知值/双空(「切片全0记0」)/形状不齐拒;iou_dynamic 只遍历标注帧+未覆盖帧记0(官方两条限定)+空标注帧;匈牙利一对一最优+**阈值丢弃**(解里 IoU<0.5 的对双方回未匹配)+空矩阵不崩;**手写 step-wise AP 的 n_gt 分母**(sklearn 虚高口径对拍钉死:单TP+99漏检 sklearn=1.0 vs 真值0.01)+同分合并用**组末下标**(首版用组首下标,同分 FP 被挤出分母 AP 虚高,A路审查实测[1,0]/[0.8,0.8]给1.0真值0.5)+手算PR曲线+TP≤n_gt护栏(A路:无护栏时 n_gt=1 两 TP 静默返回2.0);dedup「各GT最优并集」语义+多GT冲突不误删他GT最优(A路P1:逐列淘汰法`[[0.9,0.2],[0.0,0.8]]`误删gt0最优,级联ap_dedup语义反转);ap_breakdown四件套(oracle==匹配率/碎裂场景手算)+桶边界(0.1/0.35/0.5名实一致)+NaN落invalid桶(A路:NaN曾静默落">=0.5"命中桶)+anomaly行跳过+主/次病灶匹配率+临界带占比 | 22 | — |
| `test_diagnose_detection.py` | **检测诊断仪表盘·端到端**(2026-09-05 新增)——合成 P1静态单GT/P2静态一图两GT(钉死只推理一次+一起进匈牙利)/P3动态5帧mask帧1/3非零;fake与真实`diagnose_*_case`返回**逐键一致**(B路P0教训:fake多给`n_total_frames`自证键,把「动态链漏注入」藏在测试外);fake预测图用**MODEL_HW=32×32**≠原图64×64(B路变异注入实证:同尺寸resize恒等映射,插值/尺寸漂移全被吞);**等价回归双桩**(动:write_case_v21 vs run_dynamic_inference_from_result,n_total=100抽帧{0,70,99}使gap±1落在区分窗口——n_sampled=32/gap去+1/静态conf常量/插值错误四个变异注入全RED;静:write_case_v21("static") vs run_static_inference_from_result,conf∈(0.1,0.9)开区间排除常量与二值化漂移);GT帧数≠n_total记anomaly跳过;mask_path=None过滤;动态conf「帧覆盖率语义」钉死(scatter二值化后=1.0,与生产逐位一致) | 5 | — |

---

## `test_official_loader.py` — 官方数据加载器契约(14 用例,2026-08-10 + 2026-08-12 阶段2.4 扩 + 2026-08-13 Phase E hflip RNG 防回归 + Gate2 mask 命名 glob 防回归;缓存依赖**另见** `test_dynamic_frame_cache.py`)

合成 `.dcm`/`.nii.gz`/xlsx fixture(运行时生成,不入库二进制),依赖 pydicom/nibabel/pandas/openpyxl(缺则 skip)。

🔴 `test_build_datasets_does_not_freeze_hflip_rng`(Phase E 新增):断言 `build_official_datasets`
返回的 dataset **`_rng is None`** —— 该 `seed` 只能喂 `patient_grouped_split`,传给 Dataset 会让
持久 `random.Random(seed)` 实例被 pickle 复制进每个 worker(状态相同)、且每 epoch 重建 worker 使
序列重置,**实测三个 epoch 翻转序列逐位相同**(增强白做,不报错、loss 照常降)。生产必须走全局
`random`(由 `worker_init_fn` 按 `torch.initial_seed()` 播种,torch 每 epoch 重抽 base_seed)。

## `test_official_schedule.py` — 官方三头 A/B/C 调度 + JSONL 日志(**26 用例**,2026-08-17 `--collect-only` 实测;原记 16 是 Phase E 初版数、已过时)

`schedule_official.py` 的数值/契约锁定,与 `test_stage_schedule.py`(legacy 三阶段)**语义互补不重叠**:
后者钉的是"已知缺陷 100 倍跳变确实存在",本文件钉的是"新调度**不许**出现这种跳变"。
锁定内容:**相邻 epoch lr 比值 ≤5**(逐点扫 1..35,`MAX_LR_JUMP_RATIO`)+ A/B/C 边界与端点值
(A 段 e1=lr_base×0.1 线性升至 e10=lr_base、B 段 e11 回到 lr_base 余弦退至 e30=×0.1、C 段固定 ×0.05)
+ A 段严格上升 / B 段严格下降 + **各 stage 冻结集合**(A/C 训四头 + **分割前向链** + vclub;B 额外解冻且**仅**额外解冻
`cls_backbone.blocks[5]/[6]`)+ **optimizer 全覆盖**(vclub 除外,双向:漏进=永不更新、
误进=双重更新)+ AUCM 对偶随 `freeze_legacy` 的绑定/冻结双态 + **LLRD 方向**(blocks[6]=头 lr×0.1 >
blocks[5]=×0.1×0.75,越靠输出 lr 越大)+ **BN 状态**(冻结的 cls 主干 BN 置 eval、四头与分割保持 train)
+ 🔴 **分割解冻契约**(2026-08-16 新增,A/B/C 三 stage 各自:可训 ∧ 进 optimizer;解码器 lr=head_lr、
主干 lr=head_lr×scale 且只落一档;无参数进两组)
+ JSONL 必填字段/ISO8601 带毫秒 Z/逐行可解析/非法 phase·mode 拒绝。纯 CPU、`pretrained=False`。

| Test | 验证内容 |
|:---|:---|
| `test_parse_xlsx_mixed_cells` | 混写单元格按 prediction_json 权威常量解析(2左中叶→idx1、ETE 2甲状腺包膜→码2-1)、空列→valid=0 |
| `test_build_samples_dispatch_and_skip` | 静动分派、清晰帧行跳过、动态按非零 mask 帧展开 |
| `test_output_contract_static` / `_dynamic` | 输出契约 img[1,S,S]∈[0,1]/mask 二值/labels 含 valid_*;动态取正确非零帧 |
| `test_patient_grouped_split_no_leak` | train/val 患者无交集(rules.md §5 防泄漏) |
| `test_ete_parse_edge_cases` | ETE 纯中文回退、越界数字(如 `0无`)不产生负类索引 |
| `test_location_pure_chinese_and_negative` | 位置纯中文回退、阴性/未知→valid=0(绝不默认 0=左上叶) |
| `test_dynamic_no_mask_skipped` | 动态病灶无 mask→无法定位帧→整例跳过 |
| `test_labels_carry_is_main_and_lesion_id` | **labels 携带 `is_main`(M→1,病灶N→0)+ `lesion_id`(病灶级聚合键)**,供三头只训主病灶/指标按病灶聚合 |
| `test_hflip_syncs_location_label` | **hflip 联动翻位置标签**(左中叶→右中叶 1→4,经 `flip_location_index`),防学到反的左右 |
| `test_hflip_skips_invalid_location` | 位置标签 invalid 时 hflip **不翻**(不编造监督) |
| `test_val_dataset_has_no_hflip` | val 数据集恒无 hflip(翻转会改位置标签,val 必须原样) |

## `test_train_step_head_targets.py` — 三头 targets 口径(12 用例,2026-08-12 阶段 2.4;闸门归属 2026-08-15 后见 `test_mixup_label_dominance.py`)

`train_step.build_official_head_targets` 的三条"错了不报错、只让指标系统性偏低"红线:① **三头只训主病灶**(`is_main==0` → validity=0,评审规则点评口径);② **mixup 后标签归属**(DR-CutMix lam==0 继承 donor 标签,含 is_main/validity 一起 gather);③ **混合归属按 lam 阈值分派**(2026-08-15 修复后:`lam<=0.35` 继承 donor、`lam>=0.65` 保留 recipient,中间段屏蔽——对 MixUp 软 lam 也恢复监督,历史以"软 MixUp 整批屏蔽"为旧行为,归属细节以 `test_mixup_label_dominance.py` 为准)。另含:is_main 缺席(legacy)全当主病灶零回归、is_main×原 validity 相乘、cutmix mask_weight=0 屏蔽、legacy 无三头键返回空 dict、部分键只出 present 任务、缺 valid_* 默认全有效、validity 必须 [B] 一维 float([B,1] 隐式广播成 [B,B])。

🔴 **`import sklearn` 必须在 torch / src.training 之前**(文件顶部先 `import sklearn.metrics` + docstring 记录):Windows + Demo_D 下先 import torch 再走 src.training → OpenMP DLL 冲突段错误(EXIT=-1073741819,pytest 报 exit 5),最小复现两行 import;**Linux 云端不复现**。新测试若 import `src.training` 链需注意导入顺序。

## `test_official_head_losses.py` — 三头损失外挂(10 用例,2026-08-12 阶段 2.4)

`compute_official_head_losses` + `MultiTaskLoss` 的固定 λ 外挂:legacy 三头全缺席零回归(返回 0.0+空 dict)、λ 精确线性、全 invalid 精确 0.0 无 NaN、validity 必须 targets 顶层直传键(塞 label_validity 静默失效)、[B,1] validity 拒绝、**梯度隔离**(cbm_head/seg_head/dec_stages 梯度精确 0、cls_backbone>0)、`register_buffer(..., None)` 不进 state_dict(legacy 零影响)、buffer 随 model 走 device。

## `test_tirads_head.py` — TI-RADS 头(backlog #124,15 用例,2026-08-13)

目标三(60%权重)四子任务里此前唯一无训练信号的任务补齐后的单测。判据强度参照 extension/clinical_t 头:① 输出契约(形状/`tirads_probs`和为1/fp32);② **梯度隔离**(五个 CBM 概念 logits 全部 `.detach()`,`test_head_does_not_backprop_into_cbm_concepts` 去掉任一 detach 即失败;`test_head_concepts_actually_feed_output` 反证非空转);③ **罗马数字精确查表**(`test_roman_numeral_exact_match_not_substring`,IV/III 不被子串判成 I/II,loader 侧 `official_loader._parse_tirads`);④ **masked loss**(全 invalid 精确 0 且无 NaN、λ 线性缩放、类权重按标签而非预测索引、validity 必须 [B] 一维、标签越界拒绝);⑤ **可选二次有序项**(默认 `ordinal_weight=0` 对 total 精确无贡献,权重>0 时才建图——2026-08-13 Gate2 审查后从"无条件计算"改为门控,`test_ordinal_term_zero_weight_has_no_effect`/`test_ordinal_term_penalizes_far_predictions_more` 双向锁定语义);⑥ `inverse_frequency_weights` re-export 校验(与 `clinical_t.py` 同一函数对象,非复制)。

## `test_mixup_label_dominance.py` — 四头监督闸门 + 软 lam 保护(12 项,2026-08-15;并对应 `src/training/AGENTS.md` 闸门红线、`test_train_step_head_targets.py` 的 mixup 归属第②条。总应付口径另见 `test_official_schedule.py` 头部注释)

钉死 `build_official_head_targets` 的归属闸门改为 **lam 阈值分派**。① **主导方明确必须保留监督且标签继承对称正确**(`mixed = lam*recipient+(1-lam)*donor` → lam 小=donor 主导;6 参数化:0.00/0.05/0.35→donor 标签,0.65/0.95/1.00→recipient 标签,边界值 0.35/0.65 掐在阈值上)——方向判反比零监督更糟(错标签是学反);② **归属不明(0.35<lam<0.65)仍屏蔽**(四头是硬标签 masked CE,真·五五开的混合图配单一硬标签就是喂错标签,这条**不是** bug);③ **CutMix 行为完全不变**(其 lams 已是 0.0/1.0,新阈值对其完全等价的边界条件);④ 🔴 **反向保护:断言 `mixup.py` 必须仍产出软 lam 且 `mask_weight` 恒 1.0**——若有人"顺手"在增强侧硬化成 0/1,CBM 概念损失(`concepts.py` 的 `lam*loss_a+(1-lam)*loss_b`)与 SLR 的软插值数学会被静默改掉;修复范围刻意只在 train_step 的闸门;⑤ **量化修复收益**:Beta(0.2,0.2) 4000 抽样的 `(lam<=0.35)|(lam>=0.65)` 保留率 > 0.75(修复前 MixUp 分支保留率恒 0%)。全部 CPU、无权重。本次改动新增测试数 +12(加上 `test_dynamic_frame_cache.py` +9 共 +21),精确值以 `pytest --collect-only` 为准。

## `test_dynamic_frame_cache.py` — 预抽取 PNG 缓存契约(9 项,2026-08-15;`test_complete_pkl_is_accepted_by_the_real_gate` 是 2 参数化用例)

`OfficialThyroidDataset.__getitem__` 动态分支命中 `cache_img` 时不得再碰 `.nii.gz`:缓存命中路径像素与走原路径**逐位一致**、mask 前景像素数精确、缺文件**抛异常**(不静默零图——静默补零=拿全黑图+全零 mask 当真样本训,loss 照常降查不出)、无 `cache_img` 键**回落原 .nii.gz 路径**(向后兼容零回归,用不存在路径反证进了解压分支)、`frame_key` 正确区分同影像不同 mask 且同输入稳定(只哈希影像路径会撞车,让两个病灶读到同一张 mask,不报错、标签与像素静默错配)。另 2 项钉 `samples.pkl` 契约:**`--limit` 残件(`_meta["partial"]=True`)必须被硬拒**(残件里未处理的动态样本没有缓存 → 静默回落到全量解压路径,属"合法但全错",堵在入口才安全)与反向边界(完整 pkl 的 `partial` 键缺失/False 不得被误拒)。

## `test_preextract_parallel.py` — 预抽取并行化契约(5 项,2026-09-11 新增)

`preextract_dynamic.py` 的 `--workers` 进程池改造回归。🔴 **fixture 策略教训:mock 与
`ProcessPoolExecutor` 天然不兼容**——monkeypatch 只在主进程生效,子进程 re-import 模块后
patch 不存在(首版用 monkeypatch mock `load_nifti_frames`,并行臂全失败 3 例红);故本文件
**不 mock 任何 IO**,用 nibabel 造真实 mini .nii.gz(6 帧 32×32,含一个真损坏文件走
IMG 失败路径),主/子进程走完全相同的真代码路径。锁定:① `_process_one_file` 单元回填
(成功回填路径/越界帧回填 None 且记 failure);② **并行 workers=2 vs 串行 workers=1 的
PNG 逐字节一致** + pkl 样本逐字段等价;③ 幂等重跑不重写已有 PNG(mtime 不变);④ 分辨率
抽检护栏串行/并行两路都 SystemExit 硬拦(probe_hw 从 worker 报回主进程)。**本测试抓出
并已修的真 bug**:「已存在(跳过)」样本必须回填 fills,否则幂等重跑把全部已缓存动态
样本当失败剔除、pkl 只剩静态样本(静默丢全部动态数据,旧版串行在扫描阶段就写
`s["cache_img"]` 故无此问题——重构丢了这个语义)。

## `test_static_tta.py` — 静态 hflip TTA 契约(20 项,2026-09-12 新增,D2)

D2 TTA 的契约锁。fixture 用「确定性的空间不对称响应」构造法(镜像测试须验
区分力):假模型分割 logits 只随归一化列坐标线性变化(左右不对称,翻转前向
≠ 原图,TTA 平均可解析验证),分类头输出恒定。锁定 5 条契约:① **分类各头
(含位置头)与无 TTA 逐位一致**(禁止融合);② mask = 两路 sigmoid 概率的逐
像素平均(解析式对拍);③ TTA vs 无 TTA mask **必须不同**(fixture 区分力
反证,防测试退化成全 GREEN);④ 翻转前向异常分流——OOM 回退单前向(逐位
等价),**非 OOM RuntimeError 冒泡**(Gate2 F2:except 收窄到 _is_oom,真
故障不得静默劣化成半 TTA);⑤ override NaN 硬拦 + **形状 dim()!=2 硬拦**
(Gate2 F3:3D/4D 在下游 cv2.resize 被当多通道图静默放行,与 t_probs 长度
校验同一教义)。另钉 env 开关边界 9 档 + override=None 逐位等价旧版。

## 测试总数(单一权威,2026-09-13 实测)

**全套 1180 collected**(2026-09-16 阶段3 手段4 滞回生长 Gate2 收官,`Demo_D`
环境实测(--ignore=test_preextract_parallel.py,该文件依赖 nibabel 本环境缺失;
演进链 1104→1124→1133(阶段1方向A轮,详见下方历史条)→1150(阶段2 seg-only:
`test_seg_only_schedule.py` 7 + `test_seg_only_losses.py` 10)→1151(Gate2 修复轮:`test_seg_only_losses.py` +1 接线断言/官方对拍/legacy-masked 等价)
→1156(修复轮专审:`test_seg_only_gate.py` 新增 6——护栏纯函数终于有测试,
→1158(第三批专审:NaN 防御用例+γ 起点断言+nextafter 边界重构;含顶层 None 不崩/缺值报无法判定/恰好跌0.01档)
→1180(阶段3 手段4:`test_nodule_split.py` TestHysteresisBinarize 7+哨兵/1px核
回归锁 2、`test_probability_output.py` TestGrowThresholdEnv 5+敏感带 1、
`test_diagnose_detection.py` grow 扫参轴 3;Gate2 三路审查 2026-09-16)。
判回归须同
环境前后对拍。
`test_seg_only_schedule.py`(新增,7):`configure_official_stage(seg_only=True)`
契约——恒可训集=SEG_TRAIN_PREFIXES(vclub 显式冻,与 ete_only 相反)/lr 恒
Stage C 档/param_groups **两档 lr、四组**(每档 decay/no_decay 拆两组;decoder@head_lr、seg_backbone@
head_lr×backbone_lr_scale,不是 ete_only 式两组)/optimizer 覆盖=可训参数
精确集合/AUCM 对偶恒冻/与 ete_only 互斥(实测抓到一个真 bug:互斥检查原写
在 seg_only 分支内部,而 ete_only 分支先 return,导致检查永远走不到——已
把检查提到函数最前)/默认路径零回归。
`test_seg_only_losses.py`(新增 10+Gate2 修复轮 1=11):`compute_segmentation_loss` 的
Boundary(Kervadec 2019)/Lovász-Hinge(Berman 2018)扩展——权重=0 零回归/
GT 全零阴性样本恒 0 不炸/valid_mask=0 梯度精确 0/legacy 路径对非零权重
显式 raise(防静默丢变量,r7 事故同型)/**两项均实测抓到第一版符号错误**:
Lovász 用 inclusive-cumsum + 商法则展开时 worst 预测得负分(应最大);
Boundary 用纯内部距离(内部为正)时对齐预测的 loss 反而大于偏移预测(方向
反了),已改带符号距离场(GT 内负/外正)。均已钉方向性 + 梯度符号回归测试防回退。
🔴 **Lovász 第二版仍错(2026-09-14 Gate2 实锤)**:exclusive cumsum 只是
索引平移、漏了官方差分步 `jaccard[1:]-=jaccard[:-1]`——量级 O(1e5) 且
梯度方向不对(cos 0.40),而**方向性测试(perfect<half<worst)对两版都
绿,无区分力**;修为 inclusive+差分后与官方参考逐值对拍全等、量级回 O(1)。
**教训:方向性测试钉不住"错误但单调"的实现,数学函数必须与参考实现逐值
对拍**。旧记"O(1e4) 量级/不能照抄同档权重"是该 bug 的症状非固有属性,
已废(见 config 注释)。分项经 loss_dict 的 `seg_boundary`/`seg_lovasz`
进 JSONL,开工前跑一步真实数据核对分项与 seg 主项同档再定权重。
🔴 **同轮修了一个独立生产 bug**:`validate_step.py` 的 targets 组装此前
漏加 `valid_mask` 键(只有 train_step.py 加了),导致验证阶段永远走
`compute_segmentation_loss` 的 legacy 分支——boundary/lovasz 权重=0 时
无副作用被掩盖至今,阶段2 加显式 raise 后当场炸出。已修(2026-09-14 起
与 train_step 同存在性闸门:labels 有 valid_composition 才写,两侧同式),
不影响既有 dice 指标计算(dice_sum 独立算,不吃 valid_mask)。
⚠️ val_loss 口径自本轮起与 r5/r6/r7 轮 JSONL 不可比(此前 train masked/
val legacy 本就不同式);验证阶段权重非 0 时也会计算两项(EDT 75ms/批,
val_loss 随实验权重漂移,选模用 composite 不受影响)。

(历史)1133 collected(2026-09-13 阶段1方向A轮,max_nodules env 化 +
扫参轴三处联动):1104→1124(TTA 默认值翻转 +0/归因脚本扩展 +4/`conftest.py`
env 隔离 autouse +0)→1133(max_nodules env + spy/护栏/裸轴/静态-only 5 +
env 解析含上界/12.9 哨兵参数化 +4/测试类归位 +0)。
`test_nodule_split.py`(+12=53):`TestMaxNodulesEnv` env 三层优先级 + 畸形值
参数化(含 `12.9` 截断哨兵/`101`/`1e308` 上界反例/`0` 灾难值) + ceiling 放行
决策值(30/50/100);`TestSplit2DMisc` 归位(原 TestMinAreaEnv 的两个非 env
测试,TestMaxNodulesEnv 插入时被吞,按类名选跑会漏)。
`test_diagnose_detection.py`(+5=21):max_nodules int 轴解析(20.5/0 报错、
值须 int 非 float)/双池派生/spy 钉 kw 透传真达 split_nodules(变异实测:
删透传后 dual_pool 测试零区分力,fixture 实例数<20)/裸轴派生空护栏拦截
(修复:绕护栏静态池静默缺席)/静态-only 不拖动态(修复:combos_dynamic 无条件
吸收会拉起全帧动态推理跑两遍)。
`tests/conftest.py`(新增):autouse 删 `AICOMP_MAX_NODULES`/`AICOMP_MIN_AREA_RATIO`
——不传参的既有测试在 env 残留时 14 假红(实测),反过来也会掩盖真回归。
`test_diagnose_detection.py` 归因脚本扩展(+4):n_pred 直方图分池+饱和率
(anomaly 剔除/空池 None)/候选数交叉表分桶/sweep sidecar 入口拦截回归锁。

(历史)1104 passed + 1 skipped(2026-09-12 ETE-only+mask清理轮;
1064→1084→1089→1104 = `test_static_tta.py` +20 / `test_ete_only_schedule.py` +5 /
`test_mask_clean_2d.py` +10)。
`test_mask_clean_2d.py`(10 项):静态 2D 形态学清理契约——默认关逐位等价旧版
(红线)/填孔(环→实心)/缝缝合(穿透缝 2 实例→1)/实心恒等/🔴 竖带反例(四角
种子修复的回归锁:单种子把外部背景当孔洞,实测 53% 图幅)/🔴 flood no-op 兜底
(填孔后>0.5 回退原 mask)/垃圾 env 回关。
演进链 1016 → **+27 = 批量前向+内存治理**(2026-09-09,
`test_video_batch_inference.py` 7 + `test_nodule_split` 稀疏往返 1 + 其余修复)
→ **1043 → +16 = 零训练四件套**(2026-09-10:`test_probability_output` 阈值 env
矩阵 ~17(含 (key,dynamic) 成对参数化 18 组)、`test_nodule_split` env 三层优先级
6(含 inf/nan 反例)、`test_v21_output` 期望取整 5(尖锐/分叉/低尾/半整数tie/
hedge 期望取整口径)、`test_diagnose_detection` 扫参 5(一次前向×多组合/阈值轴/
动态轴/解析表单/归因脚本)。
Gate2 双路审查修复的 3×P1+5×P2 全部随测试钉死:sweep sidecar gts 行补
max_iou/area_ratio(归因脚本读缺键会出全 miss 假报告)、组合派生空护栏+
未知键白名单(键名拼错白跑 54 分钟)、阈值钳制测试首版次序错(动态键的钳制
从未被真正测过,0.5==0.5 巧合通过)、inf/nan env 反例(min_area 的
`AICOMP_MIN_AREA_RATIO=inf` 曾只挡负值不挡 inf → OverflowError 炸整例)、
`_qwk_round` 半整数 tie 钉死(banker's rounding,E=0.5→码1/floor(x+.5) 会得 2,
防复刻分叉)。
此前口径:**1043 passed + 1 skipped**(2026-09-09 批量前向+内存治理轮,含
Gate2 A 路 P1 修复的 OOM traceback 钉显存 RED 测试)。
此前口径:**1016 passed + 1 skipped**(2026-09-06 二轮三路复审+D 路修复后,`torch28` 预演
环境实测)。演进链 1002 **+5 = 二轮复审修复**(2026-09-06:第二轮 A/B/C 审查挖出
4×P0/P1——A-F1 invalid 桶 KeyError/A-F2 匈牙利少算 TP/A-F3 dedup 删真 TP/
C-F1 静态 fake 自证键——修复时 `test_detection_match.py` 22+5=27、
`test_diagnose_detection.py` 5 不变但 P2 fixture 加近失碎片;D 路复核 7 处修复
全对、5 变异注入 4 RED)。
此前口径:**1002 passed + 1 skipped**(2026-09-05 检测诊断仪表盘后,`torch28`
预演环境实测,含修复后重跑)。演进链 975 **+27 = 检测诊断仪表盘**(2026-09-05:
`test_detection_match.py` 22 + `test_diagnose_detection.py` 5;
三路 Gate2 审查 A/B/C 后修复 B-P0(动态链漏注入 n_total_frames)×1 +
A-P1/C-P1(dedup 语义双向失真)×2 + P2 若干并补测试;
**变异注入四连 RED 验证**:`n_sampled=32` 替代 `len(per_frame)` / gap 去掉 `+1` /
静态 conf 换常量 / 静态插值方式错误——四个镜像漂移全部被等价回归拦住)。
此前口径:**975 passed + 1 skipped**(2026-09-03 推理提速三件套后,`Demo_D` 口径,
`--collect-only` 实测 **976 collected**)。演进链 940 **+19 = backlog #128**
(损失 10 + 监控 9)→ 959 **+8 = Gate2 三处修复**(负 la_tau 守卫 6 项 /
`ete_probs` 长度契约 1 项)→ **967 +2 = 变异审计补测**
(2026-08-30 系统变异审计发现「丢 dist 项」「ordinal 归一化指数错」两个变异在既有
测试下存活——`total` 的数值构成与 `/(C-1)²` 归一化此前无任何数值断言,补
`test_ordinal_distance_loss_exact_value_with_normalization` /
`test_compute_extension_loss_total_decomposes_into_qwk_and_dist` 后两变异均被杀死,
10 变异 8 KILLED / 2 存活且存活者已全部补测覆盖)。
**969 +6 = 推理提速三件套**(2026-09-03:审计帧抽样 3 项 + Dedup 并发加锁 1 项 +
worker 默认值改名 1 项 + api fixture 隔离 1 项)。
⚠️ 上一段(940+1)时曾罕见地**两个环境同数**:`Demo_D` 与 `torch28` 预演环境都是
940+1 —— 那是巧合不是规律,别据此以为跨环境可比(见下方「跨环境不可比」条)。

🔴 **backlog #128 的 19 项在系统 Python 3.14 环境的口径是 793 collected /
785 passed + 14 skipped**(改动前同环境 775 passed)——**与 `Demo_D` 的 959
不可比**(该环境本就少收集若干模块,缺 pandas/openpyxl 等依赖致整模块 skip)。
**不要凭一个环境的数推算另一个环境**(memory `test-count-not-comparable-across-envs`)。

演进链:**891**(#127 后)**+32 = #116b 四头接线** → **923** **+6 = 上云前终审修复**
→ **929** **+11 = 第三轮 review 修复** → **940**;
之后 **+10 = backlog #128 LA-SoftQWK-OD 损失**、**+9 = 该 backlog 的 4 项 ETE
诊断监控**(2026-08-19,系统 py3.14 环境实测 775→785,**`Demo_D` 未跑、故上方
总数仍记 940**):

**+9 = 4 项 ETE 监控**(`test_metrics_official.py` 22→**31**,DR §5 `:118-138`):
`ete_pred_hist` / `ete_extreme_error_rate` / `ete_kappa_gap` /
`ete_logit_mean_per_class`。钉的是「诊断量混进评分」与「静默降级」两类:
① **监控子集必须与 `_ete_weighted_kappa` 同一口径**(主病灶+有标注+**恶性**)——
塞入良性/次病灶/无标注三种应排除样本反证,否则会出现「监控说分布正常、kappa
却在跌」的自相矛盾;② **诊断量不得进 composite**(`ete_pred_hist` 是 list,
混进 `np.mean` 会报错或静默产出错误标量,而 composite 是选模判据);
③ **`ete_logits` 键缺席时降级为 None,不得 KeyError、更不得拿 probs 顶替**
(softmax 平移不变、反推不出绝对量级,而 DR 判据 `z̄3−z̄0>0` 要的正是绝对量级);
④ **logits 必须与 probs 一起按病灶聚合**(漏了会停在首帧值、与已聚合的 probs 失同步);
⑤ **任一 kappa 不可算 → gap 必须 None 不得填 0**(0 会被读成"soft 与 hard 完全一致",
真相是"其中一个没算出来");⑥ 空 records 下四键仍须存在(否则 `.get(k)` 与
`k in metrics` 两种写法行为不一致)。
🔴 **JSONL 侧另有一条不在 pytest 里的硬约束**:`_epoch_metrics_extra` 必须**自行
清理数组内的 NaN** —— `JsonlLogger._drop_nonfinite` 只判 `isinstance(v, float)`、
**不遍历 list**,而这两项恰恰是数组;裸 `NaN` 会让严格解析器**拒读整行**,
V2.1 对"日志读不到"的判罚是「可能被标记为可疑对象」。已用直调 `_epoch_metrics_extra`
+ `json.dumps(allow_nan=False)` 实测(正常写出 / NaN 数组整项丢弃且不影响同批其他项 /
None 跳过),并经 `--mock-labels --smoke` 端到端确认四项真实落盘。

**+10 = LA-SoftQWK-OD 损失**(`test_extension_head.py` 39→49),明细见上方该文件行。
**+2 = 变异审计补测**(2026-08-30,`test_extension_head.py` 49→**51**):见上方
「测试总数」节——两个存活变异(dist 项丢失 / ordinal 归一化指数)的数值断言。

**+11 = 第三轮 review 修复**(929→940,2026-08-17):
- `test_missing_n_total_frames_raises_not_guesses`(改写:原 `..._falls_back` 钉的是
  「回退猜 Z 长度」这个**已被判定为整例清零风险**的旧行为,现钉硬失败)
- `test_invalid_n_total_frames_raises`×3(0 / 负数 / None 三种畸形值)
- `test_official_head_wrong_length_raises`×8(四头 × ±1 双向参数化 —— 双向防
  「只校验了下界」这类半拉子实现)
- `test_official_head_length_constants_come_from_models`(常量必须来自 `src.models`
  单一权威,不得在推理侧另抄一份)

上一段演进 **923→929 = 上云前终审修复**:
`test_mal_prob_is_lesion_level_mean_not_keyframe`(P0.1 良恶性口径分裂)+
`test_static_mal_prob_unchanged`(反向保护)+ `test_non_finite_rejected`×2(P2.4 对冲键 NaN)+
`test_sampled_frames_do_not_fragment_one_nodule`(D 路 P1 动态碎片)+
`test_temporally_separate_nodules_not_merged`(反向:不得粘连)。
🔴 后两条**已做变异测试**:回退 `close_gap_frames` 自适应后两条都变红,证明有区分力。
再上一段:**856**(2026-08-16)**+35 = `test_seg_guards.py`**(backlog #127 分割护栏,
判据从 dice→概率 spread→预测 mask 前景占比两次改版的回归钉子,含 **9 路变异测试**
与 `test_v2_hole_*` 两个反例——那两个把「有结构但 max=0.45 被放行而 mask 仍全零」
和「常量图+4 个异常像素」钉死,勿删);
856 由 841 **+15 = 分割解冻契约改动**:`test_official_schedule.py` +8(四个旧断言翻转 +
新增 `test_seg_trainable_and_in_optimizer_in_all_stages`/`test_seg_lr_tiers` 各 3 参数化、
`test_seg_backbone_group_is_single_tier`、`test_no_param_in_two_groups`)、
`test_metrics_official.py` +7(seg_dice 并入 composite:默认/生效/退化/非有限/越界值域/
numpy标量/空 records);再往前 +9 = `test_dynamic_frame_cache.py`(动态帧预抽取缓存 8 条 def /
9 项,完整-pkl 放行那条是 2 参数化:partial=False 与键缺失);再往前 +6 =
`test_official_schedule.py` 新增 `test_jsonl_logger_extra_fields_merged_and_reserved_protected`
(含规范字段防覆盖的 5 个撞名子断言)与 `test_jsonl_logger_drops_nonfinite_extra`
(NaN/Inf 丢弃 + 严格 JSON 解析)。

- **四头(位置/侵犯/临床T/TI-RADS)相关单测合计 154**(阶段2.0 `test_head_interfaces.py` 14 + 阶段2.1 `test_location_head.py` 25 + 阶段2.2 `test_extension_head.py` **51**(含 backlog #128 LA-SoftQWK-OD +10、2026-08-30 变异审计补测 +2) + 阶段2.3 `test_clinical_t_head.py` 27 + 阶段2.4 `test_official_head_losses.py` 10 + `test_train_step_head_targets.py` 12 + backlog #124 `test_tirads_head.py` 15),Demo_D 实测通过;A800(torch2.6)/torch2.8 预演环境覆盖 TI-RADS 头待下次云端对拍确认(本轮仅 Windows+CPU 与真 CUDA `probe_heads_cuda.py` 双路验证)。
- ⚠️ **跨环境不可比**:同代码 torch2.8 预演环境(`C:\Users\cjy\envs\torch28`)为 **789 passed + 6 skipped**(2026-08-13 实测,**collected 795 与主环境一致**);旧记的 620 是 08-12 快照、已失效,且 08-13 补装 pandas+openpyxl 之前该环境只能收集 782 —— `tests/test_official_loader.py` 整模块被 `pytest.importorskip("pandas")` 跳过,预演对 `official_loader` 覆盖率为 0。skip 数随依赖集与 `os.name` 平台门浮动,**判「有无回归」必须同环境前后对拍**,详见 memory `test-count-not-comparable-across-envs`。
- **下方分模块明细数字为历史演进累积、可能有偏差**——精确值一律以 `pytest --collect-only tests/` 实跑为准,不逐一核对旧数。
- Phase5(2026-08-10)交付四模块:`test_nifti_io`(24) + `test_nodule_split`(28) + `test_prediction_json`(45) + `test_v21_output`(11) + `test_service_runner`(12) + `test_case_discovery`(40) + `test_service_api`(20)(含 pydicom/nibabel 依赖的 file 在无依赖 CI 上 skip)。

## `test_phase_refactor.py` — 重构不变式(22 项)

2026-08-11 R6 新增 `test_famo_directional_correctness` + `test_latent_lagrangian_constraint_direction` 2 例;CI 已核实绿(run 31365316179/dc4b827)。


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
| `test_latent_lagrangian_constraint_direction` | 拉格朗日乘子方向锁定:约束满足(g≤0)松绑、违反(g>0)加压;佐证 #120 符号误报(2026-08-11) |

---

## 运行

```bash
# 完整套件
python -m pytest tests/ -q

# 仅重构回归
python -m pytest tests/test_phase_refactor.py -q
```

所有 test 函数既可加入 `pytest`,也可作为 `python tests/test_xxx.py` 独立运行(每个文件底部有 `if __name__ == "__main__": pytest.main(...)`)。
