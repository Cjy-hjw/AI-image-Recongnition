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
| `test_prediction_json.py` | prediction.json schema 构造与校验(2026-08-01 新增;**2026-08-10 A3 起 v21_output 经它组装/校验,并入 V2.1 交付链路**) | 19 | Phase 5 |
| `test_v21_output.py` | **V2.1 单病例输出写盘入口**(2026-08-10 A3 新增)——`write_case_v21` 静/动两链路:静态把 case_pipeline 结果落成 `{case_dir}/{subdir_uid}/nodule_NNN.nii.gz` + `prediction.json`(StudyUID=PatientId、MaskFile 相对路径正斜杠、原子写、阴性空数组合法、缺 orig_hw 报 V21 错误);动态逐帧 resize 回原图 + scatter 回填全帧 3D mask + 3D 连通域跨帧关联;占位维度(ETE=1/Location 均匀/T=0)结构合规 | 11 | Phase 5 |
| `test_service_runner.py` | **runner 批量编排**(2026-08-10 A2 新增)——mock 掉 heavy 依赖只留编排逻辑:逐病例调 write_case_v21 且 case_dir={output}/{patient_id}、静/动 kind 分派、processing_ms 非负 int、gatekeeper 每病例照跑、**单病例失败隔离**(一例 diagnose 崩溃不影响其余 + 落结构合规 ERROR 占位)、loader 失败隔离、**模型只 build 一次**(跨病例 + 跨多次 build_service_runner 缓存)、空/无效/全非影像目录不崩 | 11 | Phase 5 |
| `test_case_discovery.py` | **病例发现**(2026-08-10 A2 新增)——`discover_cases` 递归遍历 + 静/动分派(.dcm→static,.nii.gz/.nii→dynamic)、静态读 DICOM 头取 PatientID/SeriesUID、动态按组委会口径取直接父目录名(PatientId=AccessionNumber)、路径推断兜底、空/不存在目录返回 []、确定性顺序;依赖 pydicom/nibabel 缺则 skip | 39 | Phase 5 |

---

## `test_official_loader.py` — 官方数据加载器契约(8 用例,2026-08-10)

合成 `.dcm`/`.nii.gz`/xlsx fixture(运行时生成,不入库二进制),依赖 pydicom/nibabel/pandas/openpyxl(缺则 skip)。

| Test | 验证内容 |
|:---|:---|
| `test_parse_xlsx_mixed_cells` | 混写单元格按 prediction_json 权威常量解析(2左中叶→idx1、ETE 2甲状腺包膜→码2-1)、空列→valid=0 |
| `test_build_samples_dispatch_and_skip` | 静动分派、清晰帧行跳过、动态按非零 mask 帧展开 |
| `test_output_contract_static` / `_dynamic` | 输出契约 img[1,S,S]∈[0,1]/mask 二值/labels 含 valid_*;动态取正确非零帧 |
| `test_patient_grouped_split_no_leak` | train/val 患者无交集(rules.md §5 防泄漏) |
| `test_ete_parse_edge_cases` | ETE 纯中文回退、越界数字(如 `0无`)不产生负类索引 |
| `test_location_pure_chinese_and_negative` | 位置纯中文回退、阴性/未知→valid=0(绝不默认 0=左上叶) |
| `test_dynamic_no_mask_skipped` | 动态病灶无 mask→无法定位帧→整例跳过 |

## `test_phase_refactor.py` — 重构不变式(全套 **588 passed + 1 skipped**(2026-08-11 本地实测;此前基线 357 collected +8 official_loader 2026-08-10、本地 UTF-8 通过;CI 已补 pandas/openpyxl 纳入回归,已核实绿 run 31365316179/dc4b827),此文件 22)

> **2026-08-10 Phase 5 加入新测试后,全套 = 357 collected → 更新**:表格其余文件外,另加
> `test_nifti_io`(24)/`test_nodule_split`(28)/`test_prediction_json`(45)/`test_v21_output`(11)/
> `test_service_runner`(11)/`test_case_discovery`(39)/`test_service_api`(20)。**本地收集总数 463
> tests collected**(2026-08-10 实测;含 pydicom/nibabel 依赖的 file 在无依赖 CI 上 skip)。根
> `AGENTS.md`/`tests/AGENTS.md` 里先前的 356/580 是不同口径基线(580 含 Phase 5 前的旧计数,现
> 按 grep `def test_` 实测校准为 **463**;Phase 5 V2.1 交付四模块测试另行核算)。

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
