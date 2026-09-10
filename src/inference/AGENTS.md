# src/inference/AGENTS.md - Shared Inference Pipeline Specifications

This package provides reusable inference modules shared by `predict.py` (production) and `evaluate.py` (holdout benchmarking).

---

## Package Structure

| File | Responsibility |
|:---|:---|
| `io.py` | Image loading (DICOM + PNG/JPG) and PHI anonymization;通道/位深归一化工具 `_to_gray`/`_rescale_to_uint8` 下沉于此(静/动两路径共用,video_io re-import)。`_to_gray` 覆盖全部通道形态((H,W)/(H,W,1)/(H,W,3)/RGBA/channel-first/多光谱 均收敛 2D,杜绝 3D 漏过致下游 `preprocess_image` 5 维崩溃),并以 `channel_order` 区分 RGB(DICOM 像素)与 BGR(OpenCV 视频帧)避免多普勒伪彩加权错位。单图 DICOM 经 `_to_gray` 收敛灰度 |
| `gatekeeper_runner.py` | 4-layer gatekeeper audit pipeline(实为 **3 阶段** auth→dedup→tamper;copy_move/splice 为 CatNet `detect()` 内部子判据,非独立层)(predict.py 单图 + video_predict.py 动态图)。**2026-07-29 新增 `collect_soft_flags()`**:把 `CatNetTamperingDetector.last_soft_flags` 归一化成可 JSON 序列化结构落进 audit——此前生产代码零读取该属性,仅 copy_move/splice/BAG 命中的图 `flagged` 为 False、result.json 完全沉默,而 copy_move 在 359 张真实图上误报 27.3%,信号全丢 |
| `model_runner.py` | Model build, image preprocessing, forward pass |
| `postprocess.py` | Prediction decoding and TI-RADS breakdown(可选接 `calibration.TemperatureScaler` 做温度缩放) |
| `calibration.py` | Post-hoc 置信度校准(温度缩放):`TemperatureScaler`(每任务标量 T,缺参数即 T=1.0 降级)+ 离线拟合 `fit_temperature`/`fit_ordinal_temperature` + 诊断 `compute_ece`/`reliability_diagram`。真实数据 2026-08-01 初赛才到,现仅骨架 |
| `export.py` | Artifact saving (mask, report, JSON, Grad-CAM overlay) |
| `video_io.py` | 动态图/视频(cine)抽帧:MP4/AVI/MOV/MKV via OpenCV + 多帧 DICOM cine via pydicom;复用 `io._PHI_FIELDS` 脱敏,经 `_to_gray` 统一灰度兜底(含彩色/多普勒 cine);OpenCV 视频帧传 `channel_order="bgr"`、DICOM cine 用默认 `"rgb"`。供 `video_predict.py`(根目录) |
| `video_aggregate.py` | 视频推理纯逻辑:`uniform_sample_indices`(均匀抽样下标)+ `aggregate_frame_predictions`(帧级聚合,恶性概率取**最大帧**做最终判定,召回优先;**空输入安全返回 `keyframe_index=-1` 不抛**,由 `video_predict.py` 的显式守卫拦全帧失败,见下方「坏帧隔离」) |
| `nifti_io.py` | **2026-08-01 新增,已接入(A1/V2.1 交付链路)**。NIfTI(.nii/.nii.gz)读写:`save_binary_mask`(交付 mask,值域强制 `{0,1}`)、`load_nifti_frames`(动图输入,签名与 `load_video_frames` 逐位对齐)、`scatter_frames_to_full_length`(抽样帧回填全帧序列)、`is_nifti_path`。A1 起 `entrypoint.sh` 按 `.nii.gz` 分发,`src/service/case_discovery.py`/`runner.py` 经本模块读写动态图输入与逐结节 mask 落盘。见 §8 |
| `nodule_split.py` | **2026-08-01 新增,已接入(v21_output)**。整图 mask → 逐结节实例:`split_nodules`(2D/3D 连通域)、`crop_nodule`(带上下文边距裁剪)、`detection_confidence`(实例区域概率均值)。A3 起 `v21_output.write_case_v21` 经 `split_nodules` 把整图 mask 拆成逐结节实例。见 §9 |
| `prediction_json.py` | **2026-08-01 新增,已接入(A3)**。官方 `prediction.json` schema 构造与校验:`build_nodule_entry`/`build_prediction_json`/`validate_prediction_json` + 三个新维度的类别映射常量。V2.1 起 `StudyUID` 填 PatientId、`MaskFile` 为相对病例一级目录路径。见 §10 |
| `case_pipeline.py` | **A0 重构新增**。CLI 与 service 共用的**单病例推理核心**(静态图 `diagnose_static_case` + 动态图 `diagnose_dynamic_case`),从 `predict.py`/`video_predict.py` 的 `main()` 抽出,零行为变更。见 §6 |
| `v21_output.py` | **A3 新增**。**唯一 V2.1 写盘入口** `write_case_v21`:把 `case_pipeline` 结果落成官方 V2.1 逐结节交付形态(软/二值 mask resize 回原图 → `split_nodules` 逐实例 → 每结节 `.nii.gz` + `prediction.json` 原子写)。静/动两链路,动态逐结节 3D mask 跨帧关联(最简实现,@云TODO)。见 §12 |

---

## 1. `io.py` — Image Loading & DICOM Anonymization

### `load_input_image(input_path, output_dir)` → `(img_np, is_dicom, error_msg)`

- Supports DICOM (`.dcm`/`.dicom`) via `pydicom` and standard formats via `cv2.imread(..., IMREAD_GRAYSCALE)`.
- **PHI Anonymization** (HIPAA Safe Harbor): 仅覆盖 DICOM 中**已存在**的 PHI 标签(完整清单见 `io.py` 的 `_PHI_FIELDS`),涵盖患者身份(Name/ID/BirthDate/Sex/Age/Address/Telephone/OtherIDs)、就诊(AccessionNumber/StudyID/各类 Date)、机构与设备(InstitutionName/Address/StationName/DeviceSerialNumber)、医师与操作者(Referring/Performing/Operators)。日期类置 `"00000101"`,其余置 `"ANONYMIZED"` 或空串。
- 🔴 2026-09-01 起**不再落盘匿名化副本**(原 `output_dir/anonymized_image.dcm` 已删):官方 V2.1 评分器对预测目录做白名单校验(只允许 .nii/.nii.gz/.json),混入 .dcm 即「预测文件格式错误」整批判零(云端实测)。PHI 匿名化仅内存清洗;`output_dir` 参数为签名兼容保留、不再写任何文件。回归锁:`tests/test_video.py::TestZeroFileLeakRedline`。
- Returns `(None, False, error_msg)` on failure — caller must check `img_np is None`,
  print the error **并调用 `export.save_rejection_json` 落 `status="ERROR"` 的
  `result.json`**（2026-07-11 修复：此前只 print 不落盘，容器 exit code 仍为 0 但
  `output_dir` 是空目录——比 gatekeeper 一票否决/ `except Exception` 兜底的 ERROR
  状态还差，绕过了下游所有安全网）。`predict.py`/`video_predict.py` 的
  `load_video_frames` 加载失败分支同样遵守此契约。

---

## 2. `gatekeeper_runner.py` — 4-Layer Audit

### `run_gatekeeper_audit(img_np, config, device, weights_path, output_dir, deduplicator=None, auth_module=_AUTH_UNSET, tamper_detector=None)` → `(passed: bool, audit: dict)`

**Scope**: Production inference only. evaluate.py 仅导入 model_runner/postprocess/export 三模块；gatekeeper_runner 由 predict.py(单图) 和 video_predict.py(动态图) 使用。

**`auth_module` 哨兵语义(2026-08-10 R3)**:默认值已是哨兵 `_AUTH_UNSET`,非 `None`——**不传**(单图)才触发按需懒建一次;显式传 `None`(视频/服务已预建但权重降级失败)**不重建**,避免逐帧重复 load 权重 + N 次 WARNING。判空用 `getattr(auth_module,"predict",None)`,`None`/哨兵/stub 无 predict 均标 `auth_skipped=True`。

**`audit` 恒为 dict**(2026-07-29 契约变更;旧版通过时返回 `None`)——审计结论必须能落进 `result.json`,否则 phantom/synthetic 软 flag 只打印在 stdout、评测方拿不到。任何调用点都不得再假设"通过 ⇒ 第二个返回值是 None"。

### `resolve_enforcement_mode(config)` → `"soft" | "hard"`(2026-07-29 新增)
读 `config["gatekeeper"]["enforcement_mode"]`,归一化(strip+lower),缺省/非法值一律降级 `"soft"` 并打 `[WARNING]`。
**为什么默认 soft**:官方赛前动员会(2026-07-28)明确赛道七**豁免全部医保安全考核**,gatekeeper **零计分贡献**(2026-08-12《初赛评审规则简要说明》确证赛道七总分 = Sclin,无 ×100,详见 `docs/项目现状总纲.md` §1);且"真实病例被误判为伪造,诊断结果仍参与目标三四五计分"——前提是必须先输出诊断。故硬拒绝已成**纯负期望**:拒对了零收益,误拒一次即白丢该病例全部临床得分。**失败方向必须偏向"多输出一份诊断"而非"多拒一张真实图"**,所以拼写错误(如 `"Hard_typo"`)降级 soft 而不是报错或按 hard 处理。
- `soft`(默认、交付固定值):去重/篡改命中只落 flag,`passed` 恒为 `True`,诊断照常输出;
- `hard`:恢复"命中即 REJECTED、不输出任何诊断"的旧语义,仅供回归测试与未来规则变化时启用。

- 三个可选预建对象(视频逐帧复用,避免每帧重复加载权重/播种训练集):
  - `deduplicator`: 用 `build_seeded_deduplicator(config, device)` 预建；
  - `auth_module`: 用 `build_auth_module(weights_path)` 预建(单图路径懒建,跳过 `authenticity_model.pth` 每帧重复加载)；
  - `tamper_detector`: 用 `build_tamper_detector(config)` 预建(避免每帧重探 catnet 权重文件)。
- 三者均为 None 时内部即时构建——单图 predict.py 路径行为不变(向后兼容)。
- **无状态污染**:`is_duplicate` 只读比对,不把当前帧 `add_image` 入库;故同一 cine 相邻近重复帧之间不会互相误判(去重库只播种训练集)。

Pipeline:
1. **Authenticity** — `AuthenticityModule` (13-dim GLCM+freq+ELA+DND+TAI → MLP) → phantom/synthetic detection。**2026-07-12 统一软flag制度**:分类结果**不再**驱动 REJECTED,仅作审计字段(`is_phantom`/`is_gan`)。**权重故障处理(2026-08-10 R3)**：`AuthenticityModule.load_weights` 内部对权重缺失/损坏仍可 raise，但 `build_auth_module` 调用方 catch 后**降级为返回 None + `auth_skipped=True`**——`run_gatekeeper_audit` 标 `auth_skipped=True`、`is_phantom=0`、`auth_probs=None`，诊断照常产出；这是显式跳过的诚实记录(WARNING + `auth_skipped` 字段)，非伪装 pass，依据是 auth 赛道七零计分、其故障不该让唯一得分项清零。**主模型等计分组件故障仍走 ERROR 硬报错，红线不变**。详见 `src/gatekeeper/AGENTS.md` §1。
2. **Deduplication** — `Deduplicator` (pHash + cosine) seeded from train set。命中 ⇒ `would_reject_in_hard_mode=True`;**是否真拒绝取决于 `enforcement_mode`**(默认 soft 不拒绝,仅审计)。
3. **Copy-Move** — `detect_copy_move()` dense block-matching。**恒为软flag**(`last_soft_flags`),与 CatNet 状态无关,详见下方第 5 点。
4. **Splicing(硬粘贴)** — `detect_splicing()` 轴对齐长直缝 + 角点正交相交判据(压低误报至零)。**恒为软flag**,同上。
5. **Splicing(羽化/泊松)** — `tampering_catnet.CatNetTamperingDetector` CAT-Net v2 DCT/BAG 取证(块伪影栅格相位错位),可插拔降级 dct-bag。**2026-07-12 统一软flag制度后 `detect()` 判定逻辑为**:CAT-Net **自身**前向命中(本次调用成功)"硬"路 ∪ BAG 相位不一致(仅当 `bag_hard_reject=True` 显式开启;默认 `False` 仅软flag) ∪ 几何(copy_move/splice,**恒软flag,与 CatNet 状态解耦**,不再因"CatNet 这次没崩溃"被提升)。命中源写入 `CatNetTamperingDetector.last_hard_reject_source`("catnet"|"bag_phase"|None),`gatekeeper_runner.py` 据此拼精确 reason(`"Tampering detected (CAT-Net deep model)"` / `"(BAG phase inconsistency)"` / `"(source unspecified)"`)。**注意**:这里的"硬"只表示"该路命中会置 `is_tampered=True`",最终是否 REJECTED 仍被 `enforcement_mode` 顶层封顶。详见 `src/gatekeeper/AGENTS.md` §3。

**Audit dict**(恒返回)。`is_phantom`/`is_gan` 为纯审计字段——非零值**不代表**单独触发拒绝,只有 `is_duplicate`/`is_tampered` 会置 `would_reject_in_hard_mode`;phantom/synthetic 与去重/篡改并发命中时,`reason` 追加 `"(soft-flag: ...)"` 标注保留该并发信号(2026-07-12 code-review 补强,详见 `src/gatekeeper/AGENTS.md` §1):
```json
{
  "flagged": true,                     // 任一审计项命中(含仅软flag的 phantom/synthetic/copy_move/splice/BAG)
  "would_reject_in_hard_mode": true,   // **仅**去重/篡改硬判据命中 → hard 模式下会 REJECTED;软flag 恒不计入
  "enforcement_mode": "soft",
  "reason": "Duplicate image: ..., (soft-flag: Physical Phantom suspected), (soft-flag: copy_move suspected)",
  "is_phantom": 0,
  "is_gan": 0,
  "is_duplicate": 1,
  "is_tampered": 0,
  "soft_flags": [{"type": "copy_move", "bbox": [x1, y1, x2, y2]}]
}
```
**`soft_flags`(2026-07-29 新增)**:`[{"type": "copy_move"|"splice"|"bag_phase", "bbox": [x1,y1,x2,y2]|None}, ...]`,由 `collect_soft_flags(tamper_detector)` 归一化产出(旧 detector 无该属性 → getattr 兜 `[]`,不 AttributeError;`last_soft_flags` 每次 `detect()` 开头重置,故视频逐帧复用同一 detector 无跨帧污染)。**红线:软 flag 只计入 `flagged`,绝不进 `would_reject_in_hard_mode`**——copy_move/splice 恒为软判据(真实图误报 27.3%),BAG 仅在 `bag_hard_reject=true` 时才升级硬判据。
仅当 `enforcement_mode == "hard"` **且** `would_reject_in_hard_mode` 为真时返回 `(False, {...上述字段, "status": "REJECTED"})`,可直接交 `save_rejection_json`。soft 模式产物**不含** `status` 键(靠这点区分,回归测试有断言)。

**两个入口如何消费**(`result.json` 的 `gatekeeper` 字段。两条链路 schema **不同**——静态图是单图 audit,动态图是逐帧聚合。**唯一全路径都在的键是 `skipped`**;`enforcement_mode` 只在真正跑过审计时才有(`--skip_gatekeeper` 路径只产出 `{"skipped": true}`,别的键一个都没有)。消费方须先判 `skipped`,再看 `input_type` 取各自专属键):
- `predict.py` — `{"skipped": false, ...audit}`(静态图无跳过审计的入口,`skipped` 恒 false,保留该键是为让消费方两条链路共用同一份解析代码);
- `video_predict.py` — `{skipped, enforcement_mode, n_frames_audited, n_frames_flagged, flagged_frame_indices, any_flagged, any_would_reject_in_hard_mode, flagged, would_reject_in_hard_mode, keyframe_flagged, flagged_frames}`。**soft 下单帧命中不再让整段视频零诊断**(旧逻辑 `return`,32 帧任一误报即整段归零,是本次软化的最大受益点);`--skip_gatekeeper` 时该字段为 `{"skipped": true}`(无其它键)。
  - **`flagged`/`would_reject_in_hard_mode` 无前缀别名(2026-07-29 新增)**:与 `predict.py` 单图 audit 的键名对齐——此前视频侧只有 `any_*` 前缀,消费方(评测脚本/人工复核)无法对两条链路写通用读取代码。`any_*` 保留不删(向后兼容已写好的读取方),新增无前缀别名作为**两入口共有键**。
  - **被 flag 的帧仍参与 max 聚合**——这是有意为之:soft 的语义就是"命中只记 flag、诊断照出",把 flagged 帧排除出聚合等于变相执行拒绝,又回到本次要修的坑(且 `detect_copy_move` 在真实图有 27.3% 误报,排除大概率丢的是真帧)。
  - `keyframe_flagged` 显式标注**关键帧本身是否被 flag**——mask/报告/Grad-CAM 全出自关键帧,人工复核应一眼看见"这份报告出自一张被 flag 的帧",不必自己拿 `keyframe_frame_index` 与 `flagged_frame_indices` 做集合运算(二者同一坐标系,都是 `frame_indices` 的原始帧号)。

#### 坏帧隔离(2026-07-29,修复一轮自引入回归)

Phase 1 给 `decode_predictions` 全部分类头加了 NaN/Inf 硬校验(`_assert_finite` 抛 `ValueError`),但 `video_predict.py` 逐帧循环当时**无 try/except**——异常直接冒泡到最外层 `except Exception`,**整段 32 帧全废、status=ERROR**。性质与 Phase 2 花大力气修的「一帧 gatekeeper 命中拖垮整段」完全相同,只是触发源从"审计拒绝"换成"数值校验"。修复语义与 gatekeeper 软化一致:**丢一帧损失有限,丢整段 = 该病例全部临床得分归零**。

`video_predict.py` 逐帧循环现在的契约(2026-07-30 二轮收紧):
- **只捕 `(NonFiniteProbabilityError, AssertionError)` 两类**,记入 `failed_frames` 后 `continue`。前者是 `_assert_finite` 抛的**专属子类**(2026-07-30 新增,`ValueError` 子类),后者是 `preprocess_image` 的帧形状/维度校验(`assert img_np.ndim == 2`)。**初版用宽口 `except ValueError`,会把 cv2/numpy 在畸形帧上抛的同类型异常误当"可丢帧"吞掉**——故障被伪装成"少几帧"的 PASSED(/code-review Angle C 实测 `cv2.error` 非 ValueError 子类、但 numpy 会抛 ValueError,故必须用专属类型精确区分)。`RuntimeError`(CUDA OOM/权重维度不符)、`cv2.error`、`KeyboardInterrupt` 等真故障必须冒泡成 ERROR。
- **mask 也罩 `_assert_finite`**(2026-07-30):分类头有硬拦、mask 若漏拦,NaN logits 经 sigmoid 得 NaN,`(NaN>0.5)` 恒 False → **静默产出全零分割图**而 status=PASSED、报告正常,评测只看 JSON 察觉不到分割已废。
- **`frame_np` 存 per_frame 元素,`img_tensor` 改为只留当前最恶性帧**(2026-07-30 显存优化):32 帧(或不限帧 DICOM cine)的 GPU 张量全量常驻会在官方平台显存上界(A800 同量级;2026-08-12 实测 32G,见总纲 §3)上叠加模型+Grad-CAM 而 OOM。逐帧用严格 `>` 刷新 `kf_tensor`(与 `aggregate_frame_predictions` 的 `max(range(n),key=...)` 取首个最大严格同序),显存常驻从 32 帧降到 1 帧;关键帧处有 `kf_img_tensor_pos != agg["keyframe_index"]` 一致性锁(不等则 warning + 重算)。**错位陷阱(2026-07-29 实测复现)**:旧版 `key_tensors` 平行 dict 以全帧下标为键、`keyframe_index` 是 per_frame 下标,跳帧后两空间错位且 `key_tensors[kf_pos]` 恒取得到值、不抛 KeyError,mask/Grad-CAM 静默出自错帧;`frame_np` 存进 per_frame 元素后下标空间只剩一个。
- **全帧失败**用显式 `if not per_frame:` 守卫落 ERROR + `failed_frames` 明细,**不靠 `assert`**(assert 在 `python -O` 下会被整条剥离)。另有 `if agg["keyframe_index"] < 0:` **双重守卫**(2026-07-30):即便全帧失败守卫未来被误改成只 print 不 return,`per_frame[-1]` 负索引**不抛异常**而静默取末帧,mask/Grad-CAM 出自末帧而 JSON 写 -1,与 key_tensors 错位同性质。
- **丢帧如实记录,不静默少算**:`result.json` 顶层 `n_frames_sampled` 改取**抽样帧数** `len(frames)`(此前取 `agg["n_frames"]` 即成功帧数),并新增 `n_frames_diagnosed`/`n_frames_failed`/`failed_frame_indices`/`failed_frames`(含每帧 `reason`)。**ERROR 路径键集与 PASSED 对齐**(2026-07-30):全帧失败/无关键帧的 ERROR JSON 也带 `n_frames_diagnosed`/`failed_frame_indices`,下游按"两路径共有 schema 超集"统一解析时不会 KeyError。无失败时 `sampled == diagnosed`、`failed_*` 为空,退化为原行为。
- **静态图 `predict.py` 不软化**:单图 NaN = 该图没有任何有效诊断,编一份出来是假的;`report_generator.generate_report` 对 NaN `malignancy_prob` 本就是硬报错(2026-07-10 决定,`nan >= 0.5` 恒 False 会静默输出"良性倾向",癌症筛查假阴性),且 ERROR 的 result.json 本身是合法 JSON,比 NaN 污染的"PASSED"更可解析。

回归测试:`tests/test_gatekeeper_soft_enforcement.py::TestBadFrameIsolation`(坏帧隔离 / 关键帧错位指纹 / 全帧失败非 AssertionError / 无失败退化 / RuntimeError 仍冒泡 / 帧形状非法 AssertionError 丢帧 / keyframe_index=-1 显式 ERROR 不负索引静默取末帧)+ `tests/test_probability_output.py::TestNonFiniteProbabilityIsRejected`(专属子类锁定 + mask NaN 也拦)。

---

## 3. `model_runner.py` — Model Build & Forward

### `load_config(config_path)` → `dict`（2026-07-11 新增）
- 读 YAML + 提前触达 `config["environment"]["device"]`/`config["data"]["img_size"]`
  两个下游必用键，缺失/格式错误/文件不存在时立即抛异常。**调用方必须
  `try/except Exception` 包裹**，详见下方「配置加载失败契约」。

### `build_model(config, weights_path, device)` → `MultiTaskThyroidModel`
- 构建**不感知 LoRA** 的普通模型（`use_lora` 恒为 `False`）。训练侧若用 `use_lora=True`
  产出 checkpoint（如 `scripts/train_thyroidxl.py`），部署前必须先跑
  `scripts/merge_lora_checkpoint.py` 把 LoRA A/B 数学合并回普通 Conv2d，本函数才能
  加载——这样 LoRA 这个训练期 PEFT 技巧完全不泄漏进推理侧代码。
- **兼容两种 checkpoint 格式**：裸 `state_dict`（`train.py` 的
  `torch.save(model.state_dict(), ...)` 约定）与包装格式
  `{"epoch":, "stage":, "state_dict":, ...}`（`train_thyroidxl.py` 另存训练元信息）——
  按 `raw["state_dict"] if "state_dict" in raw else raw` 统一解包。
- **`load_state_dict(strict=False)` 后硬校验 missing/unexpected keys，非空则 `raise
  RuntimeError`**（2026-07-10 由 `strict=False` 静默容忍改为硬报错）。
  **背景（血泪教训）**：旧版 `strict=False` 且不检查返回值，曾导致两类静默降级：
  ①不解包包装格式时，710 个真实参数全部 key 不匹配、模型等于纯随机初始化；
  ②LoRA checkpoint 误喂非 LoRA 模型时，144 个 `lora_A`/`lora_B` key 全部静默丢弃，
  FT-LoRA 阶段学到的适配完全不生效——两种情况均无任何报错提示，训练产出的
  checkpoint 实际从未在推理侧生效过。**调用方必须 `try/except RuntimeError` 兜底**
  （见下方 Usage Contract 后的「失败契约」），否则交付容器会在权重不匹配时零输出崩溃。
- Falls back to random init with `[WARNING]` **only** when checkpoint file 不存在（此分支不受本次改动影响）。

### `preprocess_image(img_np, img_size, device)` → `torch.Tensor`
- `cv2.resize` to `img_size`, normalize `/255.0`, shape `[1, 1, H, W]`.

### `run_forward(model, img_tensor, precision, device)` → `dict`
- **`torch.inference_mode()`** — 所有推理前向统一使用 `torch.inference_mode()`，替代 no_grad。
- **`torch.autocast`** with device-adaptive dtype:
  - CPU + fp16 → fallback to fp32 with `[WARNING]`.
  - **2026-07-11 修复（两轮，CI test job 实测抓到第二轮）**：这条降级最初是
    "假的"——旧代码 `enabled=False` 但 `dtype` 仍传 `torch.float16`，
    `torch.autocast.__enter__` 在 `device_type="cpu"` 时**无条件**调用
    `torch.set_autocast_cpu_dtype(dtype)`（不看 enabled），只要 dtype!=
    `torch.bfloat16` 就必然 `TORCH_CHECK` 失败（容器锁定的 `torch==2.1.0` 上
    真实触发，CI `docker-build-smoke` job 首次跑通 video 冒烟时抓到；本机
    `torch==2.5.1` 已放宽校验不复现，属版本差异陷阱）。**第一轮修复**只堵了
    `precision=="fp16"` 这一种取值，却把 `precision=="fp32"` 的 dtype 从原本
    无害的 `bfloat16` 误改成 `float16`（错误假设"`enabled=False` 时 dtype 是
    no-op"——CI `test` job 的既有测试 `test_run_forward_fp32_equals_bare_forward`
    直接实测抓到这个新回归，Gate2 code review 对这条也判断错了）。**最终修复**：
    CPU 上 `dtype` 恒为 `torch.bfloat16`（不管 precision 取值），靠 `enabled`
    门控是否真正生效；GPU 分支维持原语义不变。见 `tests/test_model_runner.py`
    的 `TestRunForwardCpuFp16Fallback`（含显式 `test_fp32_precision_on_cpu_does_not_raise`）。
- **Safety clamp**: `torch.clamp(outputs["malignancy"].float(), min=-15.0, max=15.0)` under fp16/bf16.

---

## 4. `postprocess.py` — Prediction Decoding

### `decode_predictions(outputs, calibrator=None, *, dynamic=False)` → `dict`
Returns keys: `pred_mask`, `mask_binary`, `comp_idx`, `echo_idx`, `shape_idx`, `margin_idx`, `comp_prob`, `echo_prob`, `shape_prob`, `margin_prob`, `foci_prob`, `mal_prob`, `mal_probs`, **`ete_probs`, `location_probs`, `t_probs`, `t_conf`, `tirads_probs`**(官方四头,#116b 2026-08-17 接入)。
- **分割二值化阈值 env 化(2026-09-10 零训练四件套①)**:`mask_binary = (pred_mask > _seg_threshold(dynamic))`;静/动两条 env 独立(`AICOMP_SEG_THRESHOLD`/`AICOMP_SEG_THRESHOLD_DYNAMIC`),未设时恒 0.5 = 历史行为逐位不变;垃圾/越界/inf/nan 一律钳回 0.5(NaN 比较返回 False 走越界分支,不会异常)。**`dynamic` 由调用方传**:生产静态路径 `diagnose_static_case` 不传(False),动态帧循环经 `_decode_frame_sample`/`_diagnose_frame` 透传 True(kf 一致性锁同)。非 case_pipeline 的旧调用方(`video_predict.py:42` shim、`evaluate.py:102`)默认 False 行为不变。**全链只有这一个分割阈值**——下游(case_pipeline/nodule_split/v21_output/诊断脚本)只读 mask_binary 非零性,此处一改即全链生效。
- **官方四头(位置/ETE/临床T/TI-RADS)**:**原样透传各头已算好的 fp32 softmax**(`heads/*.py` 内部已 softmax),`decode_predictions` **不再二次 softmax**(`softmax(softmax(x))≠softmax(x)`,argmax 不变但概率值全错,而"必须输出概率分布"正是官方计分项)。**不走 calibrator**(`CALIBRATED_TASKS` 不含四头,温度缩放对未标定的头是伪精度)。**键缺失即 KeyError 不兜底** —— 四头是 `MultiTaskThyroidModel.forward` 的无条件输出,缺席说明加载的是不含四头结构的旧模型(如 `thyroidxl_merged_v2.pth`),静默兜底会写出确定性占位值而目标四 40% 三项 + 目标三 TI-RADS 全部失分且 JSON 完全合法、本地看不出。**长度也硬校验**(2026-08-17 第三轮 review lane4 L3):下游一律 `argmax+1`,长度错时若 argmax 落在合法值域内就静默产出错误编码 —— 实测唯独 `t_probs` 全程无人校验(4 维且 argmax=idx1 时静默写出 `T=1`),详见文末专条。
- **`t_conf` 也过 `_assert_finite`**:官方《初赛评审规则》「置信度字段规范」明写 `Detection.confidence`/`Malignancy.confidence`/`T_conf` 若为 NaN/Inf/越界,**评估脚本统一按 0 处理、该样本在 AP/AUC-PR/AUC 中排最后** —— 不是崩溃而是静默垫底,比崩溃阴。
- `calibrator`: 可选 `TemperatureScaler`。各分类头在 softmax/sigmoid **之前**先 `apply(task, logits)` 温度缩放;`None` 或全 T=1.0 时输出逐位等同无校准。各头统一 `.float()` 后再缩放(fp16 温度除法精度对齐),T>0 单调不改 argmax。mask 分割不走校准。
- **`*_prob` = 完整概率分布**(2026-07-29 新增)。四个 softmax 头和为 1;`foci_prob` 是**多标签 sigmoid**,各元素独立、不求和。`mal_probs` = `[p_benign, p_malignant]`,`mal_prob` 恒等于 `mal_probs[1]`(同源,不允许两条计算路径)。
- `*_idx` 取自概率向量的 argmax(softmax 严格单调,与对 logits 取 argmax 等价),保证索引与分布**自洽**——否则报告文字会和概率各说各话。
- **为什么必须有**:官方硬约束(赛前动员会 2026-07-28)"所有分类/分级/分型任务必须输出功率分布或置信度,**否则计0分**"。此前只取 argmax,概率在 decode 层就永久丢失(`cbm_head.py` 内部其实已算出,属导出遗漏而非能力缺失)。

### `build_probability_export(preds)` → `dict`（2026-07-29 新增）
把 `decode_predictions` 的概率字段整成 JSON 可序列化结构,是 `predict.py`/`video_predict.py` **共用的 schema 唯一真源**(新增维度只改这一处;`tests/test_probability_output.py::TestEntryPointsShareOneSchema` 用 AST 断言两个入口都调用它,禁止任一侧手写概率字段导致 schema 分叉)。
```json
{
  "concept_probabilities": {"composition": [...], "echogenicity": [...], "shape": [...], "margin": [...], "foci": [...]},
  "concept_predicted_index": {"composition": 3, "echogenicity": 2, "shape": 0, "margin": 1},
  "malignancy_probabilities": [0.13, 0.87]
}
```
类别顺序即索引顺序,与 `src/utils/report_generator.py` 的 `COMPOSITION_MAP`/`ECHOGENICITY_MAP`/`SHAPE_MAP`/`MARGIN_MAP`/`FOCI_MAP` 键严格对应,且与 `configs/thyroid_config.yaml` 的 `num_classes_*` 一致(两条契约都有回归测试)。numpy float32 必须 `.tolist()`,否则 `json.dumps` 直接抛异常。
> ⚠️ CBM 五概念头**未经语义监督**(`train_thyroidxl.py` 只训 malignancy + 分割),概率数值可用但语义映射未经验证——报告侧已加免责标注,8-01 官方概念标注重训后删除。

### `build_tirads_breakdown(comp_idx, echo_idx, shape_idx, margin_idx, foci_prob)` → `dict`
Delegates to `src/utils/report_generator.calculate_tirads_points()`.

---

## 4b. `calibration.py` — Post-hoc 置信度校准(温度缩放)

神经网络分类置信度普遍过度自信。温度缩放(Guo et al. 2017)post-hoc 修正:模型冻结,独立验证集学每任务标量 T,`softmax(logits/T)`。**不改 argmax/准确率**,只调概率刻度贴合真实通过率。

### `TemperatureScaler`
- `from_file(path)` → 读 JSON(`{"temperatures": {...}}` 或扁平 `{...}`);文件缺失/损坏/None → **全 T=1.0 降级**(不校准,输出不变),仅 WARNING 不抛异常。
- `apply(task, logits)` → `logits / T`(T=1.0 时返原张量,真恒等);softmax/sigmoid 前调用。
- `is_identity()` → 全任务 T=1.0 时 True(断言无行为变化)。
- 校准任务:`composition/echogenicity/shape/margin/foci/malignancy`(mask 不在内)。非法 T(≤0)自动降级 1.0。

### 离线拟合(真实数据 2026-08-01 到位后跑,现不调用)
- `fit_temperature(logits, labels)` → LBFGS 优化 `log_t`(保证 T>0)最小化温度后 NLL,学单标量 T。
- `fit_ordinal_temperature(logits, labels)` → 有序任务(TI-RADS 等级有序,跨级错代价更高):NLL + 0.1×期望等级平方距离惩罚。**当前为标量-T 折中骨架,真数据到位后替换为 cumulative-link**。

### 诊断指标
- `compute_ece(probs, labels, n_bins=15)` → Expected Calibration Error(分箱 `|conf-acc|` 加权);一维正类概率自动补两列。
- `reliability_diagram(probs, labels, save_path)` → 可靠性图(无 matplotlib 时跳过不抛)。

**接线**:`predict.py`/`video_predict.py` 从 `config["calibration"]["param_file"]`(默认 `checkpoints/calibration.json`)构造一次 calibrator 传入 `decode_predictions`;video 跨帧复用同一实例。文件不存在即静默降级,**当前部署无任何行为变化**。

---

## 5. `export.py` — Artifact Export

### `save_prediction_artifacts(output_dir, mask_resized, report_txt, out_json)`
- `mask_predicted.png`, `clinical_report.txt`, `result.json`.

### `save_rejection_json(output_dir, rejection_json)`
- `result.json` with rejection details.

### `_write_json_atomic(path, payload)`（2026-07-12,entrypoint.sh 崩溃兜底 Gate2 复审补强）
- `save_prediction_artifacts`/`save_rejection_json` 写 `result.json` 均改经此函数:先写 `<path>.tmp.<pid>` 临时文件,`json.dump` 完成后 `os.replace` 原子重命名。
- **动机**:`entrypoint.sh` 的子进程崩溃兜底逻辑只检查 `result.json` 是否**存在**,不校验内容是否完整。若 SIGTERM/OOM-kill 恰好在旧版直接 `open(...,"w") + json.dump` 写盘期间到达,磁盘上会留下半写/截断的 `result.json`——这类"存在但损坏"的文件比完全不存在更危险,会被误判为"本次已正常产出"。`os.replace` 在同目录内是原子操作(POSIX `rename(2)` / Windows `MoveFileEx`),不存在可被读到的中间态。

### `save_gradcam_overlay(model, img_tensor, img_np, output_path, branch="cls")`
- 双主干显式分支:`branch="cls"`(默认)→ `cls_backbone`(EfficientNet-B2,解释恶性分类);`branch="seg"` → `seg_backbone`(ConvNeXt-Nano,解释分割)。经 `_resolve_gradcam_target_layer(model, branch)` 取 `blocks[-1][-1]` 或末个 Conv2d。
- Generates heatmap overlay via `GradCAM` + `overlay_heatmap`.
- **由 `predict.py --enable_gradcam` 触发**。GradCAM 内部自带前向+反向,**必须在 `run_forward` 的 `inference_mode` 块之外**、用 `preprocess_image` 产出的普通张量调用(inference tensor 不能参与 autograd)。失败由 `save_gradcam_overlay` 内部兜底(`finally` 移除 hook),不阻断主流程。

### `maybe_save_gradcam(model, img_tensor, img_np, output_dir, branch="cls")` → `(gradcam_done, gradcam_path)`
- 收敛 `predict.py`(单图)与 `video_predict.py`(关键帧)**共用**的 Grad-CAM 落盘+存在性判定:在 `output_dir` 下生成 `gradcam_cls.png`,文件确实写出时 `gradcam_done=1`(供 `result.json` 的 `gradcam_generated`)。
- 报告附注文字(单图「恶性激活热图」/ cine「关键帧恶性激活热图」)语义不同,由各调用方在 `gradcam_done` 为真时自行追加,不放进本函数。

---

## 6. `case_pipeline.py` — 单病例诊断核心(CLI 与 service 共用)

**A0 重构新增**。把 `predict.py`(静态)/`video_predict.py`(动态)两个 `main()` 里
「输入加载完成 → 落盘之前」那段**纯推理核心**抽到此处,CLI 与 `src/service/` runner
共用同一份诊断逻辑,零行为变更。service 不能复用 CLI 的 `if __name__=="__main__"`
主函数,故只抽这一层,两端各自落盘(CLI 落 V2.0 `result.json`,service 落 V2.1
`prediction.json`)。

**边界:什么在这里,什么留在调用方(main/runner)**。本模块**无副作用、不落盘**——只做
纯核心:`preprocess → run_forward → decode → mask resize → tirads breakdown`(静态)+
逐帧诊断(坏帧隔离)/聚合/关键帧选择(动态)。留在调用方:**`build_model`/`TemperatureScaler`
构造**(有构建成本,service 要跨病例复用)、**输入加载**(`load_input_image`/
`load_video_frames`/`load_nifti_frames`,含 DICOM 内存 PHI 匿名化,2026-09-01 起不写盘)、**gatekeeper 审计**
(有副作用 + hard 一票否决 return 属调用方控制流)、**落盘与 Grad-CAM**。

### `diagnose_static_case(model, img_np, config, calibrator, precision, device)` → `dict`

单张静态图核心(输入已加载+已过 gatekeeper)。返回键:
- `preds`: `decode_predictions` 全量输出(含软 `pred_mask`/`mask_binary`、各头 `*_idx`/`*_prob`、`mal_prob`/`mal_probs`、**官方四头 `ete_probs`/`location_probs`/`t_probs`/`t_conf`/`tirads_probs`**);
- `img_tensor`: `preprocess_image` 产出的普通张量(供调用方在 inference_mode 之外重跑 Grad-CAM 复用);
- `mask_resized`: `mask_binary` 用 `INTER_NEAREST` 缩放回**原图 H×W**;
- `breakdown`: `build_tirads_breakdown` 的 TI-RADS 分项。

**Raises `NonFiniteProbabilityError`**:概率/mask 含 NaN/Inf 时 `decode_predictions` 硬拦,
冒泡给调用方转 ERROR(与原 `predict.main()` 的 except Exception 兜底一致)。

### `diagnose_dynamic_case(model, frames, frame_indices, config, calibrator, precision, device)` → `dict`

动态图(cine)核心(输入已抽帧+已过 gatekeeper)。返回键:`per_frame`/`agg`/`kf_preds`/
`kf_frame_np`/`kf_tensor`/`failed_frames`/`breakdown`/`mask_resized`。
- **坏帧隔离**(与原 `video_predict.main()` 逐字节一致):逐帧 `_diagnose_frame`,**只捕
  `NonFiniteProbabilityError`(概率/mask NaN/Inf)+ `AssertionError`(帧形状/维度非法)**
  丢该帧记入 `failed_frames` 后 continue;`RuntimeError`(CUDA OOM/权重维度不符)、
  `cv2.error` 等真故障**冒泡**由调用方转 ERROR。
- **关键帧张量防错位**:`kf_tensor`(GPU 张量)只保留**当前最恶性帧**那一个,显存从 N 帧
  降到 1 帧(防官方平台上界 OOM——实测 32G,见总纲 §3);判据用**严格 `>`**,与 `aggregate_frame_predictions`
  的 `max`(相等取首个)严格同序。`kf_img_tensor_pos != agg["keyframe_index"]` 一致性锁:
  两条 max 判据分叉(近乎不可能)时打 warning + 重算关键帧张量,不静默用错帧张量。
- `per_frame` 每帧存 `{frame_index, mal_prob, preds, frame_np}`(preds 含软 `pred_mask`,
  供 V2.1 逐帧 mask 落盘);`mask_resized` 为**关键帧** `mask_binary` 缩放回原图 H×W。

**Raises `DynamicCaseError(reason, diagnostic)`**:两种「无有效诊断」——**全帧数值失败**
(`if not per_frame:` 显式守卫,不靠 assert)与**无有效关键帧**
(`agg["keyframe_index"] < 0` 双重守卫,防 Python 负索引静默取末帧致错位)——携带诊断字段
(`n_frames_*`/`failed_*`)由调用方 catch 转 `save_rejection_json`。本层不落盘,把「怎么落」
的控制流交回 main/runner(与原 `video_predict.main()` 两处 `save_rejection_json`+`return`
逐字节等价)。

### 私有 `_diagnose_frame(model, frame_np, precision, device, img_size, calibrator=None, *, dynamic=False)`

单帧前向 → 解码,返回 `(preds, img_tensor)`。与原 `video_predict.diagnose_frame` 同义
(该函数保留为向后兼容 shim)。`dynamic` 透传 decode 选动态阈值 env(件套①);
唯一生产调用方 = kf 一致性锁重算分支(传 True,与动态主循环同阈值)。

---

## Usage Contract

| Entry Point | Uses | Does NOT Use |
|:---|:---|:---|
| `predict.py` | `io`, `gatekeeper_runner`, `model_runner`, `postprocess`, `export` | — |
| `video_predict.py` | `video_io`, `video_aggregate`, `gatekeeper_runner`, `model_runner`, `postprocess`, `export` | — |
| `evaluate.py` | `model_runner`(含 `run_forward`), `postprocess`, `export` | `gatekeeper_runner` (holdout = clean ground truth) |

**前向一致性**: evaluate.py 必须经 run_forward(model, img, precision, device) 调用前向，保证与生产数值路径一致。`evaluate.py` 提供 `--precision {fp32,fp16,bf16}`,评估生产指标时应与部署精度一致。

### `build_model()` 失败契约（2026-07-10 新增，交付入口必须遵守）

`build_model()` 现在会在 checkpoint 与模型结构不匹配时 `raise RuntimeError`（见上）。
`predict.py`/`video_predict.py` 作为交付容器入口，**必须** `try/except RuntimeError`
把这个异常兜成 `status="ERROR"` 的 `result.json`（复用 `export.save_rejection_json`），
不能让异常裸传播——否则该次提交零输出（比赛评分管线大概率直接判零分，比"预测错误"
更差）。`evaluate.py` 是开发者本地诊断脚本、非交付入口，不需要兜底，崩溃即可暴露问题。

**2026-07-10 晚，Gate2审查后契约扩容**：兜底范围从"只包 `build_model()`"扩大到
"整条 gatekeeper审计+推理+后处理+报告生成链路"，`except` 从 `RuntimeError` 放宽到
`Exception`。原因：`src/gatekeeper/authenticity.py` 同轮修复后权重缺失也会
`raise RuntimeError`（此前静默退化为随机分类，一票否决门形同虚设，见
`src/gatekeeper/AGENTS.md`），若不把 gatekeeper 审计纳入 try 包裹，`predict.py`
会在权重缺失时整进程崩溃、零输出——自相矛盾（越权威的失败越容易把容器打崩）。
`except Exception`（而非只 `RuntimeError`）是因为 `torch.load` 遇损坏 checkpoint
会抛 `pickle.UnpicklingError`（不是 `RuntimeError` 子类），`report_generator.py`
新增的 NaN/越界idx校验抛 `ValueError`，CUDA OOM 等也需兜住——异常类型名随 `str(e)`
一起落盘，不靠拆分 except 分支也能区分故障类别。`video_predict.py` 的 gatekeeper
审计循环本就在 try 块内，此次只需扩 `predict.py`。

### 配置加载失败契约（2026-07-11 修复，`model_runner.load_config`）

`predict.py`/`video_predict.py` 此前 `open(args.config)` 读 YAML →
`config["environment"]["device"]` 取值这几步在 `try` 块**之外**裸执行——
`--config` 路径不存在（`FileNotFoundError`）、YAML 格式错误
（`yaml.YAMLError`）、或键缺失（`KeyError`）都会导致进程裸崩溃、非零
exit code、零输出，与本节修复的"图像/视频加载失败"是同一类 try 外裸操作
问题，当时未触及（`os.makedirs(args.output_dir)` 本身仍留在 try 外——它一
旦失败，output_dir 都不存在，没地方写 result.json，只能非零 exit code 退
出，这是部署方控制的启动参数、非输入数据可触达的攻击面，风险可接受，不强
求覆盖）。

修复：新增 `model_runner.load_config(config_path)`，把 `open`→
`yaml.safe_load`→`config["environment"]["device"]`/`config["data"]["img_size"]`
键存在性校验都收进这一个函数，调用方（`predict.py`/`video_predict.py`）各自
用一层 `try/except Exception` 包裹调用，兜成 `status="ERROR"` 的
`result.json`（复用 `export.save_rejection_json`）。见
`tests/test_model_runner.py::TestLoadConfig`（单元级：合法配置/文件缺失/
格式错误/键缺失四态）+ `tests/test_smoke_entrypoints.py::TestConfigLoadFailureErrorFallback`
（子进程级：`predict.py`/`video_predict.py` 真跑 `--config` 缺失/格式错误）。


---

## 8. `nifti_io.py` — NIfTI 读写(2026-08-01 新增)

> **状态:已实现 + 已测试(35 例)+ 已接入。** A1 起 `entrypoint.sh` 按 `.nii.gz`
> 分发,`src/service/case_discovery.py`/`runner.py` 经本模块读写动态图输入与逐结节
> mask 落盘(V2.1 交付链路)。

### 为什么需要

2026-08-01 官方《赛事开发规范 V1.0》§五.2 + 《数据格式说明》两条硬约束:
- **写出**:每个预测结节的 mask **必须**为 NIfTI,`1`=结节 `0`=背景,H×W 与原图严格一致;
- **读入**:**动图原始数据是 `.nii.gz`**(静态图才是 DICOM),而 `video_io` 只认
  `.mp4/.avi/.mov/.mkv` + DICOM,遇 `.nii.gz` 返回 "Unsupported video format"。

### 维度约定(最易静默出错处)

NIfTI header 的 `dim` 是 **(X,Y,Z) 列主序**,numpy 图像是 **(H,W) 行主序**,前两轴
**互为转置**。漏转置时"本地写出→读回"用同一套错误约定**看不出任何异常**(转置两次
等于没转),但评测按 header 读形状即判定与原图不符 → 整例失分且本地永远复现不了。
故 `save_binary_mask` 统一 `(H,W)→(W,H,1)` / `(n,H,W)→(W,H,n)`,`load_nifti_frames`
反向还原。所有相关测试一律用**非方阵**(H≠W),方阵会让转置错误完全隐形。
🔴 **2026-09-01 第 5 次跑分破案**:官方 GT **静态 mask 也是 3D 带单帧**
(云端实测训练集 0030d3c5:Rows=726/Columns=1068 → GT=(1068,726,1))。此前 2D 路径
写纯 (W,H) → 官方 shape 校验「differs from every GT mask」**整批判零**。此约束
官方文档无记载(V2.1 只写"尺寸 H×W 一致"),且本地永测不出(自己写自己读抵消;
训练侧 loader 走 ndim==3 分支天然消化 GT 的 Z=1,训练侧从未暴露差异)。修复
commit 559b306:`binary.T[..., None]`。

### `save_binary_mask(mask, out_path, affine=None)` → `str`

- 接受 2D `(H,W)` 或 3D `(n_frames,H,W)`;**任意非零视作前景**(兼容内部 0/255 的
  `mask_binary`),内部统一二值化为 `{0,1}` uint8。🔴 **2D 输入也写出 3D
  (W,H,1)**(2026-09-01 起,官方 GT 静态 mask 为单帧 3D,见维度约定节)。
- **写 255 是静默零分路径**:文件完全合法,但评测按 `== 1` 取前景会得到全空 mask。
- 父目录自动创建(官方 `answer/{eval_id}/{StudyUid}/{SeriesUid}/` 是推理时动态建的)。
- `affine` 默认单位阵:超声无标准空间信息,评测按体素重叠度匹配主病灶,不依赖世界坐标。

### `load_nifti_frames(path, output_dir=None, max_frames=32)` → 5 元组

**签名与 `video_io.load_video_frames(path, output_dir, max_frames)` 逐位对齐**,入口层
可直接 `loader = load_nifti_frames if is_nifti_path(p) else load_video_frames` 后统一
传三参。`output_dir` 不被使用(NIfTI 无 PHI 元数据,无需落匿名化副本)但**必须占住第二
个位置** —— 否则统一调用会把 output_dir 字符串绑到 `max_frames`,
`uniform_sample_indices(n, "data/output")` 抛 TypeError,所有 `.nii.gz` 样本零预测,
且报错与"参数错位"毫无关联极难定位。`tests/test_nifti_io.py::TestLoaderSignatureAlignment`
用 `inspect.signature` 锁死这一对齐。

⚠️ **max_frames 的调用方分歧(2026-09-08)**:本函数签名缺省仍是 32,但**service
生产链(`runner._DEFAULT_MAX_FRAMES`)已改 0=全帧**——检测诊断定案:GT 标注仅 3-5 帧,
32 帧抽样命中 0.2-0.5 帧,官方"漏帧 IoU 记 0"→动态病灶级 IoU 天花板 <0.5 必死
(TP=0 全灭)。CLI `video_predict.py` 与本函数缺省保留 32 属**调试旁路旧口径**,
勿以其推断生产行为。`AICOMP_MAX_FRAMES=32` 可回退(勿用,动态检测回全灭)。

返回 `(frames, frame_indices, is_dicom=False, error_msg, n_total_frames)`,错误契约与
`video_io` 一致(失败返回错误信息而非抛异常,`n_total_frames` 失败分支为 0)。`frame_indices`
是 **0-based 原始帧号**;官方 xlsx 的 `ANNOTATION->SliceIndex` 是 **1-based**,跨界需 ±1。
`n_total_frames` 为原始体数据 Z 轴长度(非抽样数),供 `scatter_frames_to_full_length`
把抽样帧 mask 回填成全帧序列 3D mask(规范 §五.2)。`video_io.load_video_frames` 同返回
5 元组,失败 `n_total_frames=0`。

### `scatter_frames_to_full_length(frame_masks, frame_indices, n_total_frames)` → `(n,H,W)`

规范 §五.2:动态 mask 的 Z 轴必须对应**全视频帧序列**(必须预测并输出所有帧)。而
loader 默认只抽 32 帧 —— 直接堆叠抽样帧的 mask 会让 Z 长度与原视频不符:评测按
`SliceIndex` 取标注帧会取到**错误切片**,或直接判尺寸不一致。**两种失败都不报错、
产物完全合法**,是最隐蔽的零分路径。本函数按 `frame_indices` 把抽样帧回填到原始帧位,
未抽中帧填 0(规范另有"仅抽取标注帧计算指标"兜底,故未抽中帧填 0 的代价仅限该帧恰为
标注帧的情形;传 `max_frames<=0` 取全帧可彻底规避)。

---

## 9. `nodule_split.py` — 整图 mask → 逐结节实例(2026-08-01 新增)

> **状态:已实现 + 已测试(27 例)+ 已接入。** `v21_output.write_case_v21` 在 V2.1
> 交付链路中调用 `split_nodules` 把整图 mask 拆成逐结节实例。

官方要求**每个检出结节**单独一条记录,而本项目模型输出的是**单通道整图 mask**
(`multi_task_model.seg_head` 是 `Conv2d(20,1,1)`,无实例分割概念)。本模块用连通域
分析补这一层。

### 三个设计决策

1. **在原图尺寸上拆,不在 512×512 模型分辨率上拆** —— 先拆再逐实例 resize 会让每个
   实例各经历一次最近邻插值,相邻结节边界还原后可能重叠或产生缝隙,累计误差不可控。
2. **面积阈值用相对比例**(`min_area_ratio` 默认 None→env)而非绝对像素 —— 超声图 400×300 到
   1024×768 不等,固定像素阈值在小图误杀真结节、大图放过噪声。
   **三层优先级(2026-09-10 件套④)**:显式传参 > `AICOMP_MIN_AREA_RATIO` env(None 默认时)
   > `DEFAULT_MIN_AREA_RATIO=1e-4`(env 未设/垃圾/负/inf/nan 一律回默认,inf 会经
   `int(ratio*h*w)` 抛 OverflowError 炸整例故必须挡)。**只有静态生产调用点靠默认参吃 env**
   (`v21_output._write_static` 不传);动态生产/镜像三处**显式传 DEFAULT 钉死不走 env**
   (动态放宽 = 增多 3D 实例,写盘耗时与内存都是 8h 时限下的风险面)。既有测试的显式
   `min_area_ratio=0`(不过滤)用法不受 env 影响。
3. **3D 按全帧累计体素过滤,不按单帧峰值** —— 动态图噪声多为单帧瞬时斑点,真结节必然
   跨帧持续;按单帧阈值会误杀"每帧 20px、连续 10 帧"的真结节(单帧 20 < 26,累计 200)。
   代价是单帧大面积斑点被保留,但那通常正是"真结节只在关键帧清晰"的正常情形。

### 3D 时间维缝合的边界陷阱(已修复,有回归锁)

`scipy.ndimage.binary_closing` 的**腐蚀步骤默认 `border_value=0`**(数组外视作背景),
会把**贴着首帧/末帧边界**的前景整个吃掉。实测:(帧0,帧2) 有、帧1 漏检的结节,默认
closing 后变成 (帧1,帧2) —— **帧 0 凭空消失、帧 1 凭空出现**。结节出现在视频首/末帧
极其常见,该行为等于系统性丢帧,且产出 mask 完全合法看不出异常。故显式拆成
`binary_dilation` + `binary_erosion(border_value=1)`。回归锁:
`tests/test_nodule_split.py::TestFrameGapClosing::test_gap_at_first_frame_survives`。

缝合结果**只用于确定标签归属**,最终实例 mask 仍 `& binary` 回落原始数据,不把补出来
的虚构帧写进交付 mask。

### `detection_confidence` 用均值不用最大值

最大值几乎恒为 1.0(任何结节内部都有高置信像素),失去区分度;而规范的动态 AP 排序
依赖 confidence 的**相对次序** —— 全 1.0 等于随机排序。

---

## 10. `prediction_json.py` — 官方输出 schema(2026-08-01 新增)

> **状态:已实现 + 已测试(72 例)+ 已接入。** A3 起 `v21_output.write_case_v21` 经
> `build_prediction_json`/`validate_prediction_json` 组装并校验 V2.1 输出。

### 三个新维度的类别映射(最需小心处)

《数据格式说明》给的是 **xlsx 训练标签的类别**,与**输出编码**并非一一对应,两处类别数
都不同 —— 直接把训练标签的 argmax 当输出值填会**系统性错位**:

| 维度 | xlsx 训练标签 | 输出编码 | 差异 |
|:--|:--|:--|:--|
| `extrathy_ext` | 包膜/软组织/血管/无/无病灶(阴性)/NA-UNK (**6类**) | `1`无 `2`包膜 `3`软组织 `4`血管 (**4值**) | 后两类须归并到 `1` |
| `lesion_loc` | 左上/左下/左中/右上/右下/右中/峡部/无(阴性)/NA-UNK (**9类**) | `1..7` + `LocationProbs[7]` | 后两类不进分布 |
| `clin_t_stage` | 有/无 (2类) | `T`=1/0 | 直接对应 |

`LOCATION_LABELS` 是**单一权威定义**,顺序取自官方文档列举次序。训练侧构造标签必须
`import` 该常量,**不得另抄一份** —— 顺序写错会让概率分布与类别标签系统性错位,而
JSON 完全合法、指标只是"偏低",本地永远查不出根因。

`location_label_to_index` 对阴性/未知返回 **`None` 而非 0** —— 把未知样本当"左上叶"
训练会污染整个位置分布。

### 🟢 四头已接真实预测(#116b,2026-08-17);占位路径仅剩 CLI 兜底

**生产路径无占位**:`v21_output._shared_nodule_context` 把四头真实 softmax 转成官方编码,
逐字段显式传给 `build_nodule_entry`。云端训练实测能力:`loc_auc` 0.7519 /
`t_auc` 0.7441 / `tirads_kappa` 0.6887 / `ete_kappa` 0.2086。

🔴 **编码换算 `code = 决策(idx) + 1` 只对 ETE / TI-RADS / Location 成立;
其中 ETE / TI-RADS 的决策自 2026-09-10 件套③起 = `_qwk_round`(期望取整
`int(round(Σ j·p_j))`,二次加权 Kappa 最优决策)而非 argmax;**Location 仍是
argmax**(宏 AUC 吃分布);`T_ClinSig.T` 唯一不 +1**(官方 `T∈{0,1}` 与 0-based
下标同值)。
⚠️ `_qwk_round` 的 `round` 是 Python **银行家舍入**(半整数取向偶数:E=0.5→码1,
`floor(x+0.5)` 会得码 2),复刻须用同款 `int(round(...))`;半整数处两类等距、
Kappa 损失相同,属纯 tie-break 无系统性偏差。训练侧 `metrics_official` 仍是
argmax 口径,两侧决策规则已脱节(训练关账,仅诊断性影响)。
写错 `+1` 的后果不对称:T 多加会得 2 被 `prediction_json` 拦成整例 ERROR(至少会炸);
其余三项漏加只是变成**另一个合法码**(如"包膜侵犯"报成"无侵犯"),越界校验放行、
JSON 合法、指标偏低,**本地永远查不出**。回归测试 `TestOfficialHeadWiring` 用
**全部避开 idx0** 的 one-hot 参数化钉死(用 idx0 测等于没测:漏 `+1` 时 idx0 恰好也"对");
另有非尖锐分布分叉用例(argmax 与期望取整给出不同码)+ 半整数 tie 用例钉死新规则。

**`build_nodule_entry` 的占位兜底保留不删**(ETE=1/Location 均匀/T=0),供 CLI 调试等
不产四头概率的调用方使用。→ 🔴 **在生产 `prediction.json` 里看到 `ETE=1 / 均匀
LocationProbs / T=0 / T_conf=0.0` 这组值,是"四头没接上"的故障信号,不是"占位正常"**。
最可靠单条判据:**`LocationProbs` 是均匀分布**(真实预测几乎不可能均匀,与权重质量无关)。

**#116a 对冲键**:`build_nodule_entry` 可选参数 `tirads_probs`(5维)/`ete_probs`(4维),
写出 `TI_RADS_Probs`/`ETE_Probs`;`None` 时**不写该键**(旧调用方零污染)。长度错即抛
`PredictionSchemaError`(长度错 = 上游取错了头)。⚠️ 评审规则 `:251` 正面强制「概率向量
键名必须与下表完全一致」而表里这两项只有 int → **自拟键名必然对不上,拿分概率≈0,
纯防御性下注**;净收益仍 ≥0(赛道七无"不得含额外字段"条款——该条款在**《初赛评审规则简要说明》** `:540`「其他Finding类型不得包含上述属性字段」,位于**赛道八**章节 `:390` 起,不适用赛道七;⚠️ 别把它记成开发规范里的条款,那份文件全文 516 行、无赛道八章节)。

**动态走病灶级均值**:`_lesion_level_head_preds` 对 `per_frame` 全帧的**四头 + 良恶性**
概率取**算术平均**(官方 `:215`「病灶级汇总,每个预测病灶仅贡献一个样本」——该句把
「良恶性AUC-PR」与四头**并列**列入,不是只管四头),**与训练侧
`metrics_official.aggregate_by_lesion` 同口径** —— 后者 docstring 明写「若推理侧改用
取最大/中位数,这里要同步改,否则验证指标与交付表现脱节」,**两侧算法互为约束**。
🔴 `mal_prob` 曾被留在关键帧口径造成分裂(2026-08-17 终审 P0.1 修正):关键帧是
「最恶性帧」(argmax),**系统性高于**病灶级均值,偏差直接打在良恶性 AUC-PR 上。
`breakdown`(TI-RADS 概念分项)仍用关键帧——它只喂 CBM 报告文本、初赛不计分。

🔴 **动态 `split_nodules` 的缝合跨度必须按抽样间隔自适应**(2026-08-17 终审 D 路 P1):
`max_frames=32` 抽样 + scatter 回填后**未抽中帧全 0**,帧间隔 = `ceil(n_total/抽样数)`。
`close_gap_frames` 默认 **1**,小于该间隔时 26-邻域在时间维**连不上**,同一结节被切成
一串碎片。实测(单个全程存在的结节):50 帧→1 个 ✅ / **100 帧→7 个** /
**128·200·400 帧→各 20 个**(撞 `DEFAULT_MAX_NODULES` 上限)。真实 cine 普遍上百帧,
故这是**常态而非边缘情况**;每个碎片都带一份四头预测进 `Nodules`,官方按「每病灶一个
样本」计分 → 重复计数 + 检测 AP 被稀释。现取 `间隔+1`。
⚠️ 副作用已实测排除:闭运算是 **3D** 的,时空分离的两个结节即使 gap 调到 20 仍保持
2 个实例(空间不重叠连不上)。回归钉子 `test_sampled_frames_do_not_fragment_one_nodule`
+ `test_temporally_separate_nodules_not_merged`(已做变异测试:回退修复后两条都变红)。

🔴 **`n_total_frames` 缺失是硬失败,不得回退猜测**(2026-08-17 第三轮 review lane3 P2-1):
原实现回退 `max(frame_indices)+1`,**尾部若有未抽中帧则 Z 长度短于原视频** ——
评审规则 `:205` 的罚则是「格式错误,该病例所有指标不予计分」(**整例清零**),而回退
只打一条 warning、输出看着完全正常,本地永远发现不了。现改为抛 `V21OutputError`:
宁可该病例 ERROR(日志可见)也不交一份注定被判格式错误的 mask。
生产路径不可达(`runner.py:329` 恒注入,且 loader 失败时返回 0 → `runner.py:283` 提前
raise),拦的是「未来新增调用方忘了注入」这类回归。钉子
`test_missing_n_total_frames_raises_not_guesses`(原 `..._falls_back`,其 fixture 恰好
演示旧行为的危害:6 帧原视频只写 5 帧 Z 轴)+ `test_invalid_n_total_frames_raises`(0/负/None)。

🔴 **四头概率的长度契约在 `decode_predictions` 源头硬校验**(同轮 lane4 L3):
下游 `_shared_nodule_context` 对 ETE/TI-RADS 走 `_qwk_round`、Location 走
`argmax+1`(件套③后仍以 idx 为基),**长度错时若决策恰好落在合法值域内
就静默产出错误编码** —— 该维度按错值计分(Kappa/AUC 污染),JSON 完全合法查不出来。
实测 8 个长度错场景:ETE/TI-RADS/Location 各自被下游对冲键或 `LocationProbs` 的长度
校验**侧面**拦下(**那是巧合不是设计**),**唯独 `t_probs` 全程无人校验** —— 4 维且
argmax=idx1 时一路静默写出 `T=1`。故不依赖下游偶然拦截,在源头按类别数统一校验。
⚠️ 期望值常量从 `src.models` 单一权威 import(`NUM_CLIN_T_CLASSES` 在
`heads/clinical_t.py`,其余在 `label_schema.py`),**不得在推理侧另抄** —— 抄一份则
模型改类别数时两处分叉,校验会把正确输出判成错误或放行真错误。依赖方向合规:
`.importlinter` 禁的是 `src.models` 依赖 `src.inference`,反向可以。
钉子 `test_official_head_wrong_length_raises`(四头 × ±1 双向参数化,双向防「只校验了
下界」)+ `test_official_head_length_constants_come_from_models`。

### 两条不显眼但致命的校验

- **`Malignancy.predicted` 必须是 Python `bool`** —— `numpy.bool_` 不是 `bool`,
  `json.dump` 直接抛 TypeError ⇒ **整份 prediction.json 写不出去**,比任何单字段错更严重。
- **`LocationProbs` 和必须严格为 1** —— 规范明写"顺序固定且和为 1",fp16→fp32→float
  往返会留 1e-7 量级残差,严格解析器会拒。`normalize_location_probs` 把残差并入最大项
  (相对扰动最小);`uniform_location_probs` 并入**首位**(并入末位会让占位 argmax 恒落
  第 7 类"峡部" —— 解剖上体积最小、结节发生率最低的分区,是更糟的占位偏差)。

---

## 11. Phase 5 接入待办(2026-08-01 记,2026-08-10 A1-A3 已闭合)

> **已闭合**:下列缺口在 Phase 5 / 本轮 commit 已全部落地(详见 §12 与各模块状态)。
> 保留本表作为接入历史与「还剩什么」的说明。

| 缺口 | 现状 | 处置 |
|:---|:---|:---|
| 入口不认 `.nii.gz` | `entrypoint.sh` 按扩展名分发,`${INPUT_PATH##*.}` 对 `.nii.gz` 剥出 `gz` → 落 `*)` | **A1 已闭合**:`.nii.gz`/`.nii` 全名匹配单独判定归 dynamic,魔数嗅探显式跳过 NIfTI |
| 部署形态未切换 | `entrypoint.sh` 仍是"跑一次推理就 exit"的离线 CLI | **A2 已闭合**:Dockerfile `ENTRYPOINT` 切 `python -m src.service.app`(uvicorn 8000),entrypoint.sh 降级为调试旁路 |
| 推理未接入服务层 | `src/service/app.py` 的 `set_runner()` 无调用方,`_runner` 恒为 None | **A2 已闭合**:`build_service_runner` 落地,`main()` 注入真实 runner(见 `src/service/AGENTS.md`) |
| 输出契约不相交 | V2.0 `result.json` + `mask_predicted.png`;官方要 `prediction.json` + 每结节 `.nii.gz` | **A3 已闭合**:`v21_output.write_case_v21` 唯一写盘入口,`prediction_json` 校验 |
| 模型仍是整图级 | 分类头无实例分割 | **@云TODO 存续**:分类头仍整图级,多结节属性靠 `nodule_split` 事后拆分,逐结节属性无法区分(v21_output 已接 `nodule_split`) |

## 12. `v21_output.py` — V2.1 单病例输出写盘入口(A3 新增)

**唯一 V2.1 写盘入口**(`write_case_v21`),把 `case_pipeline` 的推理结果转成官方
《赛事开发规范 V2.1》§五的逐结节交付形态:

- **mask 尺寸硬约束**:软 `pred_mask` 与二值 `mask_binary` 都 resize 回**原图 H×W**(软用
  `INTER_LINEAR` 保概率平滑,二值用 `INTER_NEAREST` 保 `{0,1}` 语义);调用方(runner)透传
  `orig_hw`,缺 `orig_hw` 直接 `V21OutputError`——绝不拿 resize 后的模型输入张量冒充原图。
- **逐结节**:`nodule_split.split_nodules` 连通域拆分(2D 静态 / 3D 动态跨帧),每结节
  一个 `.nii.gz`(`nodule_NNN.nii.gz`,值域 `{0,1}`);检测置信度:静态 `detection_confidence`
  (区域内概率均值),**动态 2026-09-08 起走 `_dynamic_det_conf`**(实例出现帧的逐帧 float
  概率均值——旧口径在 scatter 后 uint8 上算,语义=帧覆盖率恒≈1.0 无排序能力,检测诊断
  报告实锤动态 conf 挤 ~1.0,已废弃;连带 `_write_dynamic` 不再分配 soft_3d 死存储)。
- 每结节 `build_nodule_entry`(良恶性/TI-RADS/ETE/位置/临床T **全部为四头真实预测**,#116b
  2026-08-17 接线;另写 `TI_RADS_Probs`/`ETE_Probs` 对冲键)→
  `build_prediction_json` → `validate_prediction_json` → **原子写**(`.tmp` + `os.replace`)。
- **阴性病例合法**(`split_nodules` 0 结节):`Nodules`/`MaskFiles` 空数组,仍写结构合规
  `prediction.json`。
- **两个本地测不出的坑(写死在本模块)**:① NIfTI 列主序转置由 `save_binary_mask` 内部做,
  本模块只传行主序数组,不重复转置;② `MaskFile` 必须是相对病例一级目录的**正斜杠**路径
  (`{subdir_uid}/nodule_NNN.nii.gz`,禁绝对路径/`..`,`_validate_relative_mask_path` 双侧拦截),
  Windows 上 `os.path.join` 产 `\\`,官方 Linux 评测按 `/` 解析会定位失败。
- **动态 3D 跨帧关联(@云TODO 最简实现)**:抽样帧逐帧 resize → `scatter_frames_to_full_length`
  回填全帧 `(n_total, H, W)` 3D mask → 3D 连通域 = 跨帧同一结节自然连成一个实例;分割中间帧
  漏检会拆成两个实例,空间相邻结节可能被闭运算粘连——真正的 tracking 待官方数据校正。
- **`_build_entries_and_save_masks(...)` 只返回 `entries`**(2026-08-10 nit:原返回
  `(entries, rel_paths)`,`rel_paths` 无其它用途,两处调用点原都写 `entries, _ =` 丢弃,已简化)。

### 失败语义

`V21OutputError`(独立于 `PredictionSchemaError`):case_result 缺关键键 / 维度不符 / 依赖缺失
(cv2)等运行期故障,由 runner 的病例级隔离捕获转 ERROR 占位,不拖垮整批。`rejected` 标记的
case_result 不应走本函数(由 runner 的 ERROR/REJECTED 兜底)。
