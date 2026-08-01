# src/inference/AGENTS.md - Shared Inference Pipeline Specifications

This package provides reusable inference modules shared by `predict.py` (production) and `evaluate.py` (holdout benchmarking).

---

## Package Structure

| File | Responsibility |
|:---|:---|
| `io.py` | Image loading (DICOM + PNG/JPG) and PHI anonymization;通道/位深归一化工具 `_to_gray`/`_rescale_to_uint8` 下沉于此(静/动两路径共用,video_io re-import)。`_to_gray` 覆盖全部通道形态((H,W)/(H,W,1)/(H,W,3)/RGBA/channel-first/多光谱 均收敛 2D,杜绝 3D 漏过致下游 `preprocess_image` 5 维崩溃),并以 `channel_order` 区分 RGB(DICOM 像素)与 BGR(OpenCV 视频帧)避免多普勒伪彩加权错位。单图 DICOM 经 `_to_gray` 收敛灰度 |
| `gatekeeper_runner.py` | 4-layer gatekeeper audit pipeline (predict.py 单图 + video_predict.py 动态图)。**2026-07-29 新增 `collect_soft_flags()`**:把 `CatNetTamperingDetector.last_soft_flags` 归一化成可 JSON 序列化结构落进 audit——此前生产代码零读取该属性,仅 copy_move/splice/BAG 命中的图 `flagged` 为 False、result.json 完全沉默,而 copy_move 在 359 张真实图上误报 27.3%,信号全丢 |
| `model_runner.py` | Model build, image preprocessing, forward pass |
| `postprocess.py` | Prediction decoding and TI-RADS breakdown(可选接 `calibration.TemperatureScaler` 做温度缩放) |
| `calibration.py` | Post-hoc 置信度校准(温度缩放):`TemperatureScaler`(每任务标量 T,缺参数即 T=1.0 降级)+ 离线拟合 `fit_temperature`/`fit_ordinal_temperature` + 诊断 `compute_ece`/`reliability_diagram`。真实数据 2026-08-01 初赛才到,现仅骨架 |
| `export.py` | Artifact saving (mask, report, JSON, Grad-CAM overlay) |
| `video_io.py` | 动态图/视频(cine)抽帧:MP4/AVI/MOV/MKV via OpenCV + 多帧 DICOM cine via pydicom;复用 `io._PHI_FIELDS` 脱敏,经 `_to_gray` 统一灰度兜底(含彩色/多普勒 cine);OpenCV 视频帧传 `channel_order="bgr"`、DICOM cine 用默认 `"rgb"`。供 `video_predict.py`(根目录) |
| `video_aggregate.py` | 视频推理纯逻辑:`uniform_sample_indices`(均匀抽样下标)+ `aggregate_frame_predictions`(帧级聚合,恶性概率取**最大帧**做最终判定,召回优先;**空输入安全返回 `keyframe_index=-1` 不抛**,由 `video_predict.py` 的显式守卫拦全帧失败,见下方「坏帧隔离」) |
| `nifti_io.py` | **2026-08-01 新增,尚未接入主流程**。NIfTI(.nii/.nii.gz)读写:`save_binary_mask`(交付 mask,值域强制 `{0,1}`)、`load_nifti_frames`(动图输入,签名与 `load_video_frames` 逐位对齐)、`scatter_frames_to_full_length`(抽样帧回填全帧序列)、`is_nifti_path`。见 §8 |
| `nodule_split.py` | **2026-08-01 新增,尚未接入主流程**。整图 mask → 逐结节实例:`split_nodules`(2D/3D 连通域)、`crop_nodule`(带上下文边距裁剪)、`detection_confidence`(实例区域概率均值)。见 §9 |
| `prediction_json.py` | **2026-08-01 新增,尚未接入主流程**。官方 `prediction.json` schema 构造与校验:`build_nodule_entry`/`build_prediction_json`/`validate_prediction_json` + 三个新维度的类别映射常量。见 §10 |

---

## 1. `io.py` — Image Loading & DICOM Anonymization

### `load_input_image(input_path, output_dir)` → `(img_np, is_dicom, error_msg)`

- Supports DICOM (`.dcm`/`.dicom`) via `pydicom` and standard formats via `cv2.imread(..., IMREAD_GRAYSCALE)`.
- **PHI Anonymization** (HIPAA Safe Harbor): 仅覆盖 DICOM 中**已存在**的 PHI 标签(完整清单见 `io.py` 的 `_PHI_FIELDS`),涵盖患者身份(Name/ID/BirthDate/Sex/Age/Address/Telephone/OtherIDs)、就诊(AccessionNumber/StudyID/各类 Date)、机构与设备(InstitutionName/Address/StationName/DeviceSerialNumber)、医师与操作者(Referring/Performing/Operators)。日期类置 `"00000101"`,其余置 `"ANONYMIZED"` 或空串。
- Anonymized DICOM saved to `output_dir/anonymized_image.dcm`.
- Returns `(None, False, error_msg)` on failure — caller must check `img_np is None`,
  print the error **并调用 `export.save_rejection_json` 落 `status="ERROR"` 的
  `result.json`**（2026-07-11 修复：此前只 print 不落盘，容器 exit code 仍为 0 但
  `output_dir` 是空目录——比 gatekeeper 一票否决/ `except Exception` 兜底的 ERROR
  状态还差，绕过了下游所有安全网）。`predict.py`/`video_predict.py` 的
  `load_video_frames` 加载失败分支同样遵守此契约。

---

## 2. `gatekeeper_runner.py` — 4-Layer Audit

### `run_gatekeeper_audit(img_np, config, device, weights_path, output_dir, deduplicator=None, auth_module=None, tamper_detector=None)` → `(passed: bool, audit: dict)`

**Scope**: Production inference only. evaluate.py 仅导入 model_runner/postprocess/export 三模块；gatekeeper_runner 由 predict.py(单图) 和 video_predict.py(动态图) 使用。

**`audit` 恒为 dict**(2026-07-29 契约变更;旧版通过时返回 `None`)——审计结论必须能落进 `result.json`,否则 phantom/synthetic 软 flag 只打印在 stdout、评测方拿不到。任何调用点都不得再假设"通过 ⇒ 第二个返回值是 None"。

### `resolve_enforcement_mode(config)` → `"soft" | "hard"`(2026-07-29 新增)
读 `config["gatekeeper"]["enforcement_mode"]`,归一化(strip+lower),缺省/非法值一律降级 `"soft"` 并打 `[WARNING]`。
**为什么默认 soft**:官方赛前动员会(2026-07-28)明确赛道七**豁免全部医保安全考核**,总分 = 临床诊断得分 Sclin × 100,gatekeeper **零计分贡献**;且"真实病例被误判为伪造,诊断结果仍参与目标三四五计分"——前提是必须先输出诊断。故硬拒绝已成**纯负期望**:拒对了零收益,误拒一次即白丢该病例全部临床得分。**失败方向必须偏向"多输出一份诊断"而非"多拒一张真实图"**,所以拼写错误(如 `"Hard_typo"`)降级 soft 而不是报错或按 hard 处理。
- `soft`(默认、交付固定值):去重/篡改命中只落 flag,`passed` 恒为 `True`,诊断照常输出;
- `hard`:恢复"命中即 REJECTED、不输出任何诊断"的旧语义,仅供回归测试与未来规则变化时启用。

- 三个可选预建对象(视频逐帧复用,避免每帧重复加载权重/播种训练集):
  - `deduplicator`: 用 `build_seeded_deduplicator(config, device)` 预建；
  - `auth_module`: 用 `build_auth_module(weights_path)` 预建(单图路径懒建,跳过 `authenticity_model.pth` 每帧重复加载)；
  - `tamper_detector`: 用 `build_tamper_detector(config)` 预建(避免每帧重探 catnet 权重文件)。
- 三者均为 None 时内部即时构建——单图 predict.py 路径行为不变(向后兼容)。
- **无状态污染**:`is_duplicate` 只读比对,不把当前帧 `add_image` 入库;故同一 cine 相邻近重复帧之间不会互相误判(去重库只播种训练集)。

Pipeline:
1. **Authenticity** — `AuthenticityModule` (13-dim GLCM+freq+ELA+DND+TAI → MLP) → phantom/synthetic detection。**2026-07-12 统一软flag制度**:分类结果**不再**驱动 REJECTED,仅作审计字段(`is_phantom`/`is_gan`)。权重缺失/损坏仍在 `AuthenticityModule.load_weights` 硬报错,红线未变——**审计命中软化 ≠ 故障软化**,文件损坏/权重缺失/解码失败一律仍走 ERROR。详见 `src/gatekeeper/AGENTS.md` §1。
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
- **`frame_np` 存 per_frame 元素,`img_tensor` 改为只留当前最恶性帧**(2026-07-30 显存优化):32 帧(或不限帧 DICOM cine)的 GPU 张量全量常驻会在官方平台 32G 显存硬上界(rules.md §2 Q33)上叠加模型+Grad-CAM 而 OOM,本地 A800 80G 测不出。逐帧用严格 `>` 刷新 `kf_tensor`(与 `aggregate_frame_predictions` 的 `max(range(n),key=...)` 取首个最大严格同序),显存常驻从 32 帧降到 1 帧;关键帧处有 `kf_img_tensor_pos != agg["keyframe_index"]` 一致性锁(不等则 warning + 重算)。**错位陷阱(2026-07-29 实测复现)**:旧版 `key_tensors` 平行 dict 以全帧下标为键、`keyframe_index` 是 per_frame 下标,跳帧后两空间错位且 `key_tensors[kf_pos]` 恒取得到值、不抛 KeyError,mask/Grad-CAM 静默出自错帧;`frame_np` 存进 per_frame 元素后下标空间只剩一个。
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

### `decode_predictions(outputs, calibrator=None)` → `dict`
Returns keys: `pred_mask`, `mask_binary`, `comp_idx`, `echo_idx`, `shape_idx`, `margin_idx`, `comp_prob`, `echo_prob`, `shape_prob`, `margin_prob`, `foci_prob`, `mal_prob`, `mal_probs`。
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

> **状态:已实现 + 已测试(35 例),但主流程尚未调用。** `predict.py`/`video_predict.py`/
> `entrypoint.sh` 三处入口目前都不认 `.nii.gz`,接入是独立待办项(见 §11)。

### 为什么需要

2026-08-01 官方《赛事开发规范 V1.0》§五.2 + 《数据格式说明》两条硬约束:
- **写出**:每个预测结节的 mask **必须**为 NIfTI,`1`=结节 `0`=背景,H×W 与原图严格一致;
- **读入**:**动图原始数据是 `.nii.gz`**(静态图才是 DICOM),而 `video_io` 只认
  `.mp4/.avi/.mov/.mkv` + DICOM,遇 `.nii.gz` 返回 "Unsupported video format"。

### 维度约定(最易静默出错处)

NIfTI header 的 `dim` 是 **(X,Y,Z) 列主序**,numpy 图像是 **(H,W) 行主序**,前两轴
**互为转置**。漏转置时"本地写出→读回"用同一套错误约定**看不出任何异常**(转置两次
等于没转),但评测按 header 读形状即判定与原图不符 → 整例失分且本地永远复现不了。
故 `save_binary_mask` 统一 `(H,W)→(W,H)` / `(n,H,W)→(W,H,n)`,`load_nifti_frames`
反向还原。所有相关测试一律用**非方阵**(H≠W),方阵会让转置错误完全隐形。

### `save_binary_mask(mask, out_path, affine=None)` → `str`

- 接受 2D `(H,W)` 或 3D `(n_frames,H,W)`;**任意非零视作前景**(兼容内部 0/255 的
  `mask_binary`),内部统一二值化为 `{0,1}` uint8。
- **写 255 是静默零分路径**:文件完全合法,但评测按 `== 1` 取前景会得到全空 mask。
- 父目录自动创建(官方 `answer/{eval_id}/{StudyUid}/{SeriesUid}/` 是推理时动态建的)。
- `affine` 默认单位阵:超声无标准空间信息,评测按体素重叠度匹配主病灶,不依赖世界坐标。

### `load_nifti_frames(path, output_dir=None, max_frames=32)` → 4 元组

**签名与 `video_io.load_video_frames(path, output_dir, max_frames)` 逐位对齐**,入口层
可直接 `loader = load_nifti_frames if is_nifti_path(p) else load_video_frames` 后统一
传三参。`output_dir` 不被使用(NIfTI 无 PHI 元数据,无需落匿名化副本)但**必须占住第二
个位置** —— 否则统一调用会把 output_dir 字符串绑到 `max_frames`,
`uniform_sample_indices(n, "data/output")` 抛 TypeError,所有 `.nii.gz` 样本零预测,
且报错与"参数错位"毫无关联极难定位。`tests/test_nifti_io.py::TestLoaderSignatureAlignment`
用 `inspect.signature` 锁死这一对齐。

返回 `(frames, frame_indices, is_dicom=False, error_msg)`,错误契约与 `video_io` 一致
(失败返回错误信息而非抛异常)。`frame_indices` 是 **0-based 原始帧号**;官方 xlsx 的
`ANNOTATION->SliceIndex` 是 **1-based**,跨界需 ±1。

### `scatter_frames_to_full_length(frame_masks, frame_indices, n_total_frames)` → `(n,H,W)`

规范 §五.2:动态 mask 的 Z 轴必须对应**全视频帧序列**(必须预测并输出所有帧)。而
loader 默认只抽 32 帧 —— 直接堆叠抽样帧的 mask 会让 Z 长度与原视频不符:评测按
`SliceIndex` 取标注帧会取到**错误切片**,或直接判尺寸不一致。**两种失败都不报错、
产物完全合法**,是最隐蔽的零分路径。本函数按 `frame_indices` 把抽样帧回填到原始帧位,
未抽中帧填 0(规范另有"仅抽取标注帧计算指标"兜底,故未抽中帧填 0 的代价仅限该帧恰为
标注帧的情形;传 `max_frames<=0` 取全帧可彻底规避)。

---

## 9. `nodule_split.py` — 整图 mask → 逐结节实例(2026-08-01 新增)

> **状态:已实现 + 已测试(27 例),主流程尚未调用。**

官方要求**每个检出结节**单独一条记录,而本项目模型输出的是**单通道整图 mask**
(`multi_task_model.seg_head` 是 `Conv2d(20,1,1)`,无实例分割概念)。本模块用连通域
分析补这一层。

### 三个设计决策

1. **在原图尺寸上拆,不在 512×512 模型分辨率上拆** —— 先拆再逐实例 resize 会让每个
   实例各经历一次最近邻插值,相邻结节边界还原后可能重叠或产生缝隙,累计误差不可控。
2. **面积阈值用相对比例**(`min_area_ratio=1e-4`)而非绝对像素 —— 超声图 400×300 到
   1024×768 不等,固定像素阈值在小图误杀真结节、大图放过噪声。
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

> **状态:已实现 + 已测试(60 例),主流程尚未调用。**

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

### 占位值是已知缺口,不是终态

三个新维度的分类头尚未训练。`build_nodule_entry` 对未提供的维度填保守占位:ETE=1
(无侵犯)、Location 均匀分布(argmax 落"左上叶")、T=0。**占位保证 JSON 结构合规不丢
分,但这三项本身得分等同随机。** 分类头训好后必须由调用方显式传入真实预测。

### 两条不显眼但致命的校验

- **`Malignancy.predicted` 必须是 Python `bool`** —— `numpy.bool_` 不是 `bool`,
  `json.dump` 直接抛 TypeError ⇒ **整份 prediction.json 写不出去**,比任何单字段错更严重。
- **`LocationProbs` 和必须严格为 1** —— 规范明写"顺序固定且和为 1",fp16→fp32→float
  往返会留 1e-7 量级残差,严格解析器会拒。`normalize_location_probs` 把残差并入最大项
  (相对扰动最小);`uniform_location_probs` 并入**首位**(并入末位会让占位 argmax 恒落
  第 7 类"峡部" —— 解剖上体积最小、结节发生率最低的分区,是更糟的占位偏差)。

---

## 11. 接入待办(2026-08-01 记,尚未完成)

新增的三个模块与 `src/service/` 均为**独立可测单元**,但**主流程一行都没调用**。
在下列改造完成前,它们对交付**零贡献**:

| 缺口 | 现状 | 后果 |
|:---|:---|:---|
| 入口不认 `.nii.gz` | `entrypoint.sh` 按扩展名分发,`${INPUT_PATH##*.}` 对 `.nii.gz` 剥出 `gz` → 落 `*)` → 走 `predict.py` 静态图入口;内容嗅探层 `file --mime-type` 返回 `application/gzip`,既不匹配 `image/*` 也不匹配 `video/*` → 直接 "Unrecognized input format" exit 1 | **官方动图 100% 零预测** |
| 部署形态未切换 | `entrypoint.sh` 仍是"跑一次推理就 exit"的离线 CLI;规范 §二/§三 要求 8000 端口长驻 HTTP 服务 | 平台探针打不到服务 → 连续失败 3 次容器重启 → **评测无法开始** |
| 推理未接入服务层 | `src/service/app.py` 的 `set_runner()` 无任何调用方,`_runner` 恒为 None,生产请求全走 `_placeholder_runner` | 结果目录只有 `_placeholder.txt`,回调却报"已完成" → **全样本零分且无失败信号** |
| 输出契约不相交 | 现有链路写 `result.json` + `mask_predicted.png`(整图 0/255 PNG);官方要 `prediction.json` + 每结节 `.nii.gz`(0/1) | 评测按规范找不到文件 → **判"无预测输出"** |
| 模型仍是整图级 | 分类头全是整图 pooled 特征上的单次线性分类,无实例分割 | 多结节记录只能靠 `nodule_split` 事后拆分,属性无法逐结节区分 |
