# src/inference/AGENTS.md - Shared Inference Pipeline Specifications

This package provides reusable inference modules shared by `predict.py` (production) and `evaluate.py` (holdout benchmarking).

---

## Package Structure

| File | Responsibility |
|:---|:---|
| `io.py` | Image loading (DICOM + PNG/JPG) and PHI anonymization;通道/位深归一化工具 `_to_gray`/`_rescale_to_uint8` 下沉于此(静/动两路径共用,video_io re-import)。`_to_gray` 覆盖全部通道形态((H,W)/(H,W,1)/(H,W,3)/RGBA/channel-first/多光谱 均收敛 2D,杜绝 3D 漏过致下游 `preprocess_image` 5 维崩溃),并以 `channel_order` 区分 RGB(DICOM 像素)与 BGR(OpenCV 视频帧)避免多普勒伪彩加权错位。单图 DICOM 经 `_to_gray` 收敛灰度 |
| `gatekeeper_runner.py` | 4-layer gatekeeper audit pipeline (predict.py 单图 + video_predict.py 动态图) |
| `model_runner.py` | Model build, image preprocessing, forward pass |
| `postprocess.py` | Prediction decoding and TI-RADS breakdown(可选接 `calibration.TemperatureScaler` 做温度缩放) |
| `calibration.py` | Post-hoc 置信度校准(温度缩放):`TemperatureScaler`(每任务标量 T,缺参数即 T=1.0 降级)+ 离线拟合 `fit_temperature`/`fit_ordinal_temperature` + 诊断 `compute_ece`/`reliability_diagram`。真实数据 2026-08-01 初赛才到,现仅骨架 |
| `export.py` | Artifact saving (mask, report, JSON, Grad-CAM overlay) |
| `video_io.py` | 动态图/视频(cine)抽帧:MP4/AVI/MOV/MKV via OpenCV + 多帧 DICOM cine via pydicom;复用 `io._PHI_FIELDS` 脱敏,经 `_to_gray` 统一灰度兜底(含彩色/多普勒 cine);OpenCV 视频帧传 `channel_order="bgr"`、DICOM cine 用默认 `"rgb"`。供 `video_predict.py`(根目录) |
| `video_aggregate.py` | 视频推理纯逻辑:`uniform_sample_indices`(均匀抽样下标)+ `aggregate_frame_predictions`(帧级聚合,恶性概率取**最大帧**做最终判定,召回优先) |

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

### `run_gatekeeper_audit(img_np, config, device, weights_path, output_dir, deduplicator=None, auth_module=None, tamper_detector=None)` → `(passed, rejection_json)`

**Scope**: Production inference only. evaluate.py 仅导入 model_runner/postprocess/export 三模块；gatekeeper_runner 由 predict.py(单图) 和 video_predict.py(动态图) 使用。

- 三个可选预建对象(视频逐帧复用,避免每帧重复加载权重/播种训练集):
  - `deduplicator`: 用 `build_seeded_deduplicator(config, device)` 预建；
  - `auth_module`: 用 `build_auth_module(weights_path)` 预建(单图路径懒建,跳过 `authenticity_model.pth` 每帧重复加载)；
  - `tamper_detector`: 用 `build_tamper_detector(config)` 预建(避免每帧重探 catnet 权重文件)。
- 三者均为 None 时内部即时构建——单图 predict.py 路径行为不变(向后兼容)。
- **无状态污染**:`is_duplicate` 只读比对,不把当前帧 `add_image` 入库;故同一 cine 相邻近重复帧之间不会互相误判(去重库只播种训练集)。

Pipeline:
1. **Authenticity** — `AuthenticityModule` (13-dim GLCM+freq+ELA+DND+TAI → MLP) → phantom/synthetic detection。**2026-07-12 统一软flag制度**:分类结果**不再**驱动 REJECTED,仅作审计字段(`is_phantom`/`is_gan` 保留在 rejection JSON 里,但只在去重/篡改触发 REJECTED 时才会出现,自身不能单独致拒)。权重缺失/损坏仍在 `AuthenticityModule.load_weights` 硬报错,红线未变。详见 `src/gatekeeper/AGENTS.md` §1。
2. **Deduplication** — `Deduplicator` (pHash + cosine) seeded from train set。硬拒绝,未软化。
3. **Copy-Move** — `detect_copy_move()` dense block-matching。**恒为软flag**(`last_soft_flags`),与 CatNet 状态无关,详见下方第 5 点。
4. **Splicing(硬粘贴)** — `detect_splicing()` 轴对齐长直缝 + 角点正交相交判据(压低误报至零)。**恒为软flag**,同上。
5. **Splicing(羽化/泊松)** — `tampering_catnet.CatNetTamperingDetector` CAT-Net v2 DCT/BAG 取证(块伪影栅格相位错位),可插拔降级 dct-bag。**2026-07-12 统一软flag制度后 `detect()` 判定逻辑改为**:CAT-Net **自身**前向命中(本次调用成功)硬拒绝 ∪ BAG 相位不一致(仅当 `bag_hard_reject=True` 显式开启;默认 `False` 仅软flag) ∪ 几何(copy_move/splice,**恒软flag,与 CatNet 状态解耦**,不再因"CatNet 这次没崩溃"被提升为硬拒绝)。硬拒绝命中源写入 `CatNetTamperingDetector.last_hard_reject_source`("catnet"|"bag_phase"|None),`gatekeeper_runner.py` 据此拼接精确的 REJECTED reason(而非笼统的"hard-reject path")。详见 `src/gatekeeper/AGENTS.md` §3。

**Rejection JSON** (on failure)。`is_phantom`/`is_gan` 现为纯审计字段——出现非零值**不代表**该次判定单独触发了 REJECTED,只有 `is_duplicate`/`is_tampered` 驱动 `status="REJECTED"`(见上方第1点)；但若本次因去重/篡改被拒绝且 phantom/synthetic 同时命中,`reason` 文本会追加 `"(soft-flag: ...)"` 标注保留该并发信号(2026-07-12 code-review 补强,详见 `src/gatekeeper/AGENTS.md` §1):
```json
{
  "status": "REJECTED",
  "reason": "...",
  "is_phantom": 0|1,
  "is_gan": 0|1,
  "is_duplicate": 0|1,
  "is_tampered": 0|1
}
```

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
Returns keys: `pred_mask`, `mask_binary`, `comp_idx`, `echo_idx`, `shape_idx`, `margin_idx`, `foci_prob`, `mal_prob`.
- `calibrator`: 可选 `TemperatureScaler`。各分类头在 softmax/sigmoid **之前**先 `apply(task, logits)` 温度缩放;`None` 或全 T=1.0 时输出逐位等同无校准。各头统一 `.float()` 后再缩放(fp16 温度除法精度对齐),T>0 单调不改 argmax。mask 分割不走校准。

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

