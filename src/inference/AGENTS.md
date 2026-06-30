# src/inference/AGENTS.md - Shared Inference Pipeline Specifications

This package provides reusable inference modules shared by `predict.py` (production) and `evaluate.py` (holdout benchmarking).

---

## Package Structure

| File | Responsibility |
|:---|:---|
| `io.py` | Image loading (DICOM + PNG/JPG) and PHI anonymization;通道/位深归一化工具 `_to_gray`/`_rescale_to_uint8` 下沉于此(静/动两路径共用,video_io re-import)。`_to_gray` 覆盖全部通道形态((H,W)/(H,W,1)/(H,W,3)/RGBA/channel-first/多光谱 均收敛 2D,杜绝 3D 漏过致下游 `preprocess_image` 5 维崩溃),并以 `channel_order` 区分 RGB(DICOM 像素)与 BGR(OpenCV 视频帧)避免多普勒伪彩加权错位。单图 DICOM 经 `_to_gray` 收敛灰度 |
| `gatekeeper_runner.py` | 4-layer gatekeeper audit pipeline (**predict.py only**) |
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
- Returns `(None, False, error_msg)` on failure — caller must check and print error.

---

## 2. `gatekeeper_runner.py` — 4-Layer Audit

### `run_gatekeeper_audit(img_np, config, device, weights_path, output_dir, deduplicator=None)` → `(passed, rejection_json)`

**Scope**: Production inference only. `evaluate.py` MUST NOT import this module.

- `deduplicator` 可选:传入预先用 `build_seeded_deduplicator(config, device)` 播种好的实例,供**视频逐帧复用**(否则每帧重复读取+ResNet 编码最多 200 张训练图,O(帧数×200) 拖慢延迟)。为 None 时即时构建,单图路径行为不变(向后兼容)。
- **无状态污染**:`is_duplicate` 只读比对,不把当前帧 `add_image` 入库;故同一 cine 相邻近重复帧之间不会互相误判(去重库只播种训练集)。

Pipeline:
1. **Authenticity** — `AuthenticityModule` (13-dim GLCM+freq+ELA+DND+TAI → MLP) → phantom/synthetic detection
2. **Deduplication** — `Deduplicator` (pHash + cosine) seeded from train set
3. **Copy-Move** — `detect_copy_move()` dense block-matching
4. **Splicing(硬粘贴)** — `detect_splicing()` 轴对齐长直缝 + 角点正交相交判据(压低误报至零)
5. **Splicing(羽化/泊松)** — `tampering_catnet.CatNetTamperingDetector` CAT-Net v2 DCT/BAG 取证(块伪影栅格相位错位),可插拔降级 dct-bag;`detect()` = 深度CAT-Net ∪ BAG ∪ 几何三路并集

**Rejection JSON** (on failure):
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

### `build_model(config, weights_path, device)` → `MultiTaskThyroidModel`
- Loads checkpoint with `strict=False` (allows missing/extra keys).
- Falls back to random init with `[WARNING]` if weights not found.

### `preprocess_image(img_np, img_size, device)` → `torch.Tensor`
- `cv2.resize` to `img_size`, normalize `/255.0`, shape `[1, 1, H, W]`.

### `run_forward(model, img_tensor, precision, device)` → `dict`
- **`torch.inference_mode()`** — never `torch.no_grad()`.
- **`torch.autocast`** with device-adaptive dtype:
  - CPU + fp16 → fallback to fp32 with `[WARNING]`.
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
| `evaluate.py` | `model_runner`(含 `run_forward`), `postprocess`, `export` | `gatekeeper_runner` (holdout = clean ground truth) |

**前向一致性**: `evaluate.py` 必须经 `run_forward(model, img, precision, device)` 跑前向(与 `predict.py` 同一 precision + malignancy clamp),不得直接 `model(img)`,否则 holdout AUC 与生产数值路径不一致、基准失真。`evaluate.py` 提供 `--precision {fp32,fp16,bf16}`,评估生产指标时应与部署精度一致。
