# src/inference/AGENTS.md - Shared Inference Pipeline Specifications

This package provides reusable inference modules shared by `predict.py` (production) and `evaluate.py` (holdout benchmarking).

---

## Package Structure

| File | Responsibility |
|:---|:---|
| `io.py` | Image loading (DICOM + PNG/JPG) and PHI anonymization;通道/位深归一化工具 `_to_gray`/`_rescale_to_uint8` 下沉于此(静/动两路径共用,video_io re-import)。单图 DICOM 经 `_to_gray` 收敛灰度——彩色多普勒单帧 (H,W,3) 不转灰度会令下游 `preprocess_image` 产出 5 维张量崩溃 |
| `gatekeeper_runner.py` | 4-layer gatekeeper audit pipeline (**predict.py only**) |
| `model_runner.py` | Model build, image preprocessing, forward pass |
| `postprocess.py` | Prediction decoding and TI-RADS breakdown |
| `export.py` | Artifact saving (mask, report, JSON, Grad-CAM overlay) |
| `video_io.py` | 动态图/视频(cine)抽帧:MP4/AVI/MOV/MKV via OpenCV + 多帧 DICOM cine via pydicom;复用 `io._PHI_FIELDS` 脱敏,经 `_to_gray` 统一灰度兜底(含彩色/多普勒 cine)。供 `video_predict.py`(根目录) |
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
1. **Authenticity** — `AuthenticityModule` (GLCM + ELA MLP) → phantom/synthetic detection
2. **Deduplication** — `Deduplicator` (pHash + cosine) seeded from train set
3. **Copy-Move** — `detect_copy_move()` dense block-matching
4. **Splicing** — `detect_splicing()` 轴对齐长直缝 + 角点正交相交判据(压低误报至零)

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

### `decode_predictions(outputs)` → `dict`
Returns keys: `pred_mask`, `mask_binary`, `comp_idx`, `echo_idx`, `shape_idx`, `margin_idx`, `foci_prob`, `mal_prob`.

### `build_tirads_breakdown(comp_idx, echo_idx, shape_idx, margin_idx, foci_prob)` → `dict`
Delegates to `src/utils/report_generator.calculate_tirads_points()`.

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
