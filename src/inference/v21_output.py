"""V2.1 单病例输出落盘:`{case_dir}/{subdir_uid}/nodule_NNN.nii.gz` + `{case_dir}/prediction.json`。

本模块是**唯一的 V2.1 写盘入口**。它把 `case_pipeline` 的推理结果(模型输出尺寸的
软/二值 mask、良恶性概率、TI-RADS 分项)转成官方《赛事开发规范 V2.1》§五要求的
逐结节交付形态:

  1. 软 `pred_mask`(sigmoid)与 `mask_binary` 都先 resize 回**原始输入图 H×W**——规范
     硬约束 mask 尺寸必须与原图严格一致。软 mask 用 `INTER_LINEAR`(概率平滑),二值
     mask 用 `INTER_NEAREST`(不产生 0/255 之外的中间值,保住二值语义)。
  2. `nodule_split.split_nodules` 在原图尺寸的二值 mask 上做连通域分析,得到逐结节
     实例(2D 静态 / 3D 动态跨帧)。
  3. 每结节 `nifti_io.save_binary_mask` 落一个 `.nii.gz`(值域强制 {0,1});
     检测置信度用 `detection_confidence(软图, 实例 mask)`(区域内概率均值)。
  4. 每结节 `build_nodule_entry`(良恶性/TI-RADS 用真实预测;ETE=1/Location 均匀/
     T=0 为训练阶段4未训的保守占位)→ `build_prediction_json` →
     `validate_prediction_json` → **原子写**(先 `.tmp` 再 `os.replace`)。

阴性病例(`split_nodules` 出 0 个结节)是合法输出:`Nodules`/`MaskFiles` 均为空数组,
仍写一份结构合规的 `prediction.json`(规范:阴性输出空数组,不建空子目录)。

## 两个最容易出错、且本地测不出来的点(写死在本模块,勿改)

- **维度约定**:`save_binary_mask` 内部已做 `(H,W)→(W,H)` / `(n,H,W)→(W,H,n)` 的
  NIfTI 列主序转置,本模块**只传 numpy 行主序数组**,不再额外转置(转两次等于没转,
  评测按 header 读形状会判尺寸不符)。
- **mask_file 必须是相对病例一级目录的路径** `{subdir_uid}/nodule_NNN.nii.gz`,
  禁绝对路径、禁 `..`(`_validate_relative_mask_path` 在 build/validate 双侧拦截)。
  本模块用 `os.path.join(subdir_uid, name)` 拼盘内绝对路径,用
  `f"{subdir_uid}/{name}"`(**正斜杠**)拼写进 JSON 的相对路径——Windows 上
  `os.path.join` 会产 `\\`,虽然校验放行,但官方评测在 Linux 按 `/` 解析,反斜杠会让
  文件定位失败。

## 动态跨帧结节关联(@云TODO)

动态图官方要求每个结节输出 **3D mask,Z 轴对齐全视频帧序列**(评估只抽标注帧计分,
但缺帧的帧无法计分)。本模块的取舍:

  - 各抽样帧逐帧 resize 回原图后,用 `scatter_frames_to_full_length(帧mask, 帧号,
    n_total_frames)` 回填成 `(n_total_frames, H, W)` 全帧 3D mask(未抽中帧填 0);
  - 在此 3D mask 上跑 `split_nodules`(3D 模式,26-邻域 + 时间维闭运算)——**时间上
    连续出现的同一结节自然连成一个 3D 实例**,这就是本期的"跨帧关联"。

**这是有意的最简实现,不是完备的多目标跨帧跟踪**:若分割在中间某帧漏检导致同一
结节断开,会被拆成两个实例(NoduleID 重复计数);两个空间相邻的结节也可能被闭运算
粘连。`nodule_split._split_3d` 的 docstring 已记录该局限。真正的跨帧实例关联
(tracking-by-detection / ReID)依赖官方数据校正,标 `@云TODO` 待训练阶段4后处理。

`n_total_frames` 由调用方(runner)在 case_result 里提供(A1 已让 loader 返回并一路
透传);取不到时回退为「最大帧号+1」并打 warning——此时若有未被抽到的尾部帧,3D
mask 的 Z 长度会短于原体积,评测按 SliceIndex 取标注帧可能越界,故回退路径必须告警。
"""

import json
import os

import numpy as np

try:
    import cv2

    HAS_CV2 = True
except ImportError:  # pragma: no cover - 依赖缺失路径由调用方兜底
    HAS_CV2 = False

from src.inference import prediction_json as pj
from src.inference.nifti_io import save_binary_mask, scatter_frames_to_full_length
from src.inference.nodule_split import detection_confidence, split_nodules

__all__ = ["write_case_v21"]


class V21OutputError(RuntimeError):
    """V2.1 落盘失败(输入契约不满足 / 写盘异常)。

    独立于 `PredictionSchemaError`:schema 错是「组装逻辑 bug」,本异常是「case_result
    缺关键键 / 维度不符 / 依赖缺失」这类运行期故障,由 runner 的病例级隔离捕获转
    ERROR 占位,不拖垮整批。
    """


def _tirads_class_from_breakdown(breakdown) -> int:
    """从 `build_tirads_breakdown` 的 `total_points` 推 1..5 的 TI_RADS_Class。

    复刻 `report_generator.calculate_tirads_points` 内置的分级阈值(同一份 ACR 规则,
    单一事实源在那里的 tr_level 字符串里;此处只需 int 类别,故独立实现阈值映射,
    不解析字符串——字符串是给人看的,解析它会随措辞改动而脆断):
        0 分→1(TR1),≤2→2(TR2),==3→3(TR3),≤6→4(TR4),≥7→5(TR5)。
    """
    pts = int(breakdown["total_points"])
    if pts <= 0:
        return 1
    if pts <= 2:
        return 2
    if pts == 3:
        return 3
    if pts <= 6:
        return 4
    return 5


def _static_orig_hw(case_result) -> tuple:
    """静态病例的原图 (H, W)。

    唯一可靠来源是 `case_result["orig_hw"]`(调用方在加载输入图时记下 `img_np.shape`
    并透传)。`img_tensor` 是 `preprocess_image` resize 到 `img_size`(如 512×512)后的
    模型输入张量,**其 H×W 不是原图尺寸**,绝不能拿来当原图——故缺 `orig_hw` 时直接
    报错,不静默用模型输入尺寸冒充原图(那会让 mask 尺寸与原图不符,评测判尺寸不一致
    而整例失分,且本地不易察觉)。
    """
    if "orig_hw" in case_result and case_result["orig_hw"] is not None:
        h, w = case_result["orig_hw"]
        return int(h), int(w)
    raise V21OutputError(
        "静态 case_result 缺 `orig_hw`(原图 H×W)。调用方必须在加载输入图时透传 "
        "`orig_hw=img_np.shape[:2]`——img_tensor 是 resize 到 img_size 的模型输入,"
        "不能用来推原图尺寸")


def _dynamic_orig_hw(case_result) -> tuple:
    """动态病例的原图 (H, W)。

    动态链路每帧的 `frame_np` 就是已加载的原图灰度帧(uint8 HxW,未经模型 resize),
    故最权威来源是 `kf_frame_np.shape`(关键帧原图)。这比静态的 img_tensor 回退可靠。
    仍保留 `orig_hw` 透传优先,便于调用方显式覆盖。
    """
    if "orig_hw" in case_result and case_result["orig_hw"] is not None:
        h, w = case_result["orig_hw"]
        return int(h), int(w)
    kf_frame = case_result.get("kf_frame_np")
    if kf_frame is not None:
        arr = np.asarray(kf_frame)
        if arr.ndim >= 2:
            return int(arr.shape[0]), int(arr.shape[1])
    # 最后回退:任一 per_frame 的 frame_np。
    per_frame = case_result.get("per_frame") or []
    if per_frame and per_frame[0].get("frame_np") is not None:
        arr = np.asarray(per_frame[0]["frame_np"])
        return int(arr.shape[0]), int(arr.shape[1])
    raise V21OutputError("动态 case_result 既无 orig_hw 也无可用 frame_np,无法确定原图 H×W")


def _resize_soft(soft_mask: np.ndarray, hw: tuple) -> np.ndarray:
    """软概率 mask resize 回原图,`INTER_LINEAR` 保概率平滑。返回 float 概率图(H,W)。"""
    h, w = hw
    out = cv2.resize(
        np.asarray(soft_mask, dtype=np.float32), (w, h),
        interpolation=cv2.INTER_LINEAR)
    return out


def _resize_binary(binary_mask: np.ndarray, hw: tuple) -> np.ndarray:
    """二值 mask resize 回原图,`INTER_NEAREST` 保二值语义(不出中间值)。返回 {0,1} uint8。"""
    h, w = hw
    out = cv2.resize(
        np.asarray(binary_mask), (w, h), interpolation=cv2.INTER_NEAREST)
    # save_binary_mask/split_nodules 接受任意非零为前景,但此处显式压成 {0,1},
    # 避免把内部 0/255 一路带进逐结节实例(实例 mask 值域本就 {0,1},统一更干净)。
    return (out != 0).astype(np.uint8)


def _mask_filename(index: int) -> str:
    """结节 mask 文件名(1-based,三位零填充):`nodule_001.nii.gz`。"""
    return f"nodule_{index:03d}.nii.gz"


def _atomic_write_json(path: str, payload: dict) -> None:
    """原子写 JSON(先写 `.{pid}.tmp` 再 `os.replace`),防半写文件。

    **刻意不复用 `export._write_json_atomic`**:那份用 `json.dump(payload, f, indent=2)`
    无 `ensure_ascii=False`,会把中文转义成 `\\uXXXX`;而 V2.1 的 `Interpretation.Conclusion`
    (CBM 报告/拒绝原因)是中文,必须 `encoding="utf-8"` + `ensure_ascii=False` 原样保留。
    两处算法相同(tmp+os.replace)但序列化参数不同,合并会破坏中文输出——勿"好心统一"。
    `.tmp` 名带 pid 防并发/重入撞名。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _build_entries_and_save_masks(instances, soft_fullres, case_dir, subdir_uid,
                                  nodule_base_context):
    """公共尾段:逐结节落 mask + 组装 build_nodule_entry。

    Args:
        instances: `split_nodules` 的实例列表(每项含 `mask`,2D 或 3D)。
        soft_fullres: 与实例 mask 同 shape 的软概率图(2D 原图 / 3D 全帧),供
            `detection_confidence` 取区域内概率均值。
        case_dir / subdir_uid: 落盘根与相对子目录。
        nodule_base_context: dict,含 `malignancy_confidence` / `tirads_class` /
            `ete_code` / `location_probs` / `location` / `t_clin_sig` / `t_confidence`
            等每结节共享的预测值(真实预测 + 占位)。

    Returns:
        nodule_entries — 供 build_prediction_json;阴性时为空列表。
    """
    entries = []
    for i, inst in enumerate(instances, start=1):
        name = _mask_filename(i)
        abs_path = os.path.join(case_dir, subdir_uid, name)
        # 写进 JSON 的相对路径强制正斜杠(官方 Linux 评测按 `/` 解析,见模块 docstring)。
        rel_path = f"{subdir_uid}/{name}"

        # 落盘:实例 mask(2D/3D,值域 {0,1})。save_binary_mask 内部做 NIfTI 转置。
        save_binary_mask(inst["mask"], abs_path)

        det_conf = detection_confidence(soft_fullres, inst["mask"])

        entry = pj.build_nodule_entry(
            nodule_id=f"N{i}",
            mask_file=rel_path,
            detection_confidence=det_conf,
            malignancy_confidence=nodule_base_context["malignancy_confidence"],
            tirads_class=nodule_base_context["tirads_class"],
            ete_code=nodule_base_context.get("ete_code", pj.ETE_NONE),
            location_probs=nodule_base_context.get("location_probs"),
            location=nodule_base_context.get("location"),
            t_clin_sig=nodule_base_context.get("t_clin_sig", 0),
            t_confidence=nodule_base_context.get("t_confidence", 0.0),
        )
        entries.append(entry)
    return entries


def _shared_nodule_context(preds, breakdown) -> dict:
    """从 preds/breakdown 提取每结节共享的预测上下文(真实预测 + 占位维度)。

    占位维度(训练阶段4未训,结构合规但数值等同随机,见 prediction_json docstring):
    ETE=1(无侵犯)、Location=均匀分布(argmax→1 左上叶,刻意有偏)、T=0。
    分类头训好后由调用方在 case_result 显式传真实值替换(此处读取可选覆盖键)。
    """
    return {
        "malignancy_confidence": float(preds["mal_prob"]),
        "tirads_class": _tirads_class_from_breakdown(breakdown),
        # --- 占位维度(可被 case_result 的显式键覆盖,便于未来接真实三头) ---
        "ete_code": pj.ETE_NONE,
        "location_probs": None,   # None → build_nodule_entry 内部用 uniform_location_probs
        "location": None,         # None → argmax(uniform)+1 = 1(左上叶,占位刻意有偏)
        "t_clin_sig": 0,
        "t_confidence": 0.0,
    }


def _write_static(case_dir, patient_id, subdir_uid, case_result,
                  processing_ms, interpretation):
    preds = case_result["preds"]
    breakdown = case_result["breakdown"]
    hw = _static_orig_hw(case_result)

    soft_fullres = _resize_soft(preds["pred_mask"], hw)
    binary_fullres = _resize_binary(preds["mask_binary"], hw)

    instances = split_nodules(binary_fullres)
    context = _shared_nodule_context(preds, breakdown)
    entries = _build_entries_and_save_masks(
        instances, soft_fullres, case_dir, subdir_uid, context)

    payload = pj.build_prediction_json(
        study_uid=patient_id,
        nodules=entries,
        processing_time_ms=processing_ms,
        interpretation=interpretation,
    )
    pj.validate_prediction_json(payload)
    _atomic_write_json(os.path.join(case_dir, "prediction.json"), payload)


def _write_dynamic(case_dir, patient_id, subdir_uid, case_result,
                   processing_ms, interpretation):
    if not HAS_CV2:
        raise V21OutputError("动态 V2.1 落盘需要 cv2(resize),但 cv2 未安装")

    breakdown = case_result["breakdown"]
    kf_preds = case_result["kf_preds"]
    per_frame = case_result["per_frame"]
    hw = _dynamic_orig_hw(case_result)

    # --- 全帧 3D mask:抽样帧逐帧 resize 回原图 → scatter 回填到视频全长 ---
    frame_indices = [int(f["frame_index"]) for f in per_frame]
    n_total = case_result.get("n_total_frames")
    if not n_total or int(n_total) <= 0:
        # 回退:最大帧号+1。若尾部有未抽中帧,Z 长度会短于原体积(见模块 docstring)。
        n_total = (max(frame_indices) + 1) if frame_indices else 0
        print(f"[v21_output][WARNING] 动态 case_result 缺有效 n_total_frames,回退为 "
              f"max(frame_indices)+1={n_total};尾部未抽中帧会导致 3D mask Z 长度"
              f"不足 @云TODO", flush=True)
    n_total = int(n_total)

    binary_frames = []
    soft_frames = []
    for f in per_frame:
        p = f["preds"]
        binary_frames.append(_resize_binary(p["mask_binary"], hw))
        soft_frames.append(_resize_soft(p["pred_mask"], hw))

    binary_3d = scatter_frames_to_full_length(binary_frames, frame_indices, n_total)
    soft_3d = scatter_frames_to_full_length(soft_frames, frame_indices, n_total)

    # 3D 连通域 = 跨帧实例关联(局限见模块 docstring @云TODO)。
    instances = split_nodules(binary_3d)
    context = _shared_nodule_context(kf_preds, breakdown)
    entries = _build_entries_and_save_masks(
        instances, soft_3d, case_dir, subdir_uid, context)

    payload = pj.build_prediction_json(
        study_uid=patient_id,
        nodules=entries,
        processing_time_ms=processing_ms,
        interpretation=interpretation,
    )
    pj.validate_prediction_json(payload)
    _atomic_write_json(os.path.join(case_dir, "prediction.json"), payload)


def write_case_v21(case_dir, patient_id, subdir_uid, case_result, kind,
                   processing_ms, interpretation=None) -> None:
    """把单病例推理结果落成官方 V2.1 交付形态(逐结节 .nii.gz + prediction.json)。

    Args:
        case_dir: 该病例输出目录(已含 patient_id 层,即 `{output_dir}/{PatientId}`)。
            调用方保证父目录存在;本函数负责建 `{case_dir}/{subdir_uid}/` 子目录
            (由 `save_binary_mask` 内部 `makedirs` 完成)。
        patient_id: 填 `prediction.json` 的 `StudyUID`(V2.1 起其值即 PatientId)。
        subdir_uid: mask 相对路径的子目录名(静态=SeriesUid,动态=AccessionNumber)。
        case_result: `diagnose_static_case` / `diagnose_dynamic_case` 的返回 dict
            (A2 runner 会额外注入 `orig_hw`/`n_total_frames`/`gatekeeper_audit` 等键,
            本函数按需读取,缺省有回退)。
        kind: `"static"` | `"dynamic"`。
        processing_ms: 推理总耗时(毫秒),透传进 `ProcessingTime_ms`(int)。
        interpretation: 可选 CBM 报告 `{"Conclusion": str, "KeyFindings": [str]}`。

    Returns:
        None。产物写在磁盘上。

    Raises:
        V21OutputError: case_result 契约不满足 / 依赖缺失 / 维度不符。
        PredictionSchemaError: 组装出的 payload 不合规(应由 build/validate 自拦)。
    """
    if not HAS_CV2:
        raise V21OutputError("V2.1 落盘需要 cv2(mask resize 回原图),但 cv2 未安装")

    # 拒绝/无诊断语义由调用方(runner `_write_case_error` / `_rejection_result`)兜底,
    # 本函数只处理「有真实诊断结果」的病例。`preds is None` 是 rejection 占位,不应到此处。
    if case_result.get("rejected"):
        raise V21OutputError(
            "case_result 标记为 rejected(gatekeeper 拒绝),不应走 V2.1 正常落盘路径;"
            "请由 runner 的 ERROR/REJECTED 兜底处理")

    os.makedirs(case_dir, exist_ok=True)

    if kind == "static":
        _write_static(case_dir, patient_id, subdir_uid, case_result,
                      processing_ms, interpretation)
    elif kind == "dynamic":
        _write_dynamic(case_dir, patient_id, subdir_uid, case_result,
                       processing_ms, interpretation)
    else:
        raise V21OutputError(f"未知 kind={kind!r},只支持 'static'/'dynamic'")
