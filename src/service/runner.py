"""V2.1 HTTP 服务的真实推理 runner：把一个 dataset_path 目录批量推理成逐病例 V2.1 输出。

## 定位与接线

`src/service/app.py:main()` 启动时调用 `set_runner(build_service_runner(...))` 注入本模块
产出的 runner,替换骨架模式的 `_placeholder_runner`(占位模式见 `app.py`)。runner 契约
由 `tasks.submit_inference` 锁定:

    runner(dataset_path, output_dir) -> None

- `dataset_path` = 平台 `/call` 下发的**批量病例目录**;
- `output_dir`   = `config.answer_dir(evaluation_id)` = `{ANSWER_ROOT}/{evaluation_id}`,
  `tasks._run` 已 `makedirs` 好,runner 只管往里写每病例 `{PatientId}/...`。

## 批处理流水线

    discover_cases(dataset_path)          # case_discovery:静/动分派 + PatientId/subdir_uid
      └ 逐病例(每病例独立 try/except):
          static  → load_input_image(含 DICOM 匿名化落盘) → gatekeeper → diagnose_static_case
          dynamic → load_nifti_frames → 逐帧 gatekeeper → diagnose_dynamic_case
          ↓
          case_dir = {output_dir}/{patient_id}  →  write_case_v21(..., kind, processing_ms)

## 三条红线(对应 QA_VP §1.7 / rules.md §5 先证后改)

1. **模型只 build 一次、跨病例复用**:`build_model`/`load_config`/`TemperatureScaler`/
   gatekeeper 三件重型对象(deduplicator/auth_module/tamper_detector)在闭包里建一次,
   绝不每病例重载 —— 批量推理每病例重读权重 + 重播种 200 张训练图会把延迟打爆
   (赛道有推理时延自动评分)。模块级 `_RUNNER_CACHE` 进一步让「同参数重复调
   `build_service_runner`」(多次 `/call` 或测试)也只建一次。
2. **单病例失败隔离**:每病例独立 `try/except Exception`,一例崩溃(权重不匹配/
   CUDA OOM/NIfTI 损坏/数值 NaN)只落该例的 `prediction.json` ERROR 占位 + 记日志,
   **绝不 raise 拖垮整批** —— 平台只有一次评测机会,一例崩掉整批 = 全部零输出。
   与逐帧坏帧隔离(case_pipeline)同一哲学,粒度提到病例级。
3. **匿名化/gatekeeper 不因批量而跳过**:每病例照跑 `load_input_image` 的 DICOM
   PHI 匿名化与逐帧 gatekeeper 审计,与 CLI(`predict.py`/`video_predict.py`)同语义。
   gatekeeper 默认 soft(赛道七豁免,误拒一次白丢该病例全部临床得分),故 `passed`
   恒 True;仅当配置显式 `enforcement_mode: hard` 且命中硬判据时才走拒绝分支
   (落 `prediction.json` REJECTED 占位,不产诊断)。

## 加载失败/无有效诊断的病例落盘语义

V2.1 的病例级「失败/拒绝」落盘**复用 `write_case_v21` 的原子写**,而非 V2.0 的
`save_rejection_json`(后者写 `result.json`,V2.1 目录里不该出现它)。占位 nodule
只保证 JSON 结构合规(必填字段齐全、LocationProbs 和为 1),数值是保守占位
(ETE=1 无侵犯/Location 均匀/T=0),并靠 `Interpretation.Conclusion` 写明 ERROR/REJECTED
原因 —— 与 A3 的「阴性空输出」「占位维度」同一套降级哲学:**结构合规优先,数值诚实标注**。

@云TODO(云端首跑校正):
- 权重/config 路径来自环境变量 `AICOMP_WEIGHTS`/`AICOMP_CONFIG`/`AICOMP_PRECISION`
  (缺省回落 predict.py 同款路径),容器内若挂载点不同须云端首跑核对;
- gatekeeper 在批量下的耗时(逐帧审计 × 病例数)未实测,云端首跑观察是否需降采样;
- 动态 AccessionNumber(=subdir_uid)来源已明确(组委会口径=直接父目录名,见 case_discovery),
  但推理输入目录层级仍须云端首跑核对。
"""

import logging
import os
import threading
import time

from src.service.case_discovery import discover_cases
from src.inference import prediction_json

logger = logging.getLogger(__name__)

# ── 运行时参数来源(环境变量优先,缺省回落 CLI 同款路径)────────────────────
# 与 predict.py/video_predict.py 的 --config/--weights/--precision 默认值逐位对齐,
# 让容器不注入环境变量时行为与 CLI 完全一致(本地/CI 调试可直接跑)。三个环境变量
# 按 config.py 的 `AICOMP_*` 前缀约定命名,集中一处便于平台侧挂载点变更时只改这里。
_ENV_WEIGHTS = "AICOMP_WEIGHTS"
_ENV_CONFIG = "AICOMP_CONFIG"
_ENV_PRECISION = "AICOMP_PRECISION"
# 动态 loader 的 max_frames(均匀抽样上限),<=0 取全帧。与 video_predict.py 默认 32 对齐。
_ENV_MAX_FRAMES = "AICOMP_MAX_FRAMES"

_DEFAULT_CONFIG = "configs/thyroid_config.yaml"
_DEFAULT_WEIGHTS = "checkpoints/thyroidxl_merged_v2.pth"
_DEFAULT_PRECISION = "fp16"
_DEFAULT_MAX_FRAMES = 32
_VALID_PRECISIONS = ("fp32", "fp16", "bf16")

# ── 模块级 runner 缓存(同参数只 build 一次)────────────────────────────────
# key = (config_path, weights, precision, max_frames);value = runner 闭包。
# 线程池 worker=1(tasks.INFERENCE_WORKERS)下天然串行,但 /call 的 event loop 线程
# 可能并发调 build_service_runner,故加锁防「双检查」竞态重复 build。
_RUNNER_CACHE = {}
_CACHE_LOCK = threading.Lock()


def _resolve_runtime_params(config_path, weights, precision, max_frames):
    """把 None 参数解析为环境变量值/缺省值;显式传参优先(测试注入用)。"""
    config_path = config_path or os.environ.get(_ENV_CONFIG) or _DEFAULT_CONFIG
    weights = weights or os.environ.get(_ENV_WEIGHTS) or _DEFAULT_WEIGHTS
    precision = precision or os.environ.get(_ENV_PRECISION) or _DEFAULT_PRECISION
    if precision not in _VALID_PRECISIONS:
        logger.warning(
            "[runner] precision=%r 非法,降级 %s(合法值 %s)",
            precision, _DEFAULT_PRECISION, _VALID_PRECISIONS)
        precision = _DEFAULT_PRECISION
    if max_frames is None:
        try:
            max_frames = int(os.environ.get(_ENV_MAX_FRAMES, str(_DEFAULT_MAX_FRAMES)))
        except ValueError:
            max_frames = _DEFAULT_MAX_FRAMES
    return config_path, weights, precision, max_frames


def build_service_runner(config_path=None, weights=None, precision=None, max_frames=None):
    """构建生产 runner(模型只 build 一次,跨病例/跨 `/call` 复用)。

    Args:
        config_path/weights/precision/max_frames: None 时按环境变量→缺省值解析
            (见 `_resolve_runtime_params`)。测试显式传参以隔离环境变量。

    Returns:
        `runner(dataset_path, output_dir) -> None`,签名与 `tasks.submit_inference`
        的调用点逐字对齐。

    模型构建失败(config 损坏/权重不匹配/CUDA 不可用)在**首次 runner 被调用时**才
    抛出(惰性),由 `tasks._run` 兜成 `_error.json` + FAILED + 回调 —— 不在
    `build_service_runner` 时抛,避免服务启动期(`main()`)因模型未就位直接崩掉
    8000 端口探针(平台探活连续失败 3 次会重启容器,启动期崩 = 整场评测无法开始)。
    """
    config_path, weights, precision, max_frames = _resolve_runtime_params(
        config_path, weights, precision, max_frames)
    cache_key = (os.path.abspath(config_path), os.path.abspath(weights),
                 precision, max_frames)

    with _CACHE_LOCK:
        cached = _RUNNER_CACHE.get(cache_key)
    if cached is not None:
        logger.info("[runner] 复用已建模型缓存 key=%s", cache_key)
        return cached

    # 惰性构建:真正的 build_model/load_config 推迟到 runner 首次被调用(见 docstring)。
    state = {"built": False, "model": None, "config": None, "calibrator": None,
             "device": None, "weights": None, "dedup": None, "auth": None,
             "tamper": None}

    def _ensure_built():
        """首次调用时建模型 + gatekeeper 三件重型对象,后续幂等复用。"""
        if state["built"]:
            return
        import torch
        from src.inference.model_runner import build_model, load_config
        from src.inference.calibration import TemperatureScaler
        from src.inference.gatekeeper_runner import (
            build_seeded_deduplicator, build_auth_module, build_tamper_detector)

        config = load_config(config_path)
        device = torch.device(
            config["environment"]["device"] if torch.cuda.is_available() else "cpu")
        model = build_model(config, weights, device)
        cal_path = config.get("calibration", {}).get(
            "param_file", "checkpoints/calibration.json")
        calibrator = TemperatureScaler.from_file(cal_path)
        # gatekeeper 三件重型对象建一次跨病例/跨帧复用(与 video_predict 同款:
        # dedup 播种 200 张训练图、auth 加载 authenticity 权重、tamper 探 CAT-Net 权重,
        # 每病例/每帧重建会把批量延迟打爆)。
        state.update(
            model=model, config=config, calibrator=calibrator, device=device,
            weights=weights,
            dedup=build_seeded_deduplicator(config, device),
            auth=build_auth_module(weights),
            tamper=build_tamper_detector(config),
            built=True)
        logger.info(
            "[runner] 模型已构建并缓存:config=%s weights=%s precision=%s device=%s",
            config_path, weights, precision, device)

    def runner(dataset_path, output_dir):
        """生产推理入口(tasks.submit_inference 的 runner 契约)。"""
        _ensure_built()
        cases = discover_cases(dataset_path)
        logger.info(
            "[runner] dataset_path=%s 发现 %d 病例 → output_dir=%s",
            dataset_path, len(cases), output_dir)
        if not cases:
            # 空目录/全跳过不是 runner 错误:发现层是「尽量多发现」语义,
            # 无效 dataset_path 由 tasks 层(_error.json)判 ERROR,此处如实记日志。
            logger.warning(
                "[runner] 未发现任何病例(dataset_path=%s)。整批无输出,"
                "若平台期望非空须核对目录结构(@云TODO)", dataset_path)
        for case in cases:
            _process_case(case, output_dir, state, precision, max_frames)

    with _CACHE_LOCK:
        _RUNNER_CACHE[cache_key] = runner
    return runner


def _process_case(case, output_dir, state, precision, max_frames):
    """单病例流水线:计时 → 分派静/动 → write_case_v21;独立 try/except 失败隔离。"""
    patient_id = case["patient_id"]
    kind = case["kind"]
    case_dir = os.path.join(output_dir, str(patient_id))
    t0 = time.perf_counter()
    try:
        os.makedirs(case_dir, exist_ok=True)
        if kind == "static":
            case_result = _run_static_case(case, case_dir, state, precision)
        else:
            case_result = _run_dynamic_case(case, case_dir, state, precision, max_frames)
        processing_ms = int((time.perf_counter() - t0) * 1000)
        _write_via_v21(case_dir, case, case_result, kind, processing_ms)
        logger.info(
            "[runner] 病例完成 patient_id=%s kind=%s 耗时=%dms",
            patient_id, kind, processing_ms)
    except Exception as e:  # noqa: BLE001 — 单病例失败隔离红线,见模块 docstring
        processing_ms = int((time.perf_counter() - t0) * 1000)
        logger.exception(
            "[runner] 病例失败已隔离(patient_id=%s kind=%s):%s: %s",
            patient_id, kind, type(e).__name__, e)
        _write_case_error(case_dir, case, processing_ms,
                          f"{type(e).__name__}: {e}")


def _run_static_case(case, case_dir, state, precision):
    """静态图:load_input_image(含匿名化)→ gatekeeper → diagnose_static_case。"""
    from src.inference.io import load_input_image
    from src.inference.gatekeeper_runner import run_gatekeeper_audit
    from src.inference.case_pipeline import diagnose_static_case

    img_np, is_dicom, error_msg = load_input_image(case["input_path"], case_dir)
    if img_np is None:
        raise RuntimeError(f"输入加载失败: {error_msg}")

    passed, audit = run_gatekeeper_audit(
        img_np, state["config"], state["device"], state["weights"],
        case_dir, deduplicator=state["dedup"], auth_module=state["auth"],
        tamper_detector=state["tamper"])
    if not passed:  # 仅 hard 模式可达(见模块 docstring 红线 3)
        return _rejection_result(audit)

    result = diagnose_static_case(
        state["model"], img_np, state["config"], state["calibrator"],
        precision, state["device"])
    result["gatekeeper_audit"] = {"skipped": False, **audit}
    result["is_dicom"] = is_dicom
    # A3 write_case_v21 硬需原图 H×W:img_tensor 是 resize 到 img_size 的模型输入,
    # 尺寸≠原图,拿它冒充会让结节 mask 尺寸与原图不符→官方判尺寸不一致整例失分。
    # 故在此透传加载时的原图尺寸(img_np 已在手,零成本)。
    result["orig_hw"] = tuple(img_np.shape[:2])
    return result


def _run_dynamic_case(case, case_dir, state, precision, max_frames):
    """动态图:load_nifti_frames → 逐帧 gatekeeper → diagnose_dynamic_case。"""
    from src.inference.nifti_io import load_nifti_frames
    from src.inference.gatekeeper_runner import (
        run_gatekeeper_audit, resolve_enforcement_mode)
    from src.inference.case_pipeline import diagnose_dynamic_case, DynamicCaseError

    frames, frame_indices, _is_dicom, error_msg, n_total_frames = load_nifti_frames(
        case["input_path"], case_dir, max_frames)
    if frames is None:
        raise RuntimeError(f"输入加载失败: {error_msg}")

    # 逐帧 gatekeeper(复用跨病例已建的重型对象;与 video_predict 逐帧同款)。
    frame_audits = []
    for pos, frame_np in zip(frame_indices, frames):
        passed, audit = run_gatekeeper_audit(
            frame_np, state["config"], state["device"], state["weights"],
            case_dir, deduplicator=state["dedup"], auth_module=state["auth"],
            tamper_detector=state["tamper"])
        if not passed:  # 仅 hard 模式可达
            audit["rejected_frame_index"] = int(pos)
            return _rejection_result(audit)
        if audit["flagged"]:
            frame_audits.append({"frame_index": int(pos), **audit})
    gatekeeper_audit = {
        "skipped": False,
        "enforcement_mode": resolve_enforcement_mode(state["config"]),
        "n_frames_audited": len(frames),
        "n_frames_flagged": len(frame_audits),
        "flagged_frame_indices": [a["frame_index"] for a in frame_audits],
        "flagged": bool(frame_audits),
        "would_reject_in_hard_mode": any(
            a["would_reject_in_hard_mode"] for a in frame_audits),
        "flagged_frames": frame_audits,
    }

    try:
        result = diagnose_dynamic_case(
            state["model"], frames, frame_indices, state["config"],
            state["calibrator"], precision, state["device"])
    except DynamicCaseError as e:
        # 全帧数值失败/无有效关键帧 → 无有效诊断,落 ERROR 占位(携带诊断字段)。
        raise RuntimeError(f"动态图无有效诊断: {e.reason} {e.diagnostic}") from e
    result["gatekeeper_audit"] = gatekeeper_audit
    result["n_total_frames"] = n_total_frames
    return result


def _rejection_result(audit):
    """gatekeeper hard 拒绝 → 构造「拒绝语义」的 case_result(不产诊断)。"""
    return {
        "rejected": True,
        "reason": audit.get("reason", "gatekeeper rejected"),
        "gatekeeper_audit": audit,
        "preds": None,
    }


def _write_via_v21(case_dir, case, case_result, kind, processing_ms):
    """经 A3 的 write_case_v21 落盘(唯一 V2.1 写盘入口,原子写 + 校验)。"""
    from src.inference.v21_output import write_case_v21
    write_case_v21(
        case_dir, case["patient_id"], case["subdir_uid"], case_result, kind,
        processing_ms)


def _write_case_error(case_dir, case, processing_ms, reason):
    """单病例崩溃的降级落盘:复用 write_case_v21 的原子写,产出结构合规的
    ERROR/REJECTED 占位 prediction.json(数值为保守占位,Interpretation 写明原因)。

    红线:此函数自身绝不能抛(写盘失败只记日志)—— 否则隔离语义自相矛盾
    (崩溃病例的兜底写盘再崩 = 该例零输出且整批被异常中断)。
    """
    try:
        from src.inference import prediction_json as pj
        nodule = pj.build_nodule_entry(
            nodule_id="N1",
            mask_file=f"{case['subdir_uid']}/nodule_001.nii.gz",
            detection_confidence=0.0,
            malignancy_confidence=0.0,
            tirads_class=1,
        )
        payload = pj.build_prediction_json(
            study_uid=case["patient_id"],
            nodules=[nodule],
            processing_time_ms=processing_ms,
            interpretation={
                "Conclusion": f"病例推理失败(降级占位): {reason}",
                "KeyFindings": ["该病例因运行期异常未能产出真实诊断,本 JSON 为结构合规占位"],
            })
        pj.validate_prediction_json(payload)
        _atomic_write_json(os.path.join(case_dir, "prediction.json"), payload)
    except Exception:  # noqa: BLE001 — 兜底写盘失败不得再抛,掩盖原始错误
        logger.exception(
            "[runner] 病例级 ERROR 占位写盘失败(patient_id=%s)", case["patient_id"])


def _atomic_write_json(path, payload):
    """与 export._write_json_atomic 同款原子写(先 tmp 再 os.replace),防半写文件。"""
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
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
