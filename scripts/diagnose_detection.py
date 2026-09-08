"""检测诊断仪表盘 · CLI 编排(2026-09-05,检测 AP 3.05 / Dice 8.62 提分主战场第一步)。

在云端 val 集(与训练同 `seed`/`val_frac` 患者分组切分,`build_official_datasets`
现算切分故与 `train_official.py` 同一批患者)上对 r5 权重跑**真实推理链**,
按官方匹配口径(`detection_match.py`)做匈牙利匹配,产出 oracle AP 分解四件套
+ 病因统计,落 JSON 报告 + sidecar JSONL + 控制台摘要。**生产代码零改动**——
本脚本只读 import 生产模块,逐行镜像 `v21_output._write_static/_write_dynamic`
的实例抽取链(只不落盘),与 `write_case_v21` 的等价性由
`tests/test_diagnose_detection.py` 的回归测试钉死。

## 三个高危对齐点(改本文件前必读,错了整个报告是垃圾)

1. **帧号对齐(动态)**:`per_frame[i]["frame_index"]` 是**原始帧号**,经
   `scatter_frames_to_full_length` 摊到 `(n_total,H,W)` 后帧轴语义 = 原始帧号,
   与 GT `_read_nifti_binmasks` 全帧读出的帧轴天然对齐。**绝不能用 per_frame
   的列表位置当帧号**(32 个抽样位置 ≠ n_total 原始帧号)。xlsx 的 SliceIndex
   (1-based)不使用——GT 标注帧一律从 mask 非零帧定位(与 loader
   `_nonzero_mask_frames` 同源)。
2. **gap 的 n_sampled**:`n_sampled = len(per_frame)`——per_frame 是坏帧隔离后的
   幸存帧(`case_pipeline.py:149-153`),**不是** `load_nifti_frames` 返回的 32
   (生产 `v21_output.py:455` 同口径)。取错会让 `close_gap_frames` 与生产漂移,
   实例碎裂数不同,镜像前提破产。
3. **按 img_path 聚 case**:val_ds 静态一图多 GT 病灶(每病灶一样本)、动态每
   非零帧一样本(`official_loader.build_official_samples:567`)。必须先按
   `img_path` 聚 case 一次推理、全 GT 一起进匈牙利——否则同图重复推理 N 次、
   同图其余 GT 被误判漏检,匹配率/AP 双错。

## gatekeeper 整体跳过(有意)

诊断链消费的是 `diagnose_static_case`/`diagnose_dynamic_case`,它们只吃帧数组;
gatekeeper 在生产里是调用方(runner)的守门环节,soft 模式从不 REJECT、审计帧
不改像素(提速三件套已证输出逐位相同),对检测 AP 的镜像零影响。

## 动态 detection_confidence 的语义分叉(🔴 2026-09-08 生产已修,本脚本已同步镜像)

生产 `v21_output._write_dynamic` 曾把 float `soft_frames` 喂进
`scatter_frames_to_full_length`,其内部 `out=(m!=0)` **二值化成 uint8**
(`nifti_io.py`)→ `detection_confidence(soft_3d, inst)` 实际度量的是
"实例体素落在抽样帧的比例"(帧覆盖率),对全程存在的结节恒趋近 1.0、排序能力
近零。**2026-09-08 生产改 `_dynamic_det_conf`(实例出现帧的逐帧 float 概率
均值),本脚本同日同步镜像** —— 动态 conf 直方图不再预期挤 ~1.0;若新报告仍
挤 ~1.0,说明修复未生效(先查生产改的是否被部署)。

## 口径声明(写进报告 meta)

val 集只含有 ROI 标注的患者(loader 只从 M/病灶N 行建样本),**阴性病例的假
阳性不进池 → 本报告 AP 相对官方(含阴性 FP)偏高**;分数用于病因分解与相对
比较,不用于对齐官方 3.05 绝对值。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.detection_match import (  # noqa: E402
    DEDUP_OVERLAP_THRESHOLD, MATCH_IOU_THRESHOLD, aggregate_case_rows,
    build_iou_matrix, dedup_keep_mask, match_instances)

# ── 生产模块(全部只读复用,口径单一权威)──────────────────────────────────
from src.dataset.official_loader import (  # noqa: E402
    _read_nifti_binmasks, _safe_print, build_official_datasets)
from src.inference.case_pipeline import (  # noqa: E402
    DynamicCaseError, diagnose_dynamic_case, diagnose_static_case)
from src.inference.io import load_input_image  # noqa: E402
from src.inference.nifti_io import (  # noqa: E402
    load_nifti_frames, scatter_frames_to_full_length)
from src.inference.nodule_split import (  # noqa: E402
    detection_confidence, split_nodules)
from src.inference.v21_output import (  # noqa: E402
    _dynamic_det_conf, _resize_binary, _resize_soft)


# ── case 聚合 ───────────────────────────────────────────────────────────────

def group_cases(samples: list) -> tuple:
    """val 样本 → (case 列表, 无 mask 样本计数)。**按 `img_path` 聚**(高危对齐点 3)。

    每项:{"img_path", "is_dynamic", "gts": [样本dict...](去重后)}。
    同一 img_path 下的多个样本 = 同图多 GT 病灶(静态)或同病灶多帧样本(动态),
    GT 按 `labels["lesion_id"]` 去重(mask_path 同源,与 loader 权威定义一致)。
    `mask_path=None` 的样本过滤(不计 GT、单独计数——静态 13% 非 M 行无 mask)。
    """
    by_img = {}
    n_no_mask = 0
    for s in samples:
        key = s["img_path"]
        if key not in by_img:
            by_img[key] = {"img_path": key, "is_dynamic": bool(s["is_dynamic"]),
                           "gts": {}}
        # B-P2-3 防御:同 img_path 混合 is_dynamic 说明 loader 路径空间不再
        # 互斥(当前静态 *.dcm / 动态 *.nii.gz 不相交,不可达),静默错分派
        # 比崩更危险。
        if bool(s["is_dynamic"]) != by_img[key]["is_dynamic"]:
            raise ValueError(
                f"group_cases: 同 img_path 混合静/动样本 {key} —— loader 契约破坏")
        if s.get("mask_path") is None:
            n_no_mask += 1
            continue
        lesion = s["labels"]["lesion_id"]
        if lesion not in by_img[key]["gts"]:
            by_img[key]["gts"][lesion] = s
    cases = list(by_img.values())
    for c in cases:
        c["gts"] = list(c["gts"].values())
    return cases, n_no_mask


# ── 推理(逐行镜像 v21_output 的实例抽取,只不落盘)─────────────────────────

def run_static_inference(model, config, calibrator, precision, device,
                         img_path: str) -> list:
    """静态:镜像 `_write_static`(v21_output.py:374-385)的实例抽取链。

    Returns: 实例 list,每项 {"mask": (H,W) uint8, "area": int, "confidence": float}。
    """
    img_np, _is_dicom, err = load_input_image(img_path, "")
    if img_np is None:
        raise RuntimeError(f"输入加载失败 {img_path}: {err}")
    case_result = diagnose_static_case(model, img_np, config, calibrator,
                                       precision, device)
    # 镜像 runner.py:333 的注入(orig_hw = 原图 H×W,img_tensor 是 resize 到
    # img_size 的模型输入、不能冒充原图——`_static_orig_hw` 缺键即 raise)。
    case_result["orig_hw"] = tuple(int(x) for x in img_np.shape[:2])
    return run_static_inference_from_result(case_result)


def run_static_inference_from_result(case_result) -> list:
    """从 diagnose_static_case 的返回抽实例(与 load 环节解耦,供等价回归测试)。

    实例抽取逐行镜像 `_write_static`(v21_output.py:374-385):resize →
    `split_nodules`(默认参数)→ `detection_confidence`(float 软图,区域内
    概率均值——与动态的"帧覆盖率语义"不同源,静/动分叉点)。
    """
    preds = case_result["preds"]
    hw = tuple(int(x) for x in case_result["orig_hw"])
    soft_fullres = _resize_soft(preds["pred_mask"], hw)
    binary_fullres = _resize_binary(preds["mask_binary"], hw)
    instances = split_nodules(binary_fullres)  # 默认参数同生产
    out = []
    for inst in instances:
        out.append({
            "mask": inst["mask"],
            "area": int(inst["area"]),
            "confidence": detection_confidence(soft_fullres, inst["mask"]),
        })
    return out


def run_dynamic_inference(model, config, calibrator, precision, device,
                          img_path: str, max_frames: int):
    """动态:镜像 `_write_dynamic`(v21_output.py:397-457)的实例抽取链。

    Returns: (实例 list, n_total)。实例 mask 为 `(n_total,H,W)` uint8
    (帧轴 = 原始帧号,与 GT 对齐,高危对齐点 1)。
    """
    frames, frame_indices, _is_dicom, err, n_total = load_nifti_frames(
        img_path, "", max_frames)
    if frames is None:
        raise RuntimeError(f"输入加载失败 {img_path}: {err}")
    try:
        case_result = diagnose_dynamic_case(
            model, frames, frame_indices, config, calibrator, precision, device)
    except DynamicCaseError as e:
        raise RuntimeError(f"动态图无有效诊断 {img_path}: {e.reason}") from e
    # 🔴 镜像 runner.py:410 的注入步骤:`n_total_frames` **不是**
    # diagnose_dynamic_case 的返回键(case_pipeline.py:210-219 无此键),生产链
    # 由 runner 注入后 _write_dynamic 才能读。B 路 2026-09-05 审查实测:漏注入
    # → KeyError → 被宽 except 兜成 anomaly → 云端动态 case 全灭且报告"合法"。
    case_result["n_total_frames"] = n_total
    del frames
    return run_dynamic_inference_from_result(case_result)


def run_dynamic_inference_from_result(case_result):
    """从 diagnose_dynamic_case 的返回抽实例(与 load 环节解耦,供等价回归测试)。

    实例抽取逐行镜像 `_write_dynamic`:`n_total_frames` 硬校验、逐帧 resize →
    scatter → gap 按 `len(per_frame)`(高危对齐点 2)→ `split_nodules` → conf。
    Returns: (实例 list, n_total)。
    """
    n_total = int(case_result["n_total_frames"])
    if not n_total or n_total <= 0:
        raise RuntimeError("动态 case_result 缺有效 n_total_frames")

    per_frame = case_result["per_frame"]
    hw = tuple(int(x) for x in np.asarray(per_frame[0]["frame_np"]).shape[:2])
    frame_idx = [int(f["frame_index"]) for f in per_frame]
    binary_frames, soft_frames = [], []
    for f in per_frame:
        p = f["preds"]
        binary_frames.append(_resize_binary(p["mask_binary"], hw))
        soft_frames.append(_resize_soft(p["pred_mask"], hw))
    binary_3d = scatter_frames_to_full_length(binary_frames, frame_idx, n_total)
    # 镜像生产 2026-09-08 修复:soft_3d 死分配删除(生产不再 scatter float soft,
    # conf 走 soft_by_frame 逐帧概率均值)。此处同样不分配。
    # 🔴 高危对齐点 2:n_sampled = len(per_frame)(坏帧隔离后的幸存帧),
    # 非 max_frames。生产 v21_output.py 同口径,取错碎裂数漂移。
    n_sampled = len(frame_idx)
    gap = max(1, -(-n_total // n_sampled) + 1) if n_sampled else 1
    instances = split_nodules(binary_3d, close_gap_frames=gap)
    # 直接复用生产 `_dynamic_det_conf` 保口径(A 路 F4:共享实现使等价测试退化为
    # 同函数两次调用,公式级回归仅靠 fixture 数值断言兜底——生产侧若改公式,
    # 本镜像自动跟随,等价性永真但绝对值无独立桩,这是已知取舍)。
    soft_by_frame = {int(fi): sf for fi, sf in zip(frame_idx, soft_frames)}
    out = []
    for inst in instances:
        out.append({
            "mask": inst["mask"],
            "area": int(inst["area"]),
            "confidence": _dynamic_det_conf(inst, soft_by_frame),
        })
    del binary_3d, binary_frames, soft_frames
    return out, n_total


# ── GT 装载 ─────────────────────────────────────────────────────────────────

def load_gts_for_case(case: list) -> list:
    """case 的 GT 病灶 list(读 mask,静态转 2D、动态保留 3D+标注帧)。

    每项:{"mask", "is_main", "area_ratio", "annotated_frames"(动态)}。
    静态 GT 实测为 (1,H,W) 3D 单帧 → 取首帧转 2D(多非零帧记 anomaly 由调用方
    在 shape 断言里拦)。动态 `annotated_frames` = mask 非零帧(0-based 原始帧号),
    与 loader `_nonzero_mask_frames` 同源。
    """
    gts = []
    for s in case:
        binf, idxs = _read_nifti_binmasks(s["mask_path"])
        if not binf:
            gts.append({"error": f"GT mask 读取失败: {s['mask_path']}"})
            continue
        if s["is_dynamic"]:
            g3d = np.stack(binf)  # (n_total, H, W),帧轴 = 原始帧号
            ann = [i for i, m in zip(idxs, binf) if m.any()]
            ann_set = set(ann)
            area = float(np.mean([m.sum() for i, m in zip(idxs, binf)
                                  if i in ann_set])) if ann else 0.0
            gts.append({
                "mask": g3d, "is_main": bool(s["labels"]["is_main"]),
                "annotated_frames": ann, "n_mask_frames": len(binf),
                "hw": (g3d.shape[1], g3d.shape[2]),
                "area": area,
                "area_ratio": area / float(g3d.shape[1] * g3d.shape[2]),
            })
        else:
            m0 = np.asarray(binf[0])
            if len(binf) > 1 and any(np.asarray(b).any() for b in binf[1:]):
                gts.append({"error": f"静态 GT 多非零帧: {s['mask_path']}"})
                continue
            gts.append({
                "mask": m0, "is_main": bool(s["labels"]["is_main"]),
                "n_mask_frames": len(binf), "hw": m0.shape[:2],
                "area": float(m0.sum()),
                "area_ratio": float(m0.sum()) / float(m0.size),
            })
    return gts


# ── 单 case 诊断行 ──────────────────────────────────────────────────────────

def diagnose_one_case(model, config, calibrator, precision, device,
                      case: dict, max_frames: int) -> dict:
    """推理 + GT + 匹配 → case 行(sidecar JSONL 的一行,JSON 可序列化)。"""
    img_path = case["img_path"]
    row = {"case_key": img_path, "is_dynamic": case["is_dynamic"],
           "anomaly": None, "n_pred": 0, "n_gt": 0, "preds": [], "gts": []}
    gts = load_gts_for_case(case["gts"])
    bad = [g["error"] for g in gts if "error" in g]
    if bad:
        row["anomaly"] = f"GT 读取异常: {bad[0]}"
        return row
    gts = [g for g in gts if "error" not in g]

    try:
        if case["is_dynamic"]:
            preds, n_total = run_dynamic_inference(
                model, config, calibrator, precision, device,
                img_path, max_frames)
            # 帧/HW 硬断言:GT 帧 mask 数须 == n_total、HW 一致(高危对齐点 1 的防线)。
            for g in gts:
                if g["n_mask_frames"] != n_total or g["hw"] != preds_hw(preds, g):
                    row["anomaly"] = (
                        f"动态 shape 不齐: gt_frames={g['n_mask_frames']} "
                        f"n_total={n_total} gt_hw={g['hw']} pred_hw={preds_hw(preds, g)}")
                    return row
        else:
            preds = run_static_inference(
                model, config, calibrator, precision, device, img_path)
            # 静态 HW 不齐时 build_iou_matrix 会 ValueError,由下方统一拦。
    except Exception as e:  # noqa: BLE001 — 单 case 失败不拖垮全局,记 anomaly
        row["anomaly"] = f"推理失败: {type(e).__name__}: {e}"
        return row

    n_gt = len(gts)
    row["n_gt"] = n_gt
    row["n_pred"] = len(preds)
    try:
        iou_m = build_iou_matrix(preds, gts)
        match = match_instances(iou_m, MATCH_IOU_THRESHOLD)
        # C 路 2026-09-05 审查 P1:ap_dedup 的 keep 逐 case 现算库语义
        # `dedup_keep_mask` 并写进 sidecar —— 池级用近似规则会双向失真:
        # 分叉A(独苗近失碎片 best_iou∈(0.1,0.5) 未匹配)会被误删 → 高估碎裂损失;
        # 分叉B(matched 但非该 GT 列 argmax 的 TP)按库语义本该删 → 低估。
        # 现算 = 单一权威,池化零语义漂移。
        # A 路 F3 修复(2026-09-05):必须传 match —— 匈牙利 TP 并入 keep 集,
        # 否则"非列 argmax 的真 TP"被删、无碎裂却读出碎裂损失。
        keep = dedup_keep_mask(iou_m, match=match)
    except ValueError as e:
        row["anomaly"] = f"匹配失败(shape 不齐): {e}"
        return row

    for i, p in enumerate(preds):
        best = float(iou_m[i].max()) if n_gt > 0 else 0.0
        row["preds"].append({
            "confidence": round(float(p["confidence"]), 6),
            "area": p["area"],
            "best_iou": round(best, 6),
            "matched": bool(match["pred_to_gt"][i] >= 0),
            "dedup_keep": bool(keep[i]),
        })
    for j, g in enumerate(gts):
        max_iou = float(iou_m[:, j].max()) if len(preds) > 0 else 0.0
        row["gts"].append({
            "is_main": g["is_main"],
            "area_ratio": round(float(g["area_ratio"]), 8),
            "max_iou": round(max_iou, 6),
            "matched": bool(match["gt_to_pred"][j] >= 0),
            # B-P2-2:阈值必须与 dedup_keep_mask 同源(DEDUP_OVERLAP_THRESHOLD),
            # 字面量 0.1 会在常量调整后静默分叉。
            "n_candidates": (int((iou_m[:, j] > DEDUP_OVERLAP_THRESHOLD).sum())
                             if len(preds) else 0),
        })
    return row


def preds_hw(preds: list, g: dict):
    """静态/动态统一:预测实例 mask 的 (H,W)(无预测时回退 GT 自身 HW)。"""
    if preds:
        return tuple(int(x) for x in preds[0]["mask"].shape[-2:])
    return g["hw"]


# ── 主流程 ──────────────────────────────────────────────────────────────────

def _pool(case_rows: list, is_dynamic=None) -> dict:
    """从逐 case 行重建某池(静/动/合并)的实例与 GT,算四件套。

    `ap_dedup` 消费 sidecar 行里逐 case 现算的 `dedup_keep` 标志(由
    `diagnose_one_case` 调 `detection_match.dedup_keep_mask` 产出,单一权威)
    —— 池级不得另拟近似规则(C 路 2026-09-05 审查实测两种近似均失真)。
    差值 `ap_current − ap_dedup` 即碎裂 FP 对 AP 的稀释量。
    """
    preds, n_gt = [], 0
    for row in case_rows:
        if row.get("anomaly"):
            continue
        if is_dynamic is not None and row["is_dynamic"] != is_dynamic:
            continue
        # 实例的 matched/best_iou/dedup_keep 直接来自 case 级匹配(同一图内
        # 匹配,池化不改对)。
        for p in row["preds"]:
            preds.append({"confidence": p["confidence"],
                          "_matched": p["matched"],
                          "_best_iou": p.get("best_iou", 0.0),
                          "_keep": p.get("dedup_keep", True)})
        n_gt += row["n_gt"]
    labels = [int(p["_matched"]) for p in preds]
    confs = [p["confidence"] for p in preds]
    oracle = [1.0 if l else 0.0 for l in labels]
    ranked = [p["_best_iou"] for p in preds]
    keep = [p["_keep"] for p in preds]
    labels_d = [l for l, k in zip(labels, keep) if k]
    confs_d = [c for c, k in zip(confs, keep) if k]
    from scripts.detection_match import average_precision
    return {
        "n_pred": len(preds), "n_gt": n_gt, "n_tp": sum(labels),
        "match_rate": (sum(labels) / n_gt) if n_gt else 0.0,
        "ap_current": average_precision(labels, confs, n_gt),
        "ap_oracle": average_precision(labels, oracle, n_gt),
        "ap_iou_ranked": average_precision(labels, ranked, n_gt),
        "ap_dedup": average_precision(labels_d, confs_d, n_gt),
    }


def run_diagnosis(args) -> dict:
    """主流程(供测试 import 直调)。"""
    import torch
    from src.inference.model_runner import build_model, load_config

    if not os.path.exists(args.weights):
        # 🔴 非 PRODUCTION 下 build_model 缺权重只 WARNING 随机初始化(#115 同类),
        # 整份报告会变垃圾 —— 这里硬拦。
        raise FileNotFoundError(f"权重不存在: {args.weights}")
    # 权重路径打印为绝对路径(C 路 2026-09-05 审查):--weights 默认是相对路径、
    # 按 cwd 解析,静默装错(如旧冒烟产物)时报告变垃圾且无提示——启动时把
    # 实际解析的绝对路径吼出来,让"跑的到底是哪份权重"可被运维一眼核对。
    _safe_print(f"[diag] weights = {os.path.abspath(args.weights)}")
    config = load_config(args.config)
    device = torch.device(
        config["environment"]["device"] if torch.cuda.is_available() else "cpu")
    model = build_model(config, args.weights, device)
    from src.inference.calibration import TemperatureScaler
    calibrator = TemperatureScaler.from_file(
        config.get("calibration", {}).get("param_file",
                                          "checkpoints/calibration.json"))

    _safe_print(f"[diag] 构建 val 集(root={args.data_root})…")
    seed = args.seed if args.seed is not None else int(
        config.get("environment", {}).get("seed", 42))
    _, val_ds = build_official_datasets(
        args.data_root, args.xlsx, img_size=int(config["data"]["img_size"]),
        val_frac=args.val_frac, seed=seed, samples_pkl=args.samples_pkl)
    samples = val_ds.samples
    cases, n_no_mask = group_cases(samples)
    if args.limit:
        cases = cases[: args.limit]
    _safe_print(f"[diag] val 样本 {len(samples)} → {len(cases)} case"
                f"(无 mask 样本 {n_no_mask} 已过滤;limit={args.limit or '全量'})")

    os.makedirs(args.log_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sidecar_path = os.path.join(args.log_dir, f"detection_diag_{ts}.cases.jsonl")
    report_path = os.path.join(args.log_dir, f"detection_diag_{ts}.json")
    case_rows = []
    t0 = time.time()
    with open(sidecar_path, "w", encoding="utf-8") as sf:
        for k, case in enumerate(cases):
            row = diagnose_one_case(
                model, config, calibrator, args.precision, device,
                case, args.max_frames)
            case_rows.append(row)
            sf.write(json.dumps(row, ensure_ascii=False) + "\n")
            sf.flush()
            if (k + 1) % 20 == 0 or k + 1 == len(cases):
                _safe_print(f"[diag] {k + 1}/{len(cases)} case 完成"
                            f"(anomaly {sum(1 for r in case_rows if r.get('anomaly'))})")
    runtime_s = round(time.time() - t0, 1)

    ok_rows = [r for r in case_rows if not r.get("anomaly")]
    report = {
        "meta": {
            # os.path.abspath:云端误用相对路径时报告里留的是可追的绝对路径。
            "weights": os.path.abspath(args.weights), "config": args.config,
            "seed": seed, "val_frac": args.val_frac,
            "max_frames": args.max_frames, "precision": args.precision,
            "n_samples": len(samples), "n_cases": len(cases),
            "n_no_mask_samples": n_no_mask,
            "runtime_s": runtime_s, "sidecar": sidecar_path,
            "match_threshold": MATCH_IOU_THRESHOLD,
            "caveats": [
                "val 只含有 ROI 标注患者,阴性 FP 不在池 → AP 相对官方偏高,"
                "只做相对分解不对齐绝对值",
                "动态 detection_confidence 2026-09-08 起为逐帧 float 概率均值"
                "(旧帧覆盖率语义已修);若动态 conf 直方图仍挤 ~1.0 = 修复未生效",
                f"动态抽样 max_frames={args.max_frames}(2026-09-08 起生产与"
                "本脚本默认均为 0=全帧;若为 32 属旧口径回退,动态匹配率会因"
                "漏标注帧被系统性低估)",
                "匹配目标函数 = max-ΣIoU 后在 ≥0.5 子图二次分配(匹配数对齐最大"
                "基数匹配,pair 选择以 max-Σ 为 tie-break);pair 级选择仍是实现选择",
                "anomaly 行(推理失败/shape 不齐)整 case 剔除、GT 不进 AP 分母 —— "
                "官方口径下推理失败病例 = 全部 GT 漏检、进分母,故 anomaly 率非零时"
                "本报告 ap_current/match_rate 偏乐观,读数需按 anomalies 明细折算",
            ],
        },
        "anomalies": [{"case": r["case_key"], "reason": r["anomaly"]}
                      for r in case_rows if r.get("anomaly")],
        "static": _pool(case_rows, is_dynamic=False),
        "dynamic": _pool(case_rows, is_dynamic=True),
        "combined": _pool(case_rows, is_dynamic=None),
        "stats_static": aggregate_case_rows(
            [r for r in ok_rows if not r["is_dynamic"]]),
        "stats_dynamic": aggregate_case_rows(
            [r for r in ok_rows if r["is_dynamic"]]),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print_summary(report)
    _safe_print(f"[diag] 报告: {report_path}\n[diag] sidecar: {sidecar_path}")
    report["_report_path"] = report_path
    return report


def print_summary(report: dict) -> None:
    """控制台摘要表(可直接贴回)。"""
    _safe_print("\n===== 检测诊断摘要 =====")
    for name in ("static", "dynamic", "combined"):
        p = report[name]
        _safe_print(
            f"[{name:8s}] n_gt={p['n_gt']:5d} n_pred={p['n_pred']:5d} "
            f"TP={p['n_tp']:5d} 匹配率={p['match_rate']:.4f} | "
            f"AP current={p['ap_current']:.4f} oracle={p['ap_oracle']:.4f} "
            f"iou_ranked={p['ap_iou_ranked']:.4f}")
    for name in ("stats_static", "stats_dynamic"):
        s = report[name]
        # n_anomaly 不打(ok_rows 已滤掉 anomaly,该键恒 0 是死键——真实数在
        # report["anomalies"];C 路 2026-09-05 审查)。
        _safe_print(
            f"[{name}] 临界带[0.35,0.5)占比={s['critical_band_frac']:.3f} | "
            f"主病灶匹配率={s['match_rate_main']} "
            f"次病灶={s['match_rate_secondary']} | "
            f"GT-iou桶={s['gt_iou_hist']} | 每例实例数={s['preds_per_case']}")


def main():
    ap = argparse.ArgumentParser(
        description="检测诊断仪表盘:val 集真实推理 + 官方口径匈牙利匹配 + AP 分解")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--weights", default="checkpoints/official_heads_seg_best.pth")
    ap.add_argument("--config", default="configs/thyroid_config.yaml")
    ap.add_argument("--samples-pkl", default=None)
    # 缺省 0(全帧)与生产 runner._DEFAULT_MAX_FRAMES 同步(2026-09-08 conf 修复
    # + 全帧前向同轮落地;A 路审查 F1:半截镜像会让云端忘传参数时跑回旧抽样世界)。
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--precision", default="fp16",
                    choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=None,
                    help="缺省取 config environment.seed(与训练一致)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-dir", default="logs")
    args = ap.parse_args()
    # 逐 case 完整行始终进 sidecar JSONL(崩溃保数据);--save-per-case 曾计划
    # 把行嵌进主报告,但会撑大 JSON 且与 sidecar 重复,已砍,参数不再收。
    run_diagnosis(args)


if __name__ == "__main__":
    main()
