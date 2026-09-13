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
    DEFAULT_MIN_AREA_RATIO, detection_confidence, restore_instance_mask,
    split_nodules)
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
                         img_path: str):
    """静态:镜像 `_write_static`(v21_output.py:374-385)的实例抽取链。

    Returns: (case_result, 后处理函数)。case_result 已注入 `orig_hw`
    (镜像 runner.py:333);后处理函数消费 (case_result, seg_threshold=...,
    min_area_ratio=...) → 实例 list——件套②扫参的"一次前向多参数"结构:
    模型前向只跑一次,阈值参数在后处理层循环。
    """
    img_np, _is_dicom, err = load_input_image(img_path, "")
    if img_np is None:
        raise RuntimeError(f"输入加载失败 {img_path}: {err}")
    case_result = diagnose_static_case(model, img_np, config, calibrator,
                                       precision, device)
    # 镜像 runner.py:333 的注入(orig_hw = 原图 H×W,img_tensor 是 resize 到
    # img_size 的模型输入、不能冒充原图——`_static_orig_hw` 缺键即 raise)。
    case_result["orig_hw"] = tuple(int(x) for x in img_np.shape[:2])
    return case_result, run_static_inference_from_result


def run_dynamic_inference(model, config, calibrator, precision, device,
                          img_path: str, max_frames: int):
    """动态:镜像 `_write_dynamic`(v21_output.py:397-457)的实例抽取链。

    Returns: (case_result, 后处理函数)。case_result 已注入 `n_total_frames`
    (镜像 runner.py:410);后处理函数消费 (case_result, seg_threshold=...) →
    (实例 list, n_total)——件套②扫参的"一次前向多参数"结构:模型前向只跑
    一次,阈值参数在后处理层循环。

    (2026-09-10 件套②改造:返回值从"实例 list"改为"case_result + 后处理闭包",
    单参数调用方经 `run_dynamic_inference(...)[0]` 以外的旧写法须改——
    全仓调用方只有 diagnose_one_case 与测试,均已同步。)
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
    return case_result, run_dynamic_inference_from_result


def run_static_inference_from_result(case_result, *, seg_threshold=None,
                                     min_area_ratio=None,
                                     max_nodules=None) -> list:
    """从 diagnose_static_case 的返回抽实例(与 load 环节解耦,供等价回归测试)。

    实例抽取逐行镜像 `_write_static`(v21_output.py:374-385):resize →
    `split_nodules` → `detection_confidence`(float 软图,区域内
    概率均值——与动态的"帧覆盖率语义"不同源,静/动分叉点)。

    件套②(2026-09-10)扫参参数:`seg_threshold`/`min_area_ratio` 不传(None)时
    **逐位镜像生产**——阈值 0.5 复用 preds["mask_binary"](生产静态吃
    AICOMP_SEG_THRESHOLD 后,诊断脚本的同名 env 同步生效,口径不分叉);
    min_area 走 split_nodules 默认 None→env(生产静态同源)。**显式传参优先于
    env**——扫参进程必须确定性,不受运行环境 AICOMP_* 影响(A/B两路审查后
    定稿:此前"显式钉 DEFAULT"会让云端设 env 后诊断与生产口径脱节)。
    `max_nodules` 同款三层优先级(阶段1:None→env→DEFAULT,镜像生产默认参)。
    """
    preds = case_result["preds"]
    hw = tuple(int(x) for x in case_result["orig_hw"])
    soft_fullres = _resize_soft(preds["pred_mask"], hw)
    if seg_threshold is None:
        binary_fullres = _resize_binary(preds["mask_binary"], hw)
    else:
        binary_fullres = _resize_binary(
            (preds["pred_mask"] > seg_threshold).astype(np.uint8) * 255, hw)
    instances = split_nodules(binary_fullres,
                              min_area_ratio=min_area_ratio,
                              max_nodules=max_nodules)
    out = []
    for inst in instances:
        out.append({
            "mask": inst["mask"],
            "area": int(inst["area"]),
            "confidence": detection_confidence(soft_fullres, inst["mask"]),
        })
    return out


def run_dynamic_inference_from_result(case_result, *, seg_threshold=None,
                                      max_nodules=None):
    """从 diagnose_dynamic_case 的返回抽实例(与 load 环节解耦,供等价回归测试)。

    实例抽取逐行镜像 `_write_dynamic`:`n_total_frames` 硬校验、逐帧 resize →
    scatter → gap 按 `len(per_frame)`(高危对齐点 2)→ `split_nodules` → conf。
    Returns: (实例 list, n_total)。

    件套②(2026-09-10):`seg_threshold` 不传(None)时逐位镜像生产(复用
    preds["mask_binary"]);显式传时从 `pred_mask` 现算(扫参用;thr=0.5 与
    mask_binary 逐位等价)。动态 min_area 恒 DEFAULT(镜像生产钉死,不扫)。
    `max_nodules`(阶段1):None→env→DEFAULT 同生产默认参,扫参显式传参优先。
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
        if seg_threshold is None:
            binary_frames.append(_resize_binary(p["mask_binary"], hw))
        else:
            binary_frames.append(_resize_binary(
                (p["pred_mask"] > seg_threshold).astype(np.uint8) * 255, hw))
        soft_frames.append(_resize_soft(p["pred_mask"], hw))
    binary_3d = scatter_frames_to_full_length(binary_frames, frame_idx, n_total)
    # 镜像生产 2026-09-08 修复:soft_3d 死分配删除(生产不再 scatter float soft,
    # conf 走 soft_by_frame 逐帧概率均值)。此处同样不分配。
    # 🔴 高危对齐点 2:n_sampled = len(per_frame)(坏帧隔离后的幸存帧),
    # 非 max_frames。生产 v21_output.py 同口径,取错碎裂数漂移。
    n_sampled = len(frame_idx)
    gap = max(1, -(-n_total // n_sampled) + 1) if n_sampled else 1
    # 镜像生产 2026-09-10 件套④:动态显式传 DEFAULT_MIN_AREA_RATIO(不吃 env),
    # 与 v21_output._write_dynamic 的钉死语义一致——动态放宽是 8h 时限风险面。
    # max_nodules 不钉死(None→env→DEFAULT):动态放开是阶段1扫参的决策对象,
    # 与 min_area 的"已知零收益正成本"不同,须可测。
    instances = split_nodules(binary_3d, min_area_ratio=DEFAULT_MIN_AREA_RATIO,
                              max_nodules=max_nodules,
                              close_gap_frames=gap)
    # 🔴 镜像生产 2026-09-09 稀疏化:full_shape 须在 del binary_3d 前捕获
    # (3D 实例 mask_sparse 写盘/消费前 restore_instance_mask 重建全尺寸)。
    full_shape = binary_3d.shape
    soft_by_frame = {int(fi): sf for fi, sf in zip(frame_idx, soft_frames)}
    del binary_frames, soft_frames
    out = []
    for inst in instances:
        # 镜像生产 _build_entries_and_save_masks:稀疏实例逐个 restore,
        # conf 与 IoU 消费共用,用完即弃(峰值恒 1 份全尺寸)。
        if "mask_sparse" in inst:
            full_mask = restore_instance_mask(inst["mask_sparse"], full_shape)
        else:
            full_mask = inst["mask"]
        out.append({
            "mask": full_mask,
            "area": int(inst["area"]),
            "confidence": _dynamic_det_conf(inst, soft_by_frame, full_mask),
        })
    del binary_3d
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

def _match_row(preds: list, gts: list) -> dict:
    """实例×GT 匹配的核心段(diagnose_one_case 与扫参共用)。

    Returns: {"iou_m", "match", "keep"} 或 {"anomaly": str}(shape 不齐)。
    """
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
        return {"anomaly": f"匹配失败(shape 不齐): {e}"}
    return {"iou_m": iou_m, "match": match, "keep": keep}


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
            # 件套②:动态只扫阈值(min_area 恒 DEFAULT 镜像生产钉死,不扫)。
            case_result, post = run_dynamic_inference(
                model, config, calibrator, precision, device,
                img_path, max_frames)
            n_total = int(case_result["n_total_frames"])
            # 帧/HW 硬断言:GT 帧 mask 数须 == n_total、HW 一致(高危对齐点 1 的
            # 防线)。无预测时用 g 自身 HW 预判(与旧 preds_hw(preds) 空列表回退
            # 同义,且能在 post 跑之前就拦)。
            for g in gts:
                if g["n_mask_frames"] != n_total:
                    row["anomaly"] = (
                        f"动态 shape 不齐: gt_frames={g['n_mask_frames']} "
                        f"n_total={n_total} gt_hw={g['hw']}")
                    return row
            preds, n_total_rt = post(case_result)
            assert n_total_rt == n_total
            # post 后再核 HW(有预测时用预测实例的 HW)。
            for g in gts:
                if preds and g["hw"] != preds_hw(preds, g):
                    row["anomaly"] = (
                        f"动态 shape 不齐: gt_hw={g['hw']} "
                        f"pred_hw={preds_hw(preds, g)}")
                    return row
        else:
            case_result, post = run_static_inference(
                model, config, calibrator, precision, device, img_path)
            preds = post(case_result)
            # 静态 HW 不齐时 build_iou_matrix 会 ValueError,由 _match_row 统一拦。
    except Exception as e:  # noqa: BLE001 — 单 case 失败不拖垮全局,记 anomaly
        row["anomaly"] = f"推理失败: {type(e).__name__}: {e}"
        return row

    n_gt = len(gts)
    row["n_gt"] = n_gt
    row["n_pred"] = len(preds)
    m = _match_row(preds, gts)
    if "anomaly" in m:
        row["anomaly"] = m["anomaly"]
        return row
    iou_m, match, keep = m["iou_m"], m["match"], m["keep"]

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


def _dice_for_gts(preds: list, gts: list, match: dict) -> list:
    """逐 GT 的 Dice(官方口径:匹配成功取实际 Dice,未匹配取 0)。

    🔴 只能在扫参进程内算(sidecar 行不落 mask,事后无从计算):Dice 须按
    GT mask 与匹配预测 mask 直算,不能从 IoU 反推(2IoU/(1+IoU) 是数学等价
    变换但依赖"两者都是二值集合"前提——反推公式在 IoU 舍入后的误差直接进
    Dice;直算是官方口径的字面实现)。
    """
    out = []
    for j, g in enumerate(gts):
        i = int(match["gt_to_pred"][j])
        if i < 0:
            out.append(0.0)
            continue
        p_mask = np.asarray(preds[i]["mask"]) != 0
        g_mask = np.asarray(g["mask"]) != 0
        if g.get("annotated_frames") is not None:
            # 动态:各标注帧帧级 Dice 的算术平均(与官方 IoU 同结构;
            # 预测帧全空 → 该帧 Dice=0,自然覆盖)。
            vals = []
            for f in g["annotated_frames"]:
                pm, gm = p_mask[f], g_mask[f]
                denom = pm.sum() + gm.sum()
                vals.append(2.0 * float((pm & gm).sum()) / denom if denom else 0.0)
            out.append(float(np.mean(vals)) if vals else 0.0)
        else:
            denom = p_mask.sum() + g_mask.sum()
            out.append(2.0 * float((p_mask & g_mask).sum()) / denom if denom else 0.0)
    return out


def _combo_rows(case_result, post, gts: list, is_dynamic: bool,
                combos: list) -> list:
    """单 case 的扫参核心:一次前向产物 × 参数组合列表 → 逐组合 case 行。

    前向结果(case_result)是本函数入参、组合循环只跑后处理段(resize/split/
    match)——模型前向(全量 54 分钟的大头)只跑一次,组合数只线性放大 CPU 段。
    🔴 内存边界(B路审查P2订正):①`preds` 在组合循环末尾显式 del——下一组合的
    post() 执行期间上一组合的 N 份全尺寸实例 mask 仍被引用,不 del 峰值 2×;
    ②动态例 soft 驻留估算按**原图分辨率**算(resize 后),750 帧×802×1442×4B
    ≈3.47GB/例,不是 512² 的 0.75GB;③case 级生命周期(循环完即弃),不跨 case
    缓存。另外 post() 返回的动态实例是逐实例 restore 的全尺寸 mask(N 份同时
    驻留,生产 _build_entries_and_save_masks 才是逐个用完即弃)。
    """
    rows = []
    for combo in combos:
        kw = {}
        if combo.get("thr") is not None:
            kw["seg_threshold"] = combo["thr"]
        if not is_dynamic and combo.get("min_area") is not None:
            kw["min_area_ratio"] = combo["min_area"]
        # 阶段1三处联动之三:max_nodules 透传 post(静/动同)——显式传参优先于
        # env,扫参确定性;不传(None)时生产 env/默认参路径不变。
        if combo.get("max_nodules") is not None:
            kw["max_nodules"] = combo["max_nodules"]
        if is_dynamic:
            preds, _n_total = post(case_result, **kw)
        else:
            preds = post(case_result, **kw)
        row = {"combo": combo, "anomaly": None, "n_pred": len(preds),
               "n_gt": len(gts), "preds": [], "gts": [], "dices": []}
        try:
            m = _match_row(preds, gts)
            if "anomaly" in m:
                row["anomaly"] = m["anomaly"]
                rows.append(row)
                continue
            iou_m, match, keep = m["iou_m"], m["match"], m["keep"]
            for i, p in enumerate(preds):
                best = float(iou_m[i].max()) if len(gts) else 0.0
                row["preds"].append({
                    "confidence": round(float(p["confidence"]), 6),
                    "area": p["area"], "best_iou": round(best, 6),
                    "matched": bool(match["pred_to_gt"][i] >= 0),
                    "dedup_keep": bool(keep[i]),
                })
            dices = _dice_for_gts(preds, gts, match)
            for j, g in enumerate(gts):
                # 🔴 max_iou/area_ratio 必须进行(B路审查P1):analyze_diag_sidecar
                # 对缺键用 0.0 兜底——匹配率 100% 的病例会被全计入 miss 桶,
                # 归因报告指向完全错误的药方。与 diagnose_one_case 的 gts 字段对齐。
                row["gts"].append({
                    "is_main": g["is_main"],
                    "matched": bool(match["gt_to_pred"][j] >= 0),
                    "max_iou": round(float(iou_m[:, j].max()) if len(preds) else 0.0, 6),
                    "area_ratio": round(float(g["area_ratio"]), 8),
                })
            row["dices"] = [round(d, 6) for d in dices]
        finally:
            del preds  # 🔴 下一组合 post() 前释放 N 份全尺寸 mask(防峰值 2×)
        rows.append(row)
    return rows


def _parse_sweep(specs: list) -> list:
    """`key=v1,v2,v3` 形式 → 组合列表(笛卡尔积,保序)。

    keys: `thr_static`/`thr_dynamic`(分割阈值)/`min_area`(静态最小面积占比,
    仅静态路径消费)/`max_nodules`(预测结节数上限,静/动两池都消费)。
    交叉维度 = 静态阈值×min_area×max_nodules(静态池)/动态阈值×max_nodules
    (动态池,min_area 不扫——生产动态钉死 DEFAULT)。实现上为简化:统一笛卡尔积,
    动态侧忽略 min_area 维度(`_combo_rows` 按分支消费),组合数会略多于理论必要,
    CPU 段每组合 ~秒级,可接受。

    🔴 max_nodules 是 int 轴(阶段1三处联动之一):float 解析会产 20.0 →
    `nodules[:20.0]` TypeError,按键走 int 转换;非整数值入口报错(扫参拼错
    20.5 应该炸而不是静默回默认)。
    """
    import itertools
    # 🔴 白名单校验(B路审查P1):未知键(如 thr= 漏写 _static)在派生后两侧皆空,
    # 会静默跑完全量 54 分钟推理再产出空报告——键名错误必须在入口就报。
    _ALLOWED = {"thr_static", "thr_dynamic", "min_area", "max_nodules"}
    _INT_KEYS = {"max_nodules"}
    axes = []
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"--sweep 需要 key=v1,v2,v3 形式,收到 {spec!r}")
        key, vals = spec.split("=", 1)
        key = key.strip()
        if key not in _ALLOWED:
            raise ValueError(
                f"--sweep 未知键 {key!r}(允许: {sorted(_ALLOWED)})——"
                f"键名拼错会让派生组合为空、白跑全量推理后产出空报告")
        parsed = []
        for v in vals.split(","):
            v = v.strip()
            if not v:
                continue
            if key in _INT_KEYS:
                iv = int(v)          # 非整数串直接 ValueError(入口报,好过下游炸)
                if iv < 1:
                    raise ValueError(
                        f"--sweep {key} 须为正整数,收到 {iv}")
                parsed.append(iv)
            else:
                parsed.append(float(v))
        if parsed:
            axes.append((key, parsed))
    if not axes:
        return []
    keys = [k for k, _ in axes]
    combos = []
    for values in itertools.product(*[v for _, v in axes]):
        combos.append(dict(zip(keys, values)))
    return combos


def run_sweep(args) -> dict:
    """件套②扫参主流程:一次前向 × 参数组合,产出逐组合静/动/合并池指标。

    每组合输出:匹配率、平均 Dice(未匹配计 0,官方口径)、ap_current、
    n_pred。与 run_diagnosis 共用建模/val 切分/匹配逻辑,但 case 行按组合组织
    (sidecar 每行带 combo 字段),便于横向对拍。
    """
    import torch
    from src.inference.model_runner import build_model, load_config

    combos = _parse_sweep(args.sweep)
    if not combos:
        raise ValueError("扫参模式需要至少一个 --sweep key=v1,v2,v3")
    # 派生:静态/动态各自适用的组合(动态忽略 min_area)。
    # 静态必须有 thr_static 才扫(纯 min_area 无阈值轴时不扫静态——min_area
    # 是阈值的伴随参数,单独扫它对官方阈值 0.5 的基线没有对拍意义)。
    # 🔴 阶段1三处联动之二:max_nodules 必须同时进两侧派生白名单——漏掉则
    # 双臂同参(20 vs 20),方向A被假阴性杀掉(B路审查 P1 同型)。静/动两池
    # 都消费它(split_nodules 通用参数)。
    # 🔴 动态侧须 thr_dynamic 在场才吸收 max_nodules(1-2路审查 P1,实测两种
    # 失败形态:①裸 max_nodules 轴绕过下方派生空护栏,静态池静默缺席出半份
    # 报告;②纯静态扫参(thr_static+max_nodules)被拖去跑全帧动态推理两遍,
    # 54 分钟级成本)。对称语义:动态池被扫 ⇔ 显式带了 thr_dynamic 轴;
    # 要扫动态 max_nodules 就显式传 thr_dynamic(哪怕单值)。
    combos_static = []
    for c in combos:
        if "thr_static" not in c:
            continue
        d = {k: v for k, v in c.items()
             if k in ("thr_static", "min_area", "max_nodules")}
        d["thr"] = d.pop("thr_static")
        if d not in combos_static:
            combos_static.append(d)
    combos_dynamic = []
    for c in combos:
        if "thr_dynamic" not in c:
            continue
        d = {"thr": c["thr_dynamic"]}
        if "max_nodules" in c:
            d["max_nodules"] = c["max_nodules"]
        if d not in combos_dynamic:
            combos_dynamic.append(d)
    # 🔴 派生空护栏(B路审查P1;阶段1实测加固:裸 max_nodules 轴在此被拦):
    # --sweep min_area=... / max_nodules=... 这类"无阈值轴"组合派生后
    # 两侧皆空,白跑全量推理后产出空报告——入口即拦。min_area 是静态阈值的
    # 伴随参数;max_nodules 是阈值的伴随参数(两侧都是),单扫它对默认阈值
    # 基线没有对拍意义(饱和归因走 run_diagnosis + analyze_diag_sidecar)。
    if not combos_static and not combos_dynamic:
        raise ValueError(
            f"扫参组合派生为空(输入 {args.sweep}):至少需要一个 "
            "thr_static=... 或 thr_dynamic=... 轴;min_area/max_nodules 是"
            "阈值轴的伴随参数")
    _safe_print(f"[sweep] 静态 {len(combos_static)} 组合 × 动态 "
                f"{len(combos_dynamic)} 组合")

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"权重不存在: {args.weights}")
    _safe_print(f"[sweep] weights = {os.path.abspath(args.weights)}")
    config = load_config(args.config)
    device = torch.device(
        config["environment"]["device"] if torch.cuda.is_available() else "cpu")
    model = build_model(config, args.weights, device)
    from src.inference.calibration import TemperatureScaler
    calibrator = TemperatureScaler.from_file(
        config.get("calibration", {}).get("param_file",
                                          "checkpoints/calibration.json"))

    seed = args.seed if args.seed is not None else int(
        config.get("environment", {}).get("seed", 42))
    _, val_ds = build_official_datasets(
        args.data_root, args.xlsx, img_size=int(config["data"]["img_size"]),
        val_frac=args.val_frac, seed=seed, samples_pkl=args.samples_pkl)
    samples = val_ds.samples
    cases, n_no_mask = group_cases(samples)
    if args.limit:
        cases = cases[: args.limit]
    _safe_print(f"[sweep] val 样本 {len(samples)} → {len(cases)} case"
                f"(limit={args.limit or '全量'})")

    os.makedirs(args.log_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    sidecar_path = os.path.join(args.log_dir, f"sweep_{ts}.cases.jsonl")
    report_path = os.path.join(args.log_dir, f"sweep_{ts}.json")
    # 池化结构:{("static"|"dynamic"|None, combo_key): rows}
    all_rows = {}
    t0 = time.time()
    n_anomaly = 0
    with open(sidecar_path, "w", encoding="utf-8") as sf:
        for k, case in enumerate(cases):
            gts = load_gts_for_case(case["gts"])
            bad = [g["error"] for g in gts if "error" in g]
            if bad:
                n_anomaly += 1
                sf.write(json.dumps({"case_key": case["img_path"],
                                     "anomaly": f"GT 读取异常: {bad[0]}"},
                                    ensure_ascii=False) + "\n")
                continue
            gts = [g for g in gts if "error" not in g]
            # B路审查P3:该池无组合时跳过整个 case 的推理(动态推理是全跑里最贵
            # 的一档,无组合还跑 = 纯浪费)。
            has_combo = (combos_dynamic if case["is_dynamic"]
                         else combos_static)
            if not has_combo:
                continue
            try:
                if case["is_dynamic"]:
                    case_result, post = run_dynamic_inference(
                        model, config, calibrator, args.precision, device,
                        case["img_path"], args.max_frames)
                    rows = _combo_rows(case_result, post, gts, True,
                                       combos_dynamic)
                else:
                    case_result, post = run_static_inference(
                        model, config, calibrator, args.precision, device,
                        case["img_path"])
                    rows = _combo_rows(case_result, post, gts, False,
                                       combos_static)
                # 🔴 用完即弃:动态例 resize 回原图后的 soft ≈ 帧数×H×W×4B
                # (750×802×1442 例 ≈3.47GB,B路审查订正——不是 512² 的 0.75GB),
                # 勿跨 case。
                del case_result
            except Exception as e:  # noqa: BLE001
                n_anomaly += 1
                sf.write(json.dumps({"case_key": case["img_path"],
                                     "anomaly": f"推理失败: {type(e).__name__}: {e}"},
                                    ensure_ascii=False) + "\n")
                continue
            for r in rows:
                if r.get("anomaly"):
                    n_anomaly += 1
                r["case_key"] = case["img_path"]
                r["is_dynamic"] = case["is_dynamic"]
                sf.write(json.dumps(r, ensure_ascii=False) + "\n")
                key = json.dumps(r["combo"], sort_keys=True)
                all_rows.setdefault((case["is_dynamic"], key), []).append(r)
            if (k + 1) % 20 == 0 or k + 1 == len(cases):
                _safe_print(f"[sweep] {k + 1}/{len(cases)} case 完成"
                            f"(anomaly {n_anomaly})")
            sf.flush()
    runtime_s = round(time.time() - t0, 1)

    # 池化:每组合 × (静/动) 的匹配率/平均Dice/AP;合并池 = 静动行并集。
    from scripts.detection_match import average_precision
    results = []
    for (is_dyn, key), rows in sorted(all_rows.items()):
        ok = [r for r in rows if not r.get("anomaly")]
        preds_flat, dices_flat, n_gt = [], [], 0
        for r in ok:
            for p in r["preds"]:
                preds_flat.append(p)
            dices_flat.extend(r["dices"])
            n_gt += r["n_gt"]
        labels = [int(p["matched"]) for p in preds_flat]
        confs = [p["confidence"] for p in preds_flat]
        results.append({
            "is_dynamic": is_dyn, "combo": json.loads(key),
            "n_cases": len(rows), "n_gt": n_gt,
            "n_tp": sum(labels),
            "match_rate": (sum(labels) / n_gt) if n_gt else 0.0,
            "mean_dice": (float(np.mean(dices_flat))
                          if dices_flat else 0.0),  # 未匹配已在 _dice_for_gts 计 0
            "ap_current": average_precision(labels, confs, n_gt),
            "n_pred": len(preds_flat),
        })
    # 合并池(静+动同组合并存时):按 combo 键聚合。动态组合键含 thr_dynamic、
    # 静态含 thr_static/min_area,天然不重合——合并只在"两侧都扫了同一 thr 值"
    # 的人工对拍场景下由使用者自行汇总,报告不强行合并(避免口径混淆)。
    report = {
        "meta": {
            "weights": os.path.abspath(args.weights), "config": args.config,
            "seed": seed, "val_frac": args.val_frac,
            "max_frames": args.max_frames, "precision": args.precision,
            "n_samples": len(samples), "n_cases": len(cases),
            "n_no_mask_samples": n_no_mask, "runtime_s": runtime_s,
            "sidecar": sidecar_path,
            "sweep_args": list(args.sweep),
            "combos_static": combos_static, "combos_dynamic": combos_dynamic,
            "caveats": [
                "mean_dice = 官方口径(匹配成功取实际Dice,未匹配取0,全部金标准"
                "结节的平均);Dice 只能在扫参进程内算(sidecar 不落 mask)",
                "动态 min_area 不扫(生产动态钉死 DEFAULT,镜像同口径);"
                "静态 min_area 只作用于静态池",
                "max_nodules 静/动两池都扫(生产 None→env→DEFAULT 同源),"
                "min_area 只静态——两轴池覆盖不对称是刻意的",
                "val 只含有 ROI 标注患者,阴性 FP 不在池(与 run_diagnosis 同盲区)",
                "🔴 选阈值须看 match_rate 与 mean_dice 的联合,不能只看其一:"
                "降阈值让临界带病例越过 0.5,同时让已匹配病例 IoU 下滑跌出 0.5",
            ],
        },
        "results": results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _safe_print("\n===== 扫参摘要 =====")
    for r in results:
        _safe_print(
            f"[{'dyn' if r['is_dynamic'] else 'sta'}] {r['combo']} | "
            f"n_gt={r['n_gt']} TP={r['n_tp']} 匹配率={r['match_rate']:.4f} | "
            f"平均Dice={r['mean_dice']:.4f} | AP={r['ap_current']:.4f} "
            f"n_pred={r['n_pred']}")
    _safe_print(f"[sweep] 报告: {report_path}\n[sweep] sidecar: {sidecar_path}")
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
    # 件套②(2026-09-10)扫参模式:key=v1,v2,v3 形式,可重复传。keys:
    #   thr_static / thr_dynamic(分割二值化阈值)/ min_area(静态最小面积占比,
    #   只作用于静态池)/ max_nodules(预测结节数上限,静/动两池都消费,但
    #   须伴随同侧 thr 轴才扫该池)。给了 --sweep 就走 run_sweep(一次前向多
    #   参数),否则走原 run_diagnosis(单参数全量诊断)。
    ap.add_argument("--sweep", action="append", default=None,
                    help="例:--sweep thr_static=0.30,0.35,0.40,0.50 "
                         "--sweep thr_dynamic=0.25,0.30,0.35 "
                         "--sweep min_area=2e-5,5e-5,1e-4 "
                         "--sweep max_nodules=20,30,50")
    args = ap.parse_args()
    # 逐 case 完整行始终进 sidecar JSONL(崩溃保数据);--save-per-case 曾计划
    # 把行嵌进主报告,但会撑大 JSON 且与 sidecar 重复,已砍,参数不再收。
    if args.sweep:
        run_sweep(args)
    else:
        run_diagnosis(args)


if __name__ == "__main__":
    main()
