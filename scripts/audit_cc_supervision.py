# -*- coding: utf-8 -*-
"""C0 监督审计(plan 任务 C 开训前置,零训练,2026-09-18 新架构线)。

职责(plan C0 预注册):对**原始 mask 与训练分辨率下的每样本**做连通域
K 分布审计,回答"CC-DiceCE 的实例监督前提是否成立":

  1. K 分布:每样本(按 img_path+frame_index 聚合同图 GT)连通域数直方图;
     大量 K=1 时 CC 项退化为"局部归一化的单实例 DiceCE"(仍有效但**不得
     宣称同图多实例平权**——这是 plan 明文的防夸大条款);
  2. 同图独立 GT 数:同一 img_path(+frame)上几个**已标注**的不同 mask
     文件聚合(=同图关联独立 GT 数;M+病灶N 同帧并集才是 CC 的完整监督);
  3. 单病灶占比:各 mask 是否单连通域(病灶mask 天然单实例 vs 碎片);
  4. 小病灶消失比:resize 到训练分辨率后前景像素消失的样本比例
     (INTER_NEAREST 下 512² 网格对小病灶的物理下限);
  5. 背景侵蚀比:同图多 GT 时,mean_k(|other GT 像素| / |本实例像素|)
     ——每个单病灶样本里"被当背景的错误负监督"相对本实例面积的倍数
     (PU 背景降权 ω 的证据源;🔴 P2 订正 2026-09-18 Gate2 审查:旧口径
     erosion_pos/erosion_union 数学上恒等于 1-1/K,只由标注个数决定、
     与像素分布无关,不度量它声称的东西,已重定义)。未标注病灶不可测,
     报已标注部分的下界。

数据来源:samples.pkl(build_official_samples 产物,字段 img_path/
mask_path/frame_index/is_dynamic/patient_key/labels.is_main)——**不重复
发明样本枚举**,与训练同一份清单。

用法(云端,官方数据禁本地):
  PYTHONPATH=/2026aicompetition/workspace/pylibs python scripts/audit_cc_supervision.py \
      --samples-pkl /2026aicompetition/workspace/cache/samples.pkl \
      --img-size 512 --out /root/cloud_train_pkg/cc_audit.json

输出:JSON 报告(K 直方图/独立 GT 数直方图/消失比/背景侵蚀比/分位表),
判读写在报告的 "verdict" 键,不 print 大段文字(stdout 只走进度行)。
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import ndimage  # noqa: E402

from src.dataset.official_loader import (  # noqa: E402
    _read_nifti_binmasks,
    resize_mask_to,
)


def _load_samples(samples_pkl: str):
    """读 preextract_dynamic 的落盘 payload(🔴 是 dict 不是 list:
    {"samples": [...], "_meta": {...}};旧版按裸 list 迭代会把 dict 键名
    当样本,str 上 .get 直接 AttributeError——C0 首跑实崩)。"""
    import pickle
    with open(samples_pkl, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        meta = payload.get("_meta") or payload.get("CACHE_META") or {}
        if meta.get("partial"):
            raise RuntimeError(
                "[audit] 该 samples.pkl 是 --limit 冒烟残件(_meta.partial=True),"
                "覆盖不全,禁止用于 C0 审计——用全量预抽取产物重跑。")
        return payload["samples"], meta
    return payload, {}


def _static_mask(mask_path: str) -> np.ndarray:
    """静态 mask:(H,W) uint8 二值;读失败返回 None(计入不可读,不剔数据)。"""
    frames, _ = _read_nifti_binmasks(mask_path)
    if not frames:
        return None
    return frames[0]


def _dynamic_mask_frame(mask_path: str, frame_index: int):
    """动态 mask 指定原始帧号:(H,W) uint8 或 None。"""
    frames, idxs = _read_nifti_binmasks(mask_path)
    if not frames:
        return None, None
    # loader 同口径:idxs 含 frame_index 用之,否则按位置兜底
    pos = idxs.index(frame_index) if frame_index in idxs else frame_index
    if pos >= len(frames):
        return None, None
    return frames[pos], len(frames)


def audit_cc_supervision(samples: list, img_size: int) -> dict:
    """主审计。samples = build_official_samples 清单(同一 img 聚合后)。

    返回 dict(落 JSON):
      k_hist: 同图聚合 GT 并集的连通域数直方图 {K: n}
      gt_count_hist: 同图独立已标注 GT 数直方图 {G: n}
      single_cc_frac: 单病灶 mask 中单连通域占比
      vanish: 小病灶 resize 后消失统计(含分母)
      bg_erosion: mean_k(|other|/|inst|) 平均相对倍数(同图多 GT)
      n_img / n_unreadable / n_dynamic_frame: 口径计数
    """
    # 同图聚合:静态 = img_path;动态 = (img_path, frame_index)
    by_image = defaultdict(list)   # key -> [(mask_path, roi_is_main)]
    n_unreadable = 0
    n_dynamic_frame = 0
    for s in samples:
        key = (s["img_path"], s.get("frame_index")) if s.get("is_dynamic") \
            else (s["img_path"], None)
        if s.get("is_dynamic"):
            n_dynamic_frame += 1
        by_image[key].append((s["mask_path"],
                              bool(s["labels"].get("is_main"))))

    k_hist = Counter()
    gt_count_hist = Counter()
    single_cc_total = single_cc_single = 0
    vanish_n = vanish_total = 0
    erosion_pos = erosion_union = 0
    multi_gt_images = 0
    mask_cache = {}

    items = sorted(by_image.items(), key=lambda kv: (kv[0][0], kv[0][1] or -1))
    for i, (key, gt_list) in enumerate(items):
        if i % 200 == 0:
            print(f"  [audit] {i}/{len(items)} 图聚合单元", flush=True)
        img_path, frame_index = key
        masks = []
        for mask_path, _is_main in gt_list:
            if mask_path is None:
                continue
            cache_key = (mask_path, frame_index)
            if cache_key not in mask_cache:
                if frame_index is None:
                    m = _static_mask(mask_path)
                else:
                    m, _ = _dynamic_mask_frame(mask_path, frame_index)
                mask_cache[cache_key] = m
            m = mask_cache[cache_key]
            if m is None:
                n_unreadable += 1
                continue
            masks.append(m)

        if not masks:
            k_hist[0] += 1
            gt_count_hist[len(masks)] += 1
            continue

        gt_count_hist[len(masks)] += 1
        # 同图 GT 并集(训练时该图的完整前景监督应长这样)
        union = np.zeros_like(masks[0], dtype=np.uint8)
        single_m = []
        for m in masks:
            single_m.append(m)
            union |= m
        # 1) K 分布(并集)
        _, n_cc = ndimage.label(union > 0.5)
        k_hist[int(n_cc)] += 1
        # 2) 单病灶 mask 单连通域占比
        for m in single_m:
            single_cc_total += 1
            _, c = ndimage.label(m > 0.5)
            if c == 1:
                single_cc_single += 1
        # 3) 小病灶 resize 消失(单病灶逐个;只看非零 mask)
        for m in single_m:
            if not m.any():
                continue
            r = resize_mask_to(m, img_size)
            vanish_total += 1
            if not r.any():
                vanish_n += 1
        # 4) 背景侵蚀(P2 重定义):mean_k(|other| / |inst|) 相对倍数
        if len(masks) > 1:
            multi_gt_images += 1
            for m in masks:
                inst_area = int(m.astype(np.int64).sum())
                if inst_area == 0:
                    continue
                other_area = int((union & ~m.astype(bool)).sum())
                erosion_pos += other_area / inst_area
                erosion_union += 1
            # erosion_union 改为"计入的实例数",frac = 平均相对倍数

    n_img = len(items)
    return {
        "n_image_units": n_img,
        "n_dynamic_frame_units": n_dynamic_frame,
        "n_unreadable_mask": n_unreadable,
        "k_hist": dict(sorted(k_hist.items())),
        "k_ge2_frac": float(sum(v for k, v in k_hist.items() if k >= 2)
                             / max(1, n_img)),
        "gt_count_hist": dict(sorted(gt_count_hist.items())),
        "multi_gt_image_frac": float(multi_gt_images / max(1, n_img)),
        "single_cc_frac": float(single_cc_single / max(1, single_cc_total)),
        "vanish_frac": float(vanish_n / max(1, vanish_total)),
        "vanish_n": vanish_n,
        "vanish_total": vanish_total,
        "bg_erosion_frac": float(erosion_pos / max(1, erosion_union)),
        "img_size": img_size,
    }


def main():
    parser = argparse.ArgumentParser(description="C0 监督审计(CC-DiceCE 前置)")
    parser.add_argument("--samples-pkl", required=True,
                        help="build_official_samples 产物 samples.pkl")
    parser.add_argument("--img-size", type=int, default=512,
                        help="训练分辨率(默认 512,与 config.data.img_size 同源)")
    parser.add_argument("--out", required=True, help="JSON 报告输出路径")
    parser.add_argument("--limit", type=int, default=0,
                        help="只审前 N 个图聚合单元(0=全量;冒烟用)")
    args = parser.parse_args()

    samples, meta = _load_samples(args.samples_pkl)
    print(f"[audit] samples={len(samples)} img_size={args.img_size}"
          + (f" (pkl meta: n_total={meta.get('n_total')}, "
             f"n_dynamic={meta.get('n_dynamic')})" if meta else ""))
    if args.limit:
        # 截断但保持图聚合完整性:取前 limit 个聚合单元对应的样本
        seen = set()
        kept = []
        for s in samples:
            key = (s["img_path"], s.get("frame_index")) if s.get("is_dynamic") \
                else (s["img_path"], None)
            if key not in seen:
                if len(seen) >= args.limit:
                    break
                seen.add(key)
            kept.append(s)
        samples = kept
        print(f"[audit] --limit {args.limit}: 截断为 {len(samples)} 样本")

    report = audit_cc_supervision(samples, args.img_size)

    # 判读(预注册阈值:plan C0 的"宣称条款",不是开训门)
    k_ge2 = report["k_ge2_frac"]
    vanish = report["vanish_frac"]
    erosion = report["bg_erosion_frac"]
    report["verdict"] = {
        "cc_effective_note": (
            f"K≥2 图占比 {k_ge2:.1%}——"
            + ("多实例平权成立" if k_ge2 >= 0.20
               else "大量 K=1,CC=局部归一化 DiceCE,不得宣称多实例平权")),
        "bg_erosion_note": (
            f"背景侵蚀平均相对倍数 {erosion:.2f}×"
            + "(其他已标注 GT/本实例面积;PU ω 降权的已标注下界,未标注不可测)"),
        "vanish_note": (
            f"小病灶 resize 消失比 {vanish:.2%}——"
            + ("几乎无消失,512² 分辨率对小病灶安全" if vanish < 0.01
               else "有消失,建议评估更高分辨率或保留原图")),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[audit] 报告 → {args.out}")
    print(f"[audit] K直方图={report['k_hist']} "
          f"K≥2占比={k_ge2:.1%} 消失比={report['vanish_frac']:.2%} "
          f"背景侵蚀={report['bg_erosion_frac']:.2%}")


if __name__ == "__main__":
    main()
