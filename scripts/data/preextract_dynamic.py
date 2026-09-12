#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""动态 .nii.gz 预抽取 —— 把训练真正消费的那几帧一次性导成 PNG,消灭每 epoch 的重复全量解压。

## 为什么必须做(2026-08-15 云端实测锁定的根因)

`OfficialThyroidDataset.__getitem__` 的动态分支对**每个样本**做两次全量解压:

    img  = _read_nifti_frame_gray(img_path, frame_index)      # 全解影像体积
    mask = self._dynamic_mask_frame(mask_path, frame_index)   # 全解 mask 体积 + 逐帧 >0

`load_nifti_frames` 内部是 `np.asarray(img.dataobj)` —— **一次性全量解码,无 lazy 切片**。
取 1 帧要解 2 个完整体积。云端实测单次 1.81s / 65MB 文件 → 258MB 数组;mask 文件虽只
0.2-0.3MB(二值数据 gzip 压缩比接近 DEFLATE 理论上限 1032:1),**解压后与影像同体素数**,
一样是 258MB 量级。

后果三连:
  - 每 epoch 约 1400 个动态样本 × ~4s ≈ **42 分钟**,35 epoch ≈ 24 小时纯解压;
  - `num_workers=12` 时瞬时驻留 ~700MB × 12 ≈ **8.4GB**,64G 机器上是明确的内存压力路径;
  - 该压力路径正是首轮真训里 `invalid block type` 与步耗时抖动 22 倍的来源
    (B 假设:我方读法炸了,非文件损坏)。

## 为什么缓存与现状**数值等价**(不是近似,不掉精度)

`__getitem__` 读完帧后立刻走 `_resize_img`/`_resize_mask` 缩到 `img_size`(默认 512),
再做增强/归一化。也就是说**训练消费的从来就只有 512×512 uint8**。本脚本导出的正是
resize 之后、增强之前的那一张,插值方式与 loader 逐位一致(img 用 cv2 默认双线性、
mask 用 INTER_NEAREST)。故缓存不是有损近似,是把同一次计算提前算好存起来。

体积代价:约 1400 样本 × (img+msk) 512×512 uint8,PNG 落盘约 100MB 出头。

## 产物

    <cache-dir>/
        samples.pkl          # 完整样本清单(静态+动态),动态项已带 cache_img/cache_msk 绝对路径
        frames/<key>_img.png # 512×512 uint8 灰度
        frames/<key>_msk.png # 512×512 uint8,值域 {0,255}(loader 侧统一 `>0` 二值化)
        preextract_report.txt

`samples.pkl` 一并解决 build 阶段的 23 分钟:训练脚本传 `--samples-pkl` 即可秒开,
不再重扫 xlsx、不再为定位非零帧解 343 个 mask。

## 幂等 / 可续跑

已存在且非空的 PNG 直接跳过,中途挂掉重跑不会重做已完成的文件。故障文件记进 report,
对应样本从 samples.pkl 中剔除(宁可少几个样本,不可让训练在第 30 个 epoch 才炸)。

## 用法

    python scripts/data/preextract_dynamic.py \
        --data-root /2026aicompetition/datasets/training/annotation \
        --xlsx <标注.xlsx> \
        --cache-dir /2026aicompetition/workspace/cache

    # 先小规模验证管道(只处理前 3 个动态患者,约 1 分钟):
    python scripts/data/preextract_dynamic.py ... --limit 3

⚠️ **`--img-size` 必须与训练时的 `img_size` 一致**(两侧默认都是 512)。不一致不会报错,
loader 会把缓存图再 resize 一次,等于降采样后又插值回去 —— 静默掉画质。脚本把实际值
写进 samples.pkl 的 `_meta`,loader 加载时校验并告警。
"""
import argparse
import hashlib
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import cv2
except ImportError:  # pragma: no cover - 环境问题,fail-fast 好过跑一半崩
    print("[FATAL] 缺 opencv-python: pip install -r requirements.txt")
    raise

from src.dataset.official_loader import (  # noqa: E402
    build_official_samples,
    _read_nifti_binmasks,
    _safe_print,
    # 🔴 resize 必须复用 loader 的实现,不得在本脚本另写一份:预抽取与 __getitem__
    # 现算是同一次计算的两个时点,两份代码将来任一方改插值方式就会静默漂移
    # (缓存图与实时图不同,loss 照常降、指标只是略低,本地查不出)。
    resize_img_to as _resize_img,
    resize_mask_to as _resize_mask,
)
from src.inference.nifti_io import load_nifti_frames  # noqa: E402

CACHE_META_KEY = "_meta"


def frame_key(img_path: str, mask_path: str, frame_index: int) -> str:
    """(影像, mask, 帧号) → 文件名安全的短键。

    三者全进哈希:同一患者的不同病灶共用影像但 mask 不同,只哈希影像会撞车、
    让两个病灶读到同一张 mask(不报错,标签与像素静默错配)。
    """
    raw = f"{img_path}|{mask_path}|{frame_index}".encode("utf-8", "replace")
    return hashlib.sha1(raw).hexdigest()[:20]


def _write_png(path: str, arr: np.ndarray) -> bool:
    """写 PNG。cv2.imwrite 对非 ASCII 路径在 Windows 会静默失败,故用 imencode + 二进制写。"""
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        return False
    tmp = path + ".part"          # 先写临时名再改名:中途被杀不会留下半张图冒充完成品
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
    os.replace(tmp, path)
    return True


def _probe_png_size(path: str):
    """读 PNG 头取 (h, w)(只解析 IHDR,不解码像素,微秒级)。失败返回 None。

    用途:A路审查的分辨率护栏——PNG 尺寸不可伪造(区别于 samples.pkl 的
    _meta 自称值),抽检首个"已存在"缓存帧的实际尺寸与本次 --img-size 比对。
    """
    try:
        with open(path, "rb") as f:
            head = f.read(33)     # 8B PNG 签名 + 4B 长度 + 4B "IHDR" + 13B IHDR 数据
        if len(head) < 33 or head[12:16] != b"IHDR":
            return None
        w = int.from_bytes(head[16:20], "big")
        h = int.from_bytes(head[20:24], "big")
        return (h, w)
    except OSError:
        return None


def _process_one_file(job):
    """worker:处理**一个**影像文件组(解影像一次 + 按组解各 mask + 写 PNG)。

    纯函数化(2026-09-11 并行化改造):入参全部为可 pickle 的标量/列表,返回
    可 pickle 的结果 dict —— 主进程据此回填 samples 与 failures。**不碰
    samples.pkl**(全局唯一写出点在主进程,杜绝并发写竞态)。

    job: {"img_path", "entries": [{k, fi, fm, mask_path, frame_index, s_idx}],
          "img_size"}
    返回: {"s_idx": {"cache_img": ..., "cache_msk": ...},  # 按 s_idx 回填;
                     失败样本两值为 None
           "failures": [...], "done_frames": N,
           "probe_hw": (h, w) | None}   # 首个"已存在"PNG 的实际尺寸(分辨率抽检)
    """
    img_path = job["img_path"]
    img_size = job["img_size"]
    entries = job["entries"]

    result = {"fills": {}, "failures": [], "done_frames": 0, "probe_hw": None,
              "skipped_in_job": 0}

    # ---- 先看这组是不是已经全导完了(续跑的关键) ----
    # 🔴 frame_key 不含 img_size(A路审查):同一 frames/ 目录里 512 与 640 的
    # PNG 键完全相同。换 --img-size 复用旧 cache-dir 时,这里的"已存在即跳过"
    # 会原样保留旧分辨率 PNG,而 samples.pkl 的 _meta 又写新值——下游
    # official_loader 的分辨率 raise 被整个绕过(meta 谎报)。故对首个"已存在"
    # 的帧抽检实际尺寸,尺寸≠本次 img_size 时把 probe_hw 报回主进程硬拦
    # (worker 不 SystemExit:池里的兄弟进程还有 .part 临时文件要清理,
    # 主进程收齐后统一 exit 才干净)。
    pending = []
    for e in entries:
        if (os.path.exists(e["fi"]) and os.path.getsize(e["fi"]) > 0
                and os.path.exists(e["fm"]) and os.path.getsize(e["fm"]) > 0):
            if result["probe_hw"] is None:
                result["probe_hw"] = _probe_png_size(e["fi"])
            # 🔴 已存在的缓存也要回填 fills(2026-09-11 测试抓出的真 bug):旧版串行
            # 在扫描阶段就写 s["cache_img"],并行版若只在"写出成功"时回填,幂等重跑
            # 会把全部已缓存样本当失败剔除(pkl 只剩静态样本,静默丢全部动态数据)。
            result["fills"][e["s_idx"]] = {"cache_img": e["fi"], "cache_msk": e["fm"]}
            result["skipped_in_job"] += 1
            continue
        pending.append(e)
    if not pending:
        return result

    # ---- 解影像(一次) ----
    try:
        frames, idxs, _, err, _ = load_nifti_frames(img_path, max_frames=-1)
        if err or not frames:
            raise RuntimeError(err or "load_nifti_frames 返回空")
    except Exception as e:                       # noqa: BLE001 - 逐文件降级,不中断全局
        result["failures"].append(f"IMG {img_path}: {e}")
        # 🔴 只剔 pending(本次要导出的),不清 entries 里已缓存样本(Gate2 审查 P0):
        # 旧版串行此分支只处理 pending;若遍历 entries 会把"磁盘已有缓存、可继续
        # 用"的样本覆盖成 None → 续跑场景撞上解码失败文件时静默丢已缓存样本。
        for e_ in pending:
            result["fills"][e_["s_idx"]] = {"cache_img": None, "cache_msk": None}
        return result
    pos_of = {f: p for p, f in enumerate(idxs)}

    # ---- 按 mask 分组解 mask(每个 mask 一次) ----
    by_mask: Dict[str, List[Dict]] = defaultdict(list)
    for e in pending:
        by_mask[e["mask_path"]].append(e)

    for mask_path, mentries in by_mask.items():
        try:
            binf, midxs = _read_nifti_binmasks(mask_path)
            if not binf:
                raise RuntimeError("mask 解码为空")
        except Exception as e:                   # noqa: BLE001
            result["failures"].append(f"MSK {mask_path}: {e}")
            for e_ in mentries:
                result["fills"][e_["s_idx"]] = {"cache_img": None, "cache_msk": None}
            continue
        mpos_of = {f: p for p, f in enumerate(midxs)}

        for e_ in mentries:
            fidx = int(e_["frame_index"])   # 动态样本必有帧号;int() 兼顾 numpy 整型
            ip = pos_of.get(fidx, fidx)
            mp = mpos_of.get(fidx, fidx)
            if not (0 <= ip < len(frames)) or not (0 <= mp < len(binf)):
                result["failures"].append(
                    f"FRM {img_path} frame={fidx}: 越界 img_pos={ip}/{len(frames)} "
                    f"msk_pos={mp}/{len(binf)}")
                result["fills"][e_["s_idx"]] = {"cache_img": None, "cache_msk": None}
                continue
            img = _resize_img(np.asarray(frames[ip]), img_size)
            # 0/1 → 0/255:PNG 里 0/1 肉眼不可查,排错时打开就是纯黑;
            # loader 侧统一 `>0`,两种取值都正确,但 0/255 可目视核对。
            msk = _resize_mask((np.asarray(binf[mp]) > 0).astype(np.uint8), img_size) * 255
            if _write_png(e_["fi"], img) and _write_png(e_["fm"], msk):
                result["done_frames"] += 1
                result["fills"][e_["s_idx"]] = {"cache_img": e_["fi"], "cache_msk": e_["fm"]}
            else:
                result["failures"].append(f"PNG 写失败 {e_['fi']}")
                result["fills"][e_["s_idx"]] = {"cache_img": None, "cache_msk": None}
        del binf        # mask 的 binf 是 astype 出来的**真数组**(非视图),及时放手

    del frames                                    # 尽快释放 258MB 级数组
    return result


def extract(data_root: str, xlsx: str, cache_dir: str, sheet: str,
            img_size: int, limit: int, workers: int = 4) -> int:
    frames_dir = os.path.join(cache_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    t0 = time.time()
    _safe_print(f"[1/3] 扫描标注 + 定位动态非零帧(要解 343 个 mask,预计 10-25 分钟)...")
    samples = build_official_samples(data_root, xlsx, sheet)
    n_dyn = sum(1 for s in samples if s["is_dynamic"])
    n_lesion = len({s["mask_path"] for s in samples if s["is_dynamic"]})
    _safe_print(
        f"      样本总数 {len(samples)} | 静态 {len(samples) - n_dyn} | "
        f"动态样本 {n_dyn} | 动态病灶(mask 文件数) {n_lesion} | 用时 {time.time() - t0:.0f}s"
    )

    # 按影像文件分组:同一个 .nii.gz 只解压一次,把它名下所有需要的帧一次导完。
    # 🔴 这是本脚本存在的核心 —— 逐样本解压的话总解压次数不降反升,等于没优化。
    by_img: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        if s["is_dynamic"]:
            by_img[s["img_path"]].append(s)

    if limit > 0:
        keep = sorted(by_img)[:limit]
        by_img = {k: by_img[k] for k in keep}
        _safe_print(f"      --limit {limit}: 只处理 {len(by_img)} 个动态影像文件")

    failures: List[str] = []
    done_frames = skipped = 0
    t1 = time.time()
    total_files = len(by_img)

    # ---- 组装 job:每个影像文件一个 job,entries 携带回填用 s_idx ----
    # s_idx = sample 在 samples 列表里的下标(稳定,不受分组排序影响)。
    s_index = {id(s): i for i, s in enumerate(samples)}
    jobs = []
    for img_path, group in sorted(by_img.items()):
        entries = []
        for s in group:
            k = frame_key(img_path, s["mask_path"], s["frame_index"])
            fi = os.path.join(frames_dir, f"{k}_img.png")
            fm = os.path.join(frames_dir, f"{k}_msk.png")
            entries.append({
                "k": k, "fi": fi, "fm": fm,
                "mask_path": s["mask_path"], "frame_index": s["frame_index"],
                "s_idx": s_index[id(s)],
            })
        jobs.append({"img_path": img_path, "entries": entries, "img_size": img_size})

    # ---- 并行/串行分派(2026-09-11 并行化) ----
    # --workers 1 = 原串行路径(in-process 逐 job 调 _process_one_file),
    # 不建进程池 —— 与旧版行为逐位一致,worker 内不依赖任何全局状态。
    # workers>1 走 ProcessPoolExecutor:每个 job 解压 ~700MB 峰值内存,
    # 64G 机器上 4-8 worker 安全(8×700MB≈5.6G,余量充足);
    # 🔴 不碰 GPU(纯 CPU 解压+插值),无显存问题。
    def _dispatch_all():
        if workers <= 1 or len(jobs) <= 1:
            for i, job in enumerate(jobs, 1):
                yield i, _process_one_file(job)
            return
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # map 保持顺序;chunksize=1 让大文件(几百帧)之间负载尽量均衡
            for i, res in enumerate(pool.map(_process_one_file, jobs, chunksize=1), 1):
                yield i, res

    # 已存在即跳过的帧数,由 worker 随结果回传(每 job 内首扫阶段计出,
    # 与旧版串行口径一致 = 本轮开始前就已缓存的帧数,不含本轮产量)。
    _skip_counts = 0

    _safe_print(
        f"[2/3] 导出帧(每个影像文件解压一次,共 {total_files} 个,"
        f"{'串行' if workers <= 1 else f'{workers} 进程并行'})...")
    for i, res in _dispatch_all():
        # 分辨率抽检硬拦(worker 报回的 PNG 实际尺寸;串行版语义与旧版一致)。
        # 🔴 先回填本 res 的 fills 再退出(Gate2 审查 P1-1c):fills 里含"已存在
        # 跳过"样本的缓存路径,丢弃它们等于把已缓存样本悬空——今天 SystemExit
        # 在写盘前只是碰巧无害,别把安全性押在这个实现细节上。
        if res["probe_hw"] is not None and res["probe_hw"] != (img_size, img_size):
            for s_idx, fill in res["fills"].items():
                samples[s_idx]["cache_img"] = fill["cache_img"]
                samples[s_idx]["cache_msk"] = fill["cache_msk"]
            raise SystemExit(
                f"🔴 frames/ 里已有 {res['probe_hw'][0]}x{res['probe_hw'][1]} 的缓存 PNG,"
                f"与本次 --img-size {img_size} 不符(frame_key 不含分辨率,旧帧被"
                f"原样保留且 meta 会谎报新值)。换分辨率必须用全新 --cache-dir。")
        for s_idx, fill in res["fills"].items():
            samples[s_idx]["cache_img"] = fill["cache_img"]
            samples[s_idx]["cache_msk"] = fill["cache_msk"]
        failures.extend(res["failures"])
        done_frames += res["done_frames"]
        _skip_counts += res.get("skipped_in_job", 0)
        if i % 10 == 0 or i == total_files:
            el = time.time() - t1
            eta = el / i * (total_files - i)
            _safe_print(
                f"      {i}/{total_files} 文件 | 已导 {done_frames} 帧 | "
                f"跳过 {_skip_counts} | 失败 {len(failures)} | "
                f"用时 {el:.0f}s | 预计剩余 {eta:.0f}s")
    skipped = _skip_counts   # 与旧版同口径:开跑前已存在的帧数(不含本轮产量)


    # ---- 落盘样本清单 ----
    _safe_print("[3/3] 写 samples.pkl ...")
    kept, dropped = [], 0
    for s in samples:
        if s["is_dynamic"] and not s.get("cache_img"):
            if limit > 0:                 # limit 模式下未处理的动态样本保留走原路径
                kept.append(s)
            else:
                dropped += 1
            continue
        kept.append(s)

    payload = {
        CACHE_META_KEY: {
            "img_size": img_size,
            "data_root": data_root,
            "xlsx": xlsx,
            "n_total": len(kept),
            "n_dynamic": sum(1 for s in kept if s["is_dynamic"]),
            "n_dynamic_lesions": n_lesion,
            "n_failures": len(failures),
            # 🔴 --limit 模式的 pkl 是**残件**:未处理的动态样本没有缓存、会回落到
            # 全量解压的原路径,拿它跑正式训练等于既慢又只覆盖了 N 个文件。
            # 打上标记,由 build_official_datasets 硬拒,别指望人记得住 docstring。
            "partial": bool(limit > 0),
        },
        "samples": kept,
    }
    with open(os.path.join(cache_dir, "samples.pkl"), "wb") as f:
        pickle.dump(payload, f, protocol=4)

    report = os.path.join(cache_dir, "preextract_report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"总样本(保留) {len(kept)}\n动态样本 {payload[CACHE_META_KEY]['n_dynamic']}\n")
        f.write(f"动态病灶(mask 文件) {n_lesion}\n已导帧 {done_frames}\n跳过(已存在) {skipped}\n")
        f.write(f"因故障剔除的动态样本 {dropped}\n失败明细 {len(failures)} 条:\n")
        f.writelines(x + "\n" for x in failures)

    _safe_print(
        f"[DONE] 导出 {done_frames} 帧 / 跳过 {skipped} / 失败 {len(failures)} / "
        f"剔除动态样本 {dropped} | 总用时 {time.time() - t0:.0f}s")
    _safe_print(f"       samples.pkl 与 report 已写入 {cache_dir}")
    if failures:
        _safe_print(f"       ⚠️ 有失败项,明细见 {report}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="动态 .nii.gz 预抽取为 PNG 缓存")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--sheet-name", default="ROI级别")
    ap.add_argument("--img-size", type=int, default=512,
                    help="必须与训练时 img_size 一致(默认 512)")
    ap.add_argument("--limit", type=int, default=0,
                    help=">0 时只处理前 N 个动态影像文件,用于快速验证管道")
    ap.add_argument("--workers", type=int, default=4,
                    help="并行进程数(2026-09-11 并行化);1=串行原路径。"
                         "纯 CPU 解压不占 GPU;单 job 峰值内存 ~700MB,"
                         "64G 机器 4-8 安全")
    a = ap.parse_args()
    return extract(a.data_root, a.xlsx, a.cache_dir, a.sheet_name,
                   a.img_size, a.limit, a.workers)


if __name__ == "__main__":
    sys.exit(main())
