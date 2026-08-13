"""ThyroidXL → 项目标准目录契约(image.png/mask.png/meta.json)转换 + 患者级划分。

## 为什么需要这个脚本(A0 诊断结论)

TN3K 训练的分割主干是**单域过拟合**:A800 实测在 ThyroidXL 上跨域 Dice 仅 17.80%
(90% 样本<50%),廉价归一化反更差 → 确认深层域偏移。ThyroidXL 自带 7162 张真实
像素 mask,混入 TN3K 主训练是补域不变分割特征的唯一有据手段(见 `docs/训练与架构/三头实现记录.md` §6)。

## 安全根因:loader 默认「标签存在」,方向反了

`src/dataset/loader.py:88-96` 的 valid_* 优先级链是 `meta.label_mask > source_label_mask
> 默认 1`——**缺省是 1(标签有效)**。ThyroidXL 有 4470/11632(38.4%)样本无 mask,
若本脚本漏写 `label_mask.mask=0`,这些样本会被当**有效分割监督**用全零 GT 训练,
静默把 TN3K 的 98% Dice 拖垮。5 个 TI-RADS 概念同理(ThyroidXL 无概念标注,占位 0,
漏写则被当真监督破坏概念头)。

**故本脚本的安全不押在「实现正确」上,而是转换完跑硬断言自检(见 `_self_check`),
任一契约不满足直接 raise,绝不产出可能污染训练的数据。**

## 输出契约(与 loader.ThyroidDataset 目录契约一致)

    {dest}/{train,val}/sample_{image_id}/
        image.png                 ← 必有
        mask.png                  ← **仅 mask 可解码才写**(无 mask 样本不写此文件)
        meta.json                 ← label_mask 6 键必须写全(mask + 5 概念),缺省 1 是坑

用法(A800):
    python scripts/data/format_thyroidxl.py --data-root data/public_datasets/ThyroidXL \\
        --dest data/thyroidxl_seg --val-ratio 0.15
"""

import argparse
import json
import os
import shutil

import cv2
import numpy as np

from src.dataset.multiset_adapter import ThyroidXLAdapter
from src.dataset.pathology_dataset import patient_grouped_split

# label_mask 必须写全的键(loader.py:92-96 的 concept_tasks + mask)。缺任一键 → loader
# 落默认 1 → 占位 0 标签被当真监督。5 概念 ThyroidXL 恒无标注(multiset_adapter.py:279 占位)。
_CONCEPT_KEYS = ("composition", "echogenicity", "shape", "margin", "foci")
_ALL_LABEL_MASK_KEYS = ("mask", "malignancy") + _CONCEPT_KEYS


def _build_meta(image_id, patient_id, malignancy, has_mask, has_conclusion):
    """构造 meta.json。`label_mask` **7 键全写**(阻塞2 对策:漏写落 loader 默认 1)。

    ⚠️ `label_mask.*` 语义是**标签有效性标志(0/1)**,不是标签取值——loader.py:94 读成
    `valid_<task>`。**绝不可把标签值塞进去**(Gate2 阻塞1 教训:曾写 `malignancy:int(malignancy)`,
    导致良性样本 malignancy=0 被误当"无良恶性标注",良恶性头只见恶性样本零监督良性)。
    - mask: `has_mask`(可解码为 1,否则 0)——无 mask 样本 valid_mask=0,分割零梯度。
    - malignancy: `has_conclusion`(有 Conclusion 即 1)——**与良恶性取值无关**,ThyroidXL
      实测 100% 有 Conclusion 故恒 1,但用独立信号不写死,防未来数据有缺 Conclusion 样本。
    - 5 概念: 恒 0(ThyroidXL 无概念标注,占位 0 不可当真监督)。
    """
    return {
        "image_id": image_id,
        "patient_id": patient_id,
        "label_malignancy": int(malignancy),
        # 5 概念占位 0(仅为 loader 读 label_* 不 KeyError;valid=0 已屏蔽其梯度)。
        "label_composition": 0,
        "label_echogenicity": 0,
        "label_shape": 0,
        "label_margin": 0,
        "label_foci": [0, 0, 0, 0],
        "label_authenticity": 1,  # 真实超声
        "is_duplicate": 0,
        "is_tampered": 0,
        # 🔴 valid 掩膜(标签有效性,非取值):mask 按能否解码;malignancy 按有无 Conclusion;5 概念恒 0。
        "label_mask": {
            "mask": 1 if has_mask else 0,
            "malignancy": 1 if has_conclusion else 0,
            "composition": 0,
            "echogenicity": 0,
            "shape": 0,
            "margin": 0,
            "foci": 0,
        },
    }


def convert(data_root, dest, val_ratio=0.15, seed=42):
    adapter = ThyroidXLAdapter(data_root)
    samples = adapter.samples
    print(f"[format_thyroidxl] ThyroidXL 样本总数 = {len(samples)}")

    # 【高·Gate2终审】清空 dest:重复运行(换 seed/val_ratio)时,同一 image_id 可能第一次落
    # train/、第二次落 val/,两目录并存 → 磁盘级患者泄漏,而 _self_check 的内存本轮 leak 检测
    # 查不到(逐目录 meta 都合法)。开头清空,确保产物只含本轮划分,不依赖调用方手动 rm。
    # 🔴 footgun 防护(终审 blocker):dest 是用户自由输入,裸 rmtree 若误传 --dest data/
    # 或 data/train 会**不可逆删掉 TN3K 训练集/验证集**。故 rmtree 前拒绝 dest 指向任何已知活
    # 数据根(data / data_root / 常见 train_path/val_path)。用 realpath(非 abspath)解析,
    # 顺带挡住 symlink/junction 绕过(如 data_link→data)。⚠️ 已知列表非穷尽,新增数据目录须
    # 同步维护此列表(见 scripts/data/AGENTS.md §3)。
    # 防护通过 = dest 确认非活数据目录,故整个 split 目录可安全 rmtree(与 _self_check 的
    # n_on_disk 口径一致——早期"只删 sample_*"会漏删旧命名残留致自检计数误报,终审复审修正)。
    dest_real = os.path.realpath(dest)
    _protected = {os.path.realpath(p) for p in (
        "data", "data/train", "data/val", "data/holdout", data_root,
        os.path.join(data_root, "train"), os.path.join(data_root, "test"))}
    if dest_real in _protected:
        raise ValueError(
            f"[format_thyroidxl] 拒绝清空受保护目录 dest={dest_real}"
            "(疑似指向活数据集,会不可逆删除)。请用独立目录如 data/thyroidxl_seg。")
    for split in ("train", "val"):
        split_dir = os.path.join(dest, split)
        if os.path.isdir(split_dir):
            print(f"[format_thyroidxl] 清空旧产物: {os.path.realpath(split_dir)}")
            shutil.rmtree(split_dir)

    # 中3(Gate2):复用现成 `patient_grouped_split`(pathology_dataset.py)——它带**按 label
    # 分层**(train/val 良恶性患者比例一致)+ len-1 上界(防单患者类挖空 train),比自造更稳。
    # 它要 samples 是 [{patient_key, label, ...原样本}];构造这层适配后调用。
    # get_task_mask 的 malignancy = has_conclusion(有效标志),一并在此取,避免循环内二次解析同一 nodule。
    grouped_input = []
    for sm in samples:
        labels = adapter.get_labels(sm)
        task_mask = adapter.get_task_mask(sm)
        grouped_input.append({
            "patient_key": str(sm["patient_id"]),
            "label": int(labels["malignancy"]),
            "has_conclusion": int(task_mask["malignancy"]) == 1,
            "_raw": sm,  # 透传原样本,划分后取回
        })
    # 【中·终审】image_id 撞名检测**前移到写盘前**:train/test 两侧同 stem 会写进同一
    # sample_dir 覆盖前者。在此 fail-fast(零写盘),而非写循环里部分写盘后才 raise 留半成品。
    _img_ids = [str(item["_raw"]["image_id"]) for item in grouped_input]
    if len(set(_img_ids)) != len(_img_ids):
        from collections import Counter
        dup = [k for k, v in Counter(_img_ids).items() if v > 1]
        raise ValueError(
            f"[format_thyroidxl] image_id 撞名(共 {len(dup)} 个,如 {dup[:5]}):"
            "train/test 两侧同 stem 会写进同一 sample_dir 覆盖。"
            "需在 adapter 侧对 image_id 加 split 前缀去重。")
    train_items, val_items = patient_grouped_split(
        grouped_input, val_frac=val_ratio, seed=seed)
    split_of = {"train": train_items, "val": val_items}

    counts = {  # 真实计数(A0 复审:不采信估值,转换时实数打印)
        "train": {"has_mask": 0, "no_mask": 0, "malignant": 0, "benign": 0},
        "val": {"has_mask": 0, "no_mask": 0, "malignant": 0, "benign": 0},
    }
    train_pat, val_pat = set(), set()
    n_written = 0
    seen_image_ids = set()  # 中·终审:image_id 跨 train/test 撞名会写进同一 sample_dir,及早明确报错

    for split, items in split_of.items():
        for item in items:
            sm = item["_raw"]
            pid = item["patient_key"]
            (val_pat if split == "val" else train_pat).add(pid)

            img = adapter.get_image(sm)          # uint8 [H,W]
            mask = adapter.get_mask(sm)          # 已内含 cv2.imread None 判定,None = 无有效 mask
            malignancy = int(item["label"])
            has_conclusion = item["has_conclusion"]  # 已在 grouped_input 阶段取,不重复解析
            # 🔴 高2 对策:mask 不仅要非 None,还须**有前景像素**——get_mask 对"文件存在但全黑"
            # 返回非 None 全零数组,若只判 None 会把全黑 mask 当有效分割监督(假监督漏网)。
            has_mask = mask is not None and float(mask.sum()) > 0

            image_id = str(sm["image_id"])
            if image_id in seen_image_ids:
                raise ValueError(
                    f"[format_thyroidxl] image_id 撞名: {image_id!r} 出现多次"
                    "(train/test 两侧同 stem?)——会写进同一 sample_dir 覆盖前者,"
                    "拒绝。需在 adapter 侧对 image_id 加 split 前缀去重。")
            seen_image_ids.add(image_id)
            sample_dir = os.path.join(dest, split, f"sample_{image_id}")
            os.makedirs(sample_dir, exist_ok=True)

            if not cv2.imwrite(os.path.join(sample_dir, "image.png"), img):
                raise IOError(f"image.png 写盘失败(磁盘满?): {sample_dir}")
            # 🔴 阻塞1 对策:无 mask/全黑 mask 样本**绝不写 mask.png**(双重防线:文件不存在 + meta mask:0)。
            if has_mask:
                # get_mask 返回 float32 值域[0,1];落盘转回 0/255 uint8(loader 按 >127 二值化)。
                mask_u8 = (mask * 255).astype(np.uint8)  # type: ignore[operator]  # has_mask 保证 mask 非 None
                if not cv2.imwrite(os.path.join(sample_dir, "mask.png"), mask_u8):
                    raise IOError(f"mask.png 写盘失败(磁盘满?): {sample_dir}")

            meta = _build_meta(image_id, pid, malignancy, has_mask, has_conclusion)
            with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            c = counts[split]
            c["has_mask" if has_mask else "no_mask"] += 1
            c["malignant" if malignancy else "benign"] += 1
            n_written += 1

    # 患者无跨 split 泄漏(应为 0)。
    leak = train_pat & val_pat
    print(f"[format_thyroidxl] train 患者={len(train_pat)} val 患者={len(val_pat)} "
          f"跨 split 泄漏={len(leak)}")
    for split in ("train", "val"):
        c = counts[split]
        print(f"[format_thyroidxl] {split}: 有mask={c['has_mask']} 无mask={c['no_mask']} "
              f"恶性={c['malignant']} 良性={c['benign']} "
              f"共={c['has_mask']+c['no_mask']}")

    # 【中·Gate2复审】计数对齐:写盘样本数须 == adapter.samples 总数(防中途崩溃/被吞异常
    # 留残缺数据集却全断言通过)。不等即数据集不完整,拒绝。
    if n_written != len(samples):
        raise AssertionError(
            f"[format_thyroidxl] 写盘样本数 {n_written} ≠ adapter 样本总数 {len(samples)},"
            "数据集不完整(转换中途失败?),拒绝用于训练")

    _self_check(dest, leak, expected_total=len(samples))
    return counts


def _self_check(dest, patient_leak, expected_total=None):
    """🔴 硬断言自检(sonnet 根因对策):安全不押在「实现正确」,靠此锁死。

    任一契约违反直接 raise,绝不让可能污染训练的数据落地被误用。校验 5 条契约:
      1. label_mask 7 键(mask+malignancy+5概念)全写 + 值 ∈{0,1}(是有效标志非标签取值);
      2. mask:0 样本无 mask.png(阻塞1 防线);mask:1 样本有 mask.png;
      3. mask:1 的 mask.png **有前景像素**(高2 防线:全黑 mask 也是假监督);
      4. 患者无跨 split 泄漏;
      5. 磁盘样本总数 == expected_total(防残留旧产物/中途崩溃留残缺数据集,中·Gate2复审)。
    """
    errors = []
    if patient_leak:
        errors.append(f"患者跨 split 泄漏 {len(patient_leak)} 个: {list(patient_leak)[:5]}")

    n_on_disk = 0
    for split in ("train", "val"):
        split_dir = os.path.join(dest, split)
        if not os.path.isdir(split_dir):
            errors.append(f"{split} 目录不存在")
            continue
        n_nomask_no_png = 0
        for name in os.listdir(split_dir):
            sdir = os.path.join(split_dir, name)
            if not os.path.isdir(sdir):
                continue
            n_on_disk += 1
            meta_path = os.path.join(sdir, "meta.json")
            if not os.path.exists(meta_path):  # 低6:缺 meta 并入 errors 而非裸崩
                errors.append(f"{split}/{name}: 缺 meta.json")
                continue
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            lm = meta.get("label_mask", {})
            # 契约1: label_mask 7 键全写(mask/malignancy/5概念),无缺失落 loader 默认1。
            missing = [k for k in _ALL_LABEL_MASK_KEYS if k not in lm]
            if missing:
                errors.append(f"{split}/{name}: label_mask 缺键 {missing}(会落 loader 默认1)")
            # 契约1b(阻塞1 教训): 各键值必须 ∈{0,1}——label_mask 是**有效性标志**,
            # 不得塞标签取值(曾误写 malignancy:int(malignancy) 致良性样本被屏蔽监督)。
            bad_vals = [k for k in _ALL_LABEL_MASK_KEYS if lm.get(k) not in (0, 1)]
            if bad_vals:
                errors.append(f"{split}/{name}: label_mask 键 {bad_vals} 值非 0/1"
                              f"(必须是有效标志,不是标签取值)")
            # 契约2: mask:0 样本无 mask.png(阻塞1 双重防线之一)。
            png_path = os.path.join(sdir, "mask.png")
            has_png = os.path.exists(png_path)
            if lm.get("mask") == 0:
                if has_png:
                    errors.append(f"{split}/{name}: mask:0 却存在 mask.png(假监督风险)")
                n_nomask_no_png += 1
            elif lm.get("mask") == 1:
                if not has_png:
                    errors.append(f"{split}/{name}: mask:1 却无 mask.png")
                else:
                    # 契约3(高2 防线): mask:1 的 png 必须有前景像素,否则全黑=假监督。
                    m = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
                    if m is None or int((m > 127).sum()) == 0:
                        errors.append(
                            f"{split}/{name}: mask:1 但 mask.png 无前景像素"
                            "(全黑/损坏,会成假监督)")
        print(f"[self_check] {split}: 无mask且无mask.png 样本 = {n_nomask_no_png}")

    # 契约5: 磁盘样本总数须 == 期望(防残留旧产物 / 中途崩溃留残缺数据集)。
    if expected_total is not None and n_on_disk != expected_total:
        errors.append(
            f"磁盘样本总数 {n_on_disk} ≠ 期望 {expected_total}"
            "(残留旧产物或转换不完整,数据集不可信)")

    if errors:
        raise AssertionError(
            "[format_thyroidxl] 自检失败,数据未通过契约,拒绝用于训练:\n  "
            + "\n  ".join(errors[:20]))
    print("[self_check] ✅ 全部契约通过(label_mask 7键齐且值∈{0,1} / mask:0无png / "
          "mask:1有前景 / 患者无泄漏 / 计数对齐)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/public_datasets/ThyroidXL")
    ap.add_argument("--dest", default="data/thyroidxl_seg")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    convert(args.data_root, args.dest, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
