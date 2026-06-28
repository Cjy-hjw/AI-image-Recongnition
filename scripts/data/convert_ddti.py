"""DDTI (Digital Database Thyroid Image) → 项目标准 meta.json 转换器。

DDTI 是开放获取甲状腺超声数据集(99病例/~134-637图),每病例含:
  - 原始超声图(560x360)
  - XML 标注:含 <composition>/<echogenicity>/<margins>/<calcifications>/<tirads>
    及 <mark> 内 SVG 多边形点(结节轮廓)。

本脚本把 DDTI 的**文字 TI-RADS 标签**映射到项目的整数编码(对齐
scripts/data/pseudo_labels.py 的 0-3 口径),把 SVG 轮廓光栅化为分割 mask,
并产出与 split_tn3k.py 完全一致的 sample_xxx/{image.png, mask.png, meta.json}
目录结构。

关键:DDTI 提供**真实概念标签**(非伪标签)与**真实 TI-RADS 分级**,故:
  - label_mask 标记 5 个概念 + malignancy 均为有效(=1),供 Masked Loss 启用真监督。
  - label_malignancy 由 TI-RADS 分级推导(≥4 → 恶性),并记录原始 tirads 备查。

⚠ DDTI XML 的字段名/取值用词在不同上传版本间可能有差异(单复数、缩写、大小写)。
本脚本的文字→编码映射用**子串匹配 + 容错**,未识别值回退到中性默认并计数告警,
首次接入后请抽查 _UNMAPPED 统计,据实补充映射表。
"""
import os
import csv
import json
import glob
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np


# ── 文字标签 → 项目整数编码(对齐 pseudo_labels.py 口径)─────────────────────
# composition: 0=cystic, 1=spongiform, 2=mixed, 3=solid
_COMPOSITION_MAP = {
    "cystic": 0,
    "spongiform": 1,
    "predominantly cystic": 2,
    "mixed": 2,
    "solid": 3,
    "predominantly solid": 3,
}
# echogenicity: 0=anechoic, 1=hyper/iso, 2=hypo, 3=very hypo
_ECHOGENICITY_MAP = {
    "anechoic": 0,
    "hyperechogenicity": 1,
    "hyperechoic": 1,
    "isoechogenicity": 1,
    "isoechoic": 1,
    "hypoechogenicity": 2,
    "hypoechoic": 2,
    "marked hypoechogenicity": 3,
    "very hypoechoic": 3,
}
# shape: 0=wider-than-tall, 1=taller-than-wide(DDTI 通常无此字段,从轮廓推)
# margin: 0=smooth/well-defined, 1=ill-defined, 2=lobulated/irregular, 3=ETE
_MARGIN_MAP = {
    "well defined": 0,
    "well-defined": 0,
    "smooth": 0,
    "ill defined": 1,
    "ill-defined": 1,
    "microlobulated": 2,
    "macrolobulated": 2,
    "lobulated": 2,
    "irregular": 2,
    "spiculated": 2,
    "extra-thyroidal": 3,
    "extrathyroidal": 3,
}
# foci(钙化)→ 4 维 one-hot: [none/comet, macro, peripheral/rim, punctate/micro]
_CALCIFICATION_MAP = {
    "non": [1, 0, 0, 0],
    "none": [1, 0, 0, 0],
    "no": [1, 0, 0, 0],
    "comet": [1, 0, 0, 0],
    "macrocalcification": [0, 1, 0, 0],
    "macrocalcifications": [0, 1, 0, 0],
    "coarse": [0, 1, 0, 0],
    "peripheral": [0, 0, 1, 0],
    "rim": [0, 0, 1, 0],
    "microcalcification": [0, 0, 0, 1],
    "microcalcifications": [0, 0, 0, 1],
    "punctate": [0, 0, 0, 1],
}

_UNMAPPED = {"composition": set(), "echogenicity": set(), "margin": set(), "calcifications": set()}


def _lookup(text, table, default):
    """子串容错匹配:DDTI 用词不统一,优先精确,退而子串包含。"""
    if text is None:
        return default
    key = text.strip().lower()
    if key in table:
        return table[key]
    for k, v in table.items():
        if k in key or key in k:
            return v
    return default


def _xml_text(root, *tags):
    """取第一个非空标签文本(容忍不同 DDTI 版本的标签命名差异)。"""
    for tag in tags:
        el = root.find(tag)
        if el is not None and el.text and el.text.strip():
            return el.text.strip()
    return None


def _parse_svg_points(mark_el):
    """从 <mark> 解析轮廓点。DDTI 的 points 多为 JSON 风格 [{'points':[{x,y},...]}]
    或 SVG polygon 'x1,y1 x2,y2'。两种都尝试。返回 list[(x,y)]。"""
    points = []
    if mark_el is None:
        return points
    svg_el = mark_el.find("svg")
    raw = (svg_el.text if svg_el is not None and svg_el.text else None)
    if raw is None and mark_el.text:
        raw = mark_el.text
    if not raw:
        return points
    raw = raw.strip()
    # 形式1: JSON-like  [{"points": [{"x":1,"y":2}, ...]}]
    if raw.startswith("[") or raw.startswith("{"):
        try:
            data = json.loads(raw.replace("'", '"'))
            if isinstance(data, dict):
                data = [data]
            for region in data:
                for p in region.get("points", []):
                    points.append((int(round(p["x"])), int(round(p["y"]))))
            return points
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    # 形式2: SVG polygon "x1,y1 x2,y2 ..."
    for tok in raw.replace(";", " ").split():
        if "," in tok:
            try:
                x, y = tok.split(",")[:2]
                points.append((int(round(float(x))), int(round(float(y)))))
            except ValueError:
                continue
    return points


def _tirads_to_int(text):
    """'4a'/'4b'/'5' → 数值等级(取首位数字);返回 (level:int|None)。"""
    if not text:
        return None
    for ch in text.strip():
        if ch.isdigit():
            return int(ch)
    return None


def convert_ddti(ddti_root, dest_dir, malignant_tirads_threshold=4):
    """遍历 DDTI 目录,产出项目标准 sample 目录。

    Args:
        ddti_root: DDTI 解压根目录(含图像 jpg 与同名/配套 xml)。
        dest_dir: 输出目录(如 data/ddti),内部生成 sample_xxx/。
        malignant_tirads_threshold: TI-RADS >= 此值判为恶性(默认 4)。

    Returns:
        total_converted: int
    """
    os.makedirs(dest_dir, exist_ok=True)

    # 收集所有 xml(DDTI 结构多样:可能 case 子目录,也可能扁平)
    xml_files = glob.glob(os.path.join(ddti_root, "**", "*.xml"), recursive=True)
    print(f"发现 {len(xml_files)} 个 XML 标注文件")

    total = 0
    skipped = 0
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            skipped += 1
            continue
        root = tree.getroot()

        case_id = _xml_text(root, "number", "id") or os.path.splitext(os.path.basename(xml_path))[0]

        # 找配套图像:同名 jpg/png,或同目录单图
        base = os.path.splitext(xml_path)[0]
        img_path = None
        for ext in (".jpg", ".JPG", ".png", ".PNG", ".jpeg"):
            if os.path.exists(base + ext):
                img_path = base + ext
                break
        if img_path is None:
            cand = [f for f in glob.glob(os.path.join(os.path.dirname(xml_path), "*"))
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            img_path = cand[0] if cand else None
        if img_path is None:
            skipped += 1
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue
        H, W = img.shape

        # 文字概念 → 整数编码
        comp_txt = _xml_text(root, "composition")
        echo_txt = _xml_text(root, "echogenicity")
        marg_txt = _xml_text(root, "margins", "margin")
        calc_txt = _xml_text(root, "calcifications", "calcification")

        comp = _lookup(comp_txt, _COMPOSITION_MAP, 3)
        echo = _lookup(echo_txt, _ECHOGENICITY_MAP, 1)
        margin = _lookup(marg_txt, _MARGIN_MAP, 0)
        foci = _lookup(calc_txt, _CALCIFICATION_MAP, [1, 0, 0, 0])
        if comp_txt and _lookup(comp_txt, _COMPOSITION_MAP, None) is None:
            _UNMAPPED["composition"].add(comp_txt)
        if echo_txt and _lookup(echo_txt, _ECHOGENICITY_MAP, None) is None:
            _UNMAPPED["echogenicity"].add(echo_txt)
        if marg_txt and _lookup(marg_txt, _MARGIN_MAP, None) is None:
            _UNMAPPED["margin"].add(marg_txt)
        if calc_txt and _lookup(calc_txt, _CALCIFICATION_MAP, None) is None:
            _UNMAPPED["calcifications"].add(calc_txt)

        # SVG 轮廓 → mask
        mask = np.zeros((H, W), dtype=np.uint8)
        pts = _parse_svg_points(root.find("mark"))
        if len(pts) >= 3:
            poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [poly], 255)

        # shape: DDTI 无直接字段,从 mask 外接框推(对齐 pseudo_labels)
        shape = 0
        if mask.sum() > 0:
            ys, xs = np.where(mask > 0)
            h = ys.max() - ys.min() + 1
            w = xs.max() - xs.min() + 1
            shape = 1 if h > w else 0

        # malignancy: 由 TI-RADS 推导
        tirads_level = _tirads_to_int(_xml_text(root, "tirads"))
        malignancy = int(tirads_level >= malignant_tirads_threshold) if tirads_level else 0

        # DDTI 提供真实概念标签 → 全部有效(malignancy 为推导,标有效但记录来源)
        label_mask = {
            "composition": 1, "echogenicity": 1, "shape": 1,
            "margin": 1, "foci": 1, "malignancy": 1,
        }

        image_id = f"ddti_{case_id}"
        sample_dir = os.path.join(dest_dir, f"sample_{image_id}")
        os.makedirs(sample_dir, exist_ok=True)

        meta = {
            "image_id": image_id,
            "patient_id": image_id,  # DDTI 每病例独立
            "label_composition": int(comp),
            "label_echogenicity": int(echo),
            "label_shape": int(shape),
            "label_margin": int(margin),
            "label_foci": foci,
            "label_malignancy": int(malignancy),
            "label_authenticity": 0,
            "is_duplicate": 0,
            "is_tampered": 0,
            "tamper_bbox": None,
            "foci_coords": [],
            "label_mask": label_mask,
            "source": "DDTI",
            "tirads_raw": tirads_level,
        }

        cv2.imwrite(os.path.join(sample_dir, "image.png"), img)
        cv2.imwrite(os.path.join(sample_dir, "mask.png"), mask)
        with open(os.path.join(sample_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        total += 1

    print(f"转换完成:{total} 样本写入 {dest_dir}(跳过 {skipped})")
    for field, vals in _UNMAPPED.items():
        if vals:
            print(f"  ⚠ 未识别 {field} 取值(用了默认): {sorted(vals)}")
    return total


def write_split_csv(dest_dir, split_ratio=0.85, seed=42):
    """对转换后的 ddti 样本做图像级随机 train/val 划分,写 split.csv 记录归属。"""
    samples = sorted(
        d for d in os.listdir(dest_dir)
        if os.path.isdir(os.path.join(dest_dir, d)) and d.startswith("sample_")
    )
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(samples))
    cut = int(len(samples) * split_ratio)
    split = {s: ("train" if i in set(idx[:cut].tolist()) else "val")
             for i, s in enumerate(samples)}
    csv_path = os.path.join(dest_dir, "split.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "split"])
        for s in samples:
            w.writerow([s, split[s]])
    n_tr = sum(1 for v in split.values() if v == "train")
    print(f"划分写入 {csv_path}: train={n_tr}, val={len(samples) - n_tr}")
    return csv_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="DDTI → 项目 meta.json 转换器")
    ap.add_argument("--ddti_root", required=True, help="DDTI 解压根目录")
    ap.add_argument("--dest", default="data/ddti", help="输出目录")
    ap.add_argument("--mal_tirads", type=int, default=4, help="TI-RADS>=此值判恶性")
    ap.add_argument("--split", action="store_true", help="额外生成 train/val 划分 csv")
    args = ap.parse_args()

    n = convert_ddti(args.ddti_root, args.dest, args.mal_tirads)
    if args.split and n > 0:
        write_split_csv(args.dest)
