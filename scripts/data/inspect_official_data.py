#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""官方数据一次性探查脚本（云端跑，本地无官方数据无法验证）。

上云真训前把 `official_loader` 的全部数据假设一次性核实，为训练提前扫清障碍。
输出全部 print 到 stdout，云端 agent 跑完把 `inspect_output.txt` 全文贴回即可。

自包含 —— **只依赖 pandas + openpyxl + 标准库**（torch/numpy 缺失也能跑，只是跳过对应块），
        **不 import 项目 `src/`**（本脚本先于代码库上传，云端此刻只有它自己）。
        故所有「候选列名 / 编码表 / 解析逻辑」都在本文件内联，与
        `src/dataset/official_loader.py` + `src/models/label_schema.py` 手工对齐
        （镜像标 MIRROR，2026-08-13）；**代码库上传后一切以 loader 为准，本脚本只作探查**。

用法（云端 agent）：
    python inspect_official_data.py 2>&1 | tee inspect_output.txt
    # 数据不在默认路径时显式指定：
    python inspect_official_data.py --root <标注根目录> --xlsx <标注.xlsx> --sheet ROI级别

每个探查块开头的【Q】注明它回答哪个未决问题。
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import traceback
import unicodedata
from collections import Counter
from typing import Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# MIRROR 区：与 src/dataset/official_loader.py + src/models/label_schema.py 对齐
# （2026-08-13）。改这里之前先想清楚：真正的解析以 loader 为准，本区仅供云端在
#  没有 src/ 时做「解析器能否覆盖真实写法」的对拍。
# ─────────────────────────────────────────────────────────────────────────────
_FIELD_ALIASES = {
    "patient_id":     ("PatientId", "患者ID", "患者编号", "patient_id"),
    "series_uid":     ("SeriesUid", "SeriesInstanceUID", "序列UID"),
    "image_property": ("图像性质",),
    "roi_name":       ("RoiName", "ROI名称", "roi_name"),
    "slice_index":    ("ANNOTATION->SliceIndex", "SliceIndex"),
    "malignancy":     ("ROI良恶性", "ROI->CLINICAL->良恶性", "良恶性"),
    "lesion_loc":     ("主病灶位置", "病灶位置", "lesion_loc"),
    "extrathy_ext":   ("甲状腺外侵犯", "ETE", "extrathy_ext"),
    "clin_t_stage":   ("临床T分期", "T分期", "clin_t_stage"),
    "tirads":         ("ROI-TIRADS", "ROI->CLINICAL->TI-RADS", "TI-RADS",
                       "TIRADS", "TI_RADS", "TI-RADS分类", "TI-RADS总分类"),
    "composition":    ("成分", "composition"),
    "echogenicity":   ("回声", "echogenicity"),
    "shape":          ("形状", "纵横比", "shape"),
    "margin":         ("边缘", "margin"),
}

# 三头 + TI-RADS 训练监督的实际来源列；候选名全数未命中 = 该任务零监督（且静默）。
_CLINICAL_LOGICAL_FIELDS = (
    "malignancy", "lesion_loc", "extrathy_ext", "clin_t_stage", "tirads",
)

_ETE_LABEL_TO_CODE = {
    "甲状腺包膜": 2, "毗邻软组织": 3, "血管": 4, "无": 1,
    "无病灶（阴性）": 1, "无病灶(阴性)": 1, "NA/UNK": 1,
}
_TIRADS_ROMAN_TO_CODE = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "Ⅰ": 1, "Ⅱ": 2, "Ⅲ": 3, "Ⅳ": 4, "Ⅴ": 5,
}

# ACR 五要素 / 「详细评分」子项列的搜索关键词（回答第11轮「详细评分 75.8%」是哪列）。
_ACR_KEYWORDS = [
    "成分", "composition", "回声", "echo", "形状", "shape", "纵横比",
    "边缘", "margin", "钙化", "强回声", "foci", "点状", "结构", "score",
    "评分", "acr", "ti-rads", "tirads", "ti_rads", "详细", "分值",
]

# 镜像 src/models/label_schema.py 的 LOCATION_LABELS/LOCATION_NULL_LABELS(单一权威,顺序不可改)。
_LOCATION_LABELS = ("左上叶", "左中叶", "左下叶", "右上叶", "右中叶", "右下叶", "峡部")
_LOCATION_NULL_LABELS = ("无（阴性）", "无(阴性)", "NA/UNK")

MAIN_LESION_ROI = "M"
_SECONDARY_ROI_RE = re.compile(r"^病灶\d+$")


def _cell_str(v) -> str:
    if v is None:
        return ""
    try:
        if isinstance(v, float) and v != v:  # NaN
            return ""
    except Exception:
        pass
    s = str(v).strip()
    return "" if s.lower() in ("", "nan", "none") else s


def _is_empty(v) -> bool:
    return _cell_str(v) == ""


def _digit_prefix(v) -> Optional[int]:
    m = re.match(r"^\s*(\d+)", _cell_str(v))
    return int(m.group(1)) if m else None


def _parse_malignancy(v) -> Optional[int]:
    s = _cell_str(v)
    if not s:
        return None
    if "恶" in s or "阳" in s:
        return 1
    if "良" in s or "阴" in s:
        return 0
    d = _digit_prefix(s)
    return None if d is None else int(d != 0)


def _parse_t_stage(v) -> Optional[int]:
    s = _cell_str(v)
    if not s:
        return None
    d = _digit_prefix(s)
    if d is not None:
        return int(d != 0)
    if "有" in s:
        return 1
    if "无" in s:
        return 0
    return None


def _parse_ete_code(v) -> Optional[int]:
    if _is_empty(v):
        return None
    s_full = _cell_str(v)
    if ("阴性" in s_full) or ("无病灶" in s_full):
        return None
    d = _digit_prefix(v)
    if d is not None and 1 <= d <= 4:
        return d
    s = re.sub(r"^\s*\d+\s*", "", s_full).strip()
    if s.upper() in ("NA/UNK", "NA", "N/A", "UNK", "UNKNOWN"):
        return None
    return _ETE_LABEL_TO_CODE.get(s)


def _parse_tirads(v) -> Optional[int]:
    if _is_empty(v):
        return None
    s_full = _cell_str(v)
    d = _digit_prefix(v)
    if d is not None and 1 <= d <= 5:
        return d
    s = re.sub(r"^\s*\d+\s*", "", s_full).strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"(类|级)\s*$", "", s).strip().upper()
    if not s:
        return None
    return _TIRADS_ROMAN_TO_CODE.get(s)


def _parse_location_code(v) -> Optional[int]:
    """镜像 `src/inference/prediction_json.py::location_label_to_index`(返回改 1-based 对齐本文件其它 _parse_* 的官方码约定)。

    数字前缀优先(`2左中叶`→2);无前缀回退中文标签匹配(`左中叶`→2)——真实 loader
    两步都有,此前镜像只做了第一步,纯中文无数字前缀的写法会被误判成"未命中"。
    """
    s = _cell_str(v)
    if not s:
        return None
    d = _digit_prefix(s)
    if d is not None:
        return d if 1 <= d <= 7 else None
    for null_label in _LOCATION_NULL_LABELS:
        if null_label in s:
            return None
    for i, name in enumerate(_LOCATION_LABELS):
        if name in s:
            return i + 1
    return None


def _is_lesion_roi(roi_name) -> bool:
    s = _cell_str(roi_name)
    return s == MAIN_LESION_ROI or bool(_SECONDARY_ROI_RE.match(s))


# 逻辑字段 → 内联解析器（块 [5] 对拍用；无对应解析器的列跳过）。
_PARSERS = {
    "malignancy": _parse_malignancy,
    "lesion_loc": _parse_location_code,
    "extrathy_ext": _parse_ete_code,
    "clin_t_stage": _parse_t_stage,
    "tirads": _parse_tirads,
}

# ─────────────────────────────────────────────────────────────────────────────
_LINE = "=" * 78


def _hdr(title: str) -> None:
    print("\n" + _LINE)
    print(title)
    print(_LINE)


def _safe(fn) -> None:
    """块级容错：单块抛异常不影响其余块（云端跑一次要拿尽量多信息）。"""
    try:
        fn()
    except Exception:
        print(f"\n!! 该块执行失败：\n{traceback.format_exc()}")


class _State:
    """缓存,供后续块复用。df 实际是 pandas.DataFrame,标 Any 规避静态检查噪音。"""
    df: Any = None
    xlsx: Optional[str] = None
    sheet: Optional[str] = None


_state = _State()


# ── [0] 环境 ─────────────────────────────────────────────────────────────────
def block_env() -> None:
    _hdr("[0] 环境探查  【Q: 云端有无训练依赖 pandas/openpyxl；真实 GPU/显存】")
    print(f"python   : {sys.version.splitlines()[0]}")
    print(f"platform : {sys.platform}")
    for mod in ("pandas", "openpyxl", "numpy"):
        try:
            m = __import__(mod)
            print(f"{mod:<9}: {getattr(m, '__version__', '?')}  (OK)")
        except Exception as e:
            print(f"{mod:<9}: 缺失 !!  ({e})")
    try:
        import torch
        print(f"torch    : {torch.__version__}  cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  GPU{i}: {p.name}  显存={p.total_memory/1024**3:.1f} GiB")
    except Exception as e:
        print(f"torch    : 不可用 ({e})  —— GPU 探查跳过")


# ── xlsx 定位 + sheet 选择 ────────────────────────────────────────────────────
def _find_xlsx(root: str, explicit):
    if explicit:
        return explicit
    cands = []
    for base in (root, os.path.dirname(root.rstrip("/")), "."):
        if base and os.path.isdir(base):
            cands += glob.glob(os.path.join(base, "*.xlsx"))
            cands += glob.glob(os.path.join(base, "**", "*.xlsx"), recursive=True)
    cands = [c for c in dict.fromkeys(cands) if "~$" not in c]  # 去重 + 排除 Excel 锁文件
    return cands[0] if cands else None


def _pick_roi_sheet(xls, prefer: str) -> str:
    names = xls.sheet_names
    if prefer in names:
        return prefer
    for nm in names:
        try:
            cols = list(xls.parse(nm, nrows=0).columns)
        except Exception:
            continue
        if any(a in cols for a in _FIELD_ALIASES["roi_name"]):
            return nm
    return names[0] if names else prefer


def block_locate(root, explicit_xlsx, prefer_sheet) -> None:
    _hdr("[1] 定位标注 xlsx + sheet 概览  【Q: 表在哪、有几个 sheet、ROI级别 是哪个】")
    import pandas as pd
    path = _find_xlsx(root, explicit_xlsx)
    if not path or not os.path.isfile(path):
        print(f"!! 未找到 xlsx（root={root}, explicit={explicit_xlsx}）。"
              f"请用 --xlsx 显式指定后重跑。")
        return
    print(f"xlsx = {path}")
    xls = pd.ExcelFile(path)
    print(f"sheets = {xls.sheet_names}")
    for nm in xls.sheet_names:
        try:
            overview = xls.parse(nm, nrows=0)
            ncols = overview.shape[1]
            nrows = xls.parse(nm, usecols=[0]).shape[0]
            print(f"  - {nm!r}: 约 {nrows} 行 × {ncols} 列")
        except Exception as e:
            print(f"  - {nm!r}: 概览失败 ({e})")
    sheet = _pick_roi_sheet(xls, prefer_sheet)
    print(f"--> 选用 sheet = {sheet!r}")
    # dtype=str 保留 `2左中叶`/`5 V类` 原样,与 loader 一致
    # header=1:与 official_loader.parse_annotation_xlsx 一致(官方xlsx合并单元格
    # 表头,真实列名在第2行)——此前本行漏加,导致本诊断脚本与 loader 读到不同列名,
    # 诊断结论会误导(见 2026-08-13 Gate2 审查发现)。
    df = pd.read_excel(path, sheet_name=sheet, header=1, dtype=str)
    _state.df = df
    _state.xlsx = path
    _state.sheet = sheet
    print(f"--> 载入 DataFrame: {df.shape[0]} 行 × {df.shape[1]} 列")


# ── [2] 真实列名逐列 repr ─────────────────────────────────────────────────────
def block_columns() -> None:
    _hdr("[2] ROI级别 sheet 全部真实列名（repr）"
         "  【Q: 精确匹配能否命中；有无前导空格/Unnamed/多级表头伪影】")
    df = _state.df
    if df is None:
        print("!! 无 DataFrame,跳过"); return
    for i, c in enumerate(df.columns):
        print(f"  [{i:>2}] {c!r}")


# ── [3] _FIELD_ALIASES 命中诊断 ───────────────────────────────────────────────
def block_hit() -> None:
    _hdr("[3] _FIELD_ALIASES 逐字段命中诊断"
         "  【Q: 精确匹配命中吗；若靠 strip 才命中 = loader 会静默零监督,需修候选名】")
    df = _state.df
    if df is None:
        print("!! 无 DataFrame,跳过"); return
    cols = list(df.columns)
    cols_stripped = {str(c).strip(): c for c in cols}
    for logical, aliases in _FIELD_ALIASES.items():
        exact = next((a for a in aliases if a in cols), None)
        strip_hit = next((a for a in aliases if a in cols_stripped), None)
        if exact:
            idx = cols.index(exact)
            tag = "HIT " if logical not in _CLINICAL_LOGICAL_FIELDS else "HIT*"
            print(f"  {tag} {logical:<14} → {exact!r} (第{idx}列)")
        elif strip_hit:
            real = cols_stripped[strip_hit]
            print(f"  ⚠️STRIP {logical:<14} → 精确 MISS,但 strip 后命中 {strip_hit!r} "
                  f"(真实列名={real!r}) —— loader 精确匹配会漏,须把 {real!r} 加进候选名")
        else:
            flag = " 🔴" if logical in _CLINICAL_LOGICAL_FIELDS else ""
            print(f"  MISS {logical:<14} → 全部候选 {aliases} 均未命中{flag}")


# ── [4] 命中临床列 value_counts（repr） ───────────────────────────────────────
def _resolve(logical: str) -> Optional[str]:
    df = _state.df
    if df is None:
        return None
    for a in _FIELD_ALIASES[logical]:
        if a in df.columns:
            return a
    return None


def block_value_counts() -> None:
    _hdr("[4] 命中临床列 value_counts（原始 repr,暴露前导空格/表头误入）"
         "  【Q: 解析器覆盖全部真实写法吗；缺失率多少】")
    df = _state.df
    if df is None:
        print("!! 无 DataFrame,跳过"); return
    for logical in _CLINICAL_LOGICAL_FIELDS + ("slice_index", "image_property", "roi_name"):
        col = _resolve(logical)
        if col is None:
            print(f"\n-- {logical}: (未命中,跳过)")
            continue
        s = df[col]
        n = len(s)
        empty = int(s.map(_is_empty).sum())
        print(f"\n-- {logical}  列名={col!r}  非空 {n-empty}/{n}  缺失率 {empty/n*100:.1f}%")
        vc = s.value_counts(dropna=False)
        for val, cnt in list(vc.items())[:40]:
            print(f"     {cnt:>6}  {val!r}")
        if len(vc) > 40:
            print(f"     ... 共 {len(vc)} 个唯一值,仅显示前 40")


# ── [5] 内联解析器对拍 ────────────────────────────────────────────────────────
def block_parse_check() -> None:
    _hdr("[5] 内联解析器对拍（valid/invalid + 所有解析失败的唯一原始值）"
         "  【Q: 有没有没见过的写法导致零监督】")
    df = _state.df
    if df is None:
        print("!! 无 DataFrame,跳过"); return
    for logical, parser in _PARSERS.items():
        col = _resolve(logical)
        if col is None:
            print(f"\n-- {logical}: (未命中,跳过)"); continue
        valid = invalid_nonempty = empty = 0
        bad: Counter = Counter()
        for v in df[col]:
            if _is_empty(v):
                empty += 1; continue
            if parser(v) is None:
                invalid_nonempty += 1
                bad[_cell_str(v)] += 1
            else:
                valid += 1
        print(f"\n-- {logical}  列名={col!r}  valid={valid}  "
              f"非空但解析失败={invalid_nonempty}  空={empty}")
        if bad:
            print("   解析失败的唯一原始值（这些当前会 valid=0 零监督,需确认是否该救）：")
            for val, cnt in bad.most_common(30):
                print(f"     {cnt:>5}  {val!r}")


# ── [6] ACR 五要素 / 详细评分列搜索 ───────────────────────────────────────────
def block_acr() -> None:
    _hdr("[6] ACR 五要素 / 「详细评分」子项列搜索"
         "  【Q: 第11轮『详细评分 75.8%』是哪列;composition/echo/shape/margin/foci 子项列存在吗】")
    df = _state.df
    if df is None:
        print("!! 无 DataFrame,跳过"); return
    hits = []
    for c in df.columns:
        low = str(c).strip().lower()
        if any(k in low for k in _ACR_KEYWORDS):
            hits.append(c)
    if not hits:
        print("  未找到任何含 ACR/评分/五要素 关键词的列。")
        print("  → 若确认全表无这些列,则第11轮『详细评分』很可能不是独立列,"
              "五要素评分子项在官方数据中不存在,回传明确『未找到』即可。")
        return
    print(f"  命中 {len(hits)} 列:")
    for c in hits:
        s = df[c]
        empty = int(s.map(_is_empty).sum()); n = len(s)
        print(f"\n  ▶ 列名={c!r}  非空 {n-empty}/{n}  缺失率 {empty/n*100:.1f}%")
        for val, cnt in list(s.value_counts(dropna=False).items())[:20]:
            print(f"       {cnt:>6}  {val!r}")


# ── [7] 主病灶(M) vs 非M 缺失率 ──────────────────────────────────────────────
def block_main_vs_sub() -> None:
    _hdr("[7] 主病灶(M行) vs 非M行 临床列缺失率"
         "  【Q: 三头真实监督量;masked-loss 判据『列空→valid=0』覆盖多少】")
    df = _state.df
    if df is None:
        print("!! 无 DataFrame,跳过"); return
    roi_col = _resolve("roi_name")
    if roi_col is None:
        print("!! roi_name 未命中,无法分组"); return
    is_m = df[roi_col].map(lambda v: _cell_str(v) == "M")
    m_df, o_df = df[is_m], df[~is_m]
    print(f"  M 行 {len(m_df)} / 非M 行 {len(o_df)}")
    print(f"  {'字段':<14} {'M行缺失率':>10} {'非M行缺失率':>12}")
    for logical in _CLINICAL_LOGICAL_FIELDS:
        col = _resolve(logical)
        if col is None:
            print(f"  {logical:<14} {'(未命中)':>10}"); continue
        me = m_df[col].map(_is_empty).mean() * 100 if len(m_df) else 0
        oe = o_df[col].map(_is_empty).mean() * 100 if len(o_df) else 0
        print(f"  {logical:<14} {me:>9.1f}% {oe:>11.1f}%")


# ── [8] 良恶性「无」字面复核 ──────────────────────────────────────────────────
def block_malignancy_wu() -> None:
    _hdr("[8] 良恶性列含「无」行的原始 repr"
         "  【Q: 『无』是否纯字符串、会不会含『阴』被误判成良性负样本】")
    df = _state.df
    if df is None:
        print("!! 无 DataFrame,跳过"); return
    col = _resolve("malignancy")
    if col is None:
        print("!! malignancy 未命中"); return
    wu = df[col][df[col].map(lambda v: "无" in _cell_str(v))]
    vc = Counter(_cell_str(v) for v in wu)
    if not vc:
        print("  良恶性列无任何含『无』的行。")
    for val, cnt in vc.most_common():
        parsed = _parse_malignancy(val)
        note = " ← ⚠️ 被解析为良性(0)!" if parsed == 0 else f" (解析={parsed})"
        print(f"  {cnt:>5}  {val!r}{note}")


# ── [9] mask 命名全量 glob ───────────────────────────────────────────────────
def block_mask(root: str, scan_limit: int) -> None:
    _hdr("[9] mask 文件命名全量 glob"
         "  【Q: 主/次病灶命名真实形态;有无裸 M_mask(无编号)/横杠 等变体】")
    if not os.path.isdir(root):
        print(f"!! root 不存在: {root}"); return
    re_m = re.compile(r"^M_\d+_mask\.nii\.gz$")
    re_sub = re.compile(r"^病灶\d+_\d+_mask\.nii\.gz$")
    cnt: Counter = Counter()
    others: Counter = Counter()
    n_dirs = 0
    for _dirpath, _dirnames, fns in os.walk(root):
        n_dirs += 1
        for fn in fns:
            if "mask" not in fn.lower() or not fn.endswith((".nii", ".nii.gz")):
                continue
            if re_m.match(fn):
                cnt["主病灶 M_<n>_mask.nii.gz"] += 1
            elif re_sub.match(fn):
                cnt["次病灶 病灶<i>_<n>_mask.nii.gz"] += 1
            elif fn == "M_mask.nii.gz":
                cnt["🔴 裸 M_mask.nii.gz(无编号)"] += 1
            else:
                others[fn] += 1
        if scan_limit and n_dirs >= scan_limit:
            print(f"  (达到 --mask-scan-limit={scan_limit},提前停止)")
            break
    print(f"  扫描目录数 ≈ {n_dirs}")
    for k, v in cnt.most_common():
        print(f"  {v:>6}  {k}")
    if others:
        print(f"  其余 mask 命名（{sum(others.values())} 个,唯一 {len(others)} 种,前 20）：")
        for fn, v in others.most_common(20):
            print(f"     {v:>5}  {fn!r}")
    else:
        print("  无其余变体命名（命名规范,与代码假设吻合）。")


# ── [10] SliceIndex 概览 ─────────────────────────────────────────────────────
def block_slice() -> None:
    _hdr("[10] SliceIndex 概览  【Q: 从 1 起、范围、越界吗】")
    df = _state.df
    if df is None:
        print("!! 无 DataFrame,跳过"); return
    col = _resolve("slice_index")
    if col is None:
        print("!! slice_index 未命中"); return
    vals = []
    for v in df[col]:
        s = _cell_str(v)
        if not s:
            continue
        try:
            vals.append(float(s))
        except Exception:
            pass
    if not vals:
        print("  该列全空。"); return
    print(f"  非空 {len(vals)}  min={min(vals)}  max={max(vals)}  唯一值数={len(set(vals))}")


# ── [11] 样本构建 dry-run（镜像 build_official_samples 的静动分派 + mask 定位）───
def block_sample_dry_run(root: str) -> None:
    _hdr("[11] 样本构建 dry-run（镜像 official_loader.build_official_samples）"
         "  【Q: 静动分派/img_path/mask_path 命中率;patient一级目录是否=PatientId】")
    df = _state.df
    if df is None:
        print("!! 无 DataFrame,跳过"); return
    if not os.path.isdir(root):
        print(f"!! root 不存在: {root}"); return

    pid_col = _resolve("patient_id")
    roi_col = _resolve("roi_name")
    series_col = _resolve("series_uid")
    prop_col = _resolve("image_property")
    if pid_col is None or roi_col is None:
        print("!! patient_id / roi_name 未命中,无法 dry-run"); return

    by_patient: dict = {}
    for _, row in df.iterrows():
        pid = _cell_str(row.get(pid_col))
        if pid:
            by_patient.setdefault(pid, []).append(row)

    total_lesion_rows = 0
    n_static = n_dynamic = 0
    img_hit = img_miss = 0
    mask_hit = mask_miss = 0
    dyn_no_mask = 0
    pid_dir_exists = pid_dir_missing = 0
    static_series_dir_exists = static_series_dir_missing = 0
    prop_conflicts = 0        # 表列 image_property 与磁盘扩展名判据冲突的患者数
    mixed_patients: set = set()   # 患者级静动混合(证据链第10轮实测应为 0,非 0 需回报)
    sample_bad_paths = []  # 收集少量未命中案例供人工核查

    for pid, rows in by_patient.items():
        pdir = os.path.join(root, pid)
        if os.path.isdir(pdir):
            pid_dir_exists += 1
        else:
            pid_dir_missing += 1
            continue

        # 静动判据 = **磁盘文件扩展名**,与 `official_loader.build_official_samples` 同款。
        # 🔴 **不要退回读 `image_property` 列**(2026-08-14 实测:官方 74 列表头无该列)。
        # 旧写法 `is_dynamic = (patient_prop == 2)` 在该列缺失时 `patient_prop` 恒 None
        # → `is_dynamic` 恒 False → **所有动态样本被静默判成静态**、`n_dynamic` 恒 0,
        # 本脚本是数据探查工具,这种失真不报错、只让统计结论全错,比崩溃更坏。
        patient_is_dynamic = os.path.exists(
            os.path.join(root, pid, pid, f"{pid}.nii.gz")
        )
        # 该列若真存在则仅作一致性校验,不参与判定(与 loader 行为一致)
        patient_prop = None
        for r in rows:
            pv = _digit_prefix(r.get(prop_col)) if prop_col else None
            if pv is not None:
                patient_prop = pv
                break
        if patient_prop is not None and (patient_prop == 2) != patient_is_dynamic:
            prop_conflicts += 1

        for r in rows:
            roi = _cell_str(r.get(roi_col))
            if not _is_lesion_roi(roi):
                continue
            total_lesion_rows += 1

            # 行级兜底:该行明确写了 series_uid 且那个目录下真有 .dcm → 按静态
            # (防患者级混合时静态行被误挂到动态 nii 上,与 loader 同款)
            is_dynamic = patient_is_dynamic
            if is_dynamic and series_col is not None:
                _s = _cell_str(r.get(series_col))
                if _s and glob.glob(os.path.join(root, pid, _s, "*.dcm")):
                    is_dynamic = False
                    mixed_patients.add(pid)

            if is_dynamic:
                n_dynamic += 1
                img_dir = os.path.join(root, pid, pid)
                img_path = os.path.join(img_dir, f"{pid}.nii.gz")
                if os.path.isdir(img_dir):
                    pass
                if os.path.isfile(img_path):
                    img_hit += 1
                else:
                    img_miss += 1
                    if len(sample_bad_paths) < 15:
                        sample_bad_paths.append(("动态img缺失", img_path))
                    continue
            else:
                n_static += 1
                series = _cell_str(r.get(series_col)) if series_col else ""
                img_dir = os.path.join(root, pid, series) if series else os.path.join(root, pid)
                if os.path.isdir(img_dir):
                    static_series_dir_exists += 1
                else:
                    static_series_dir_missing += 1
                dcms = sorted(glob.glob(os.path.join(img_dir, "*.dcm")))
                if dcms:
                    img_hit += 1
                    img_path = dcms[0]
                else:
                    img_miss += 1
                    if len(sample_bad_paths) < 15:
                        sample_bad_paths.append(("静态img缺失(无.dcm)", img_dir))
                    continue

            hits = sorted(
                glob.glob(os.path.join(img_dir, f"{roi}_*_mask.nii.gz"))
                + glob.glob(os.path.join(img_dir, f"{roi}_mask.nii.gz"))
            )
            if hits:
                mask_hit += 1
            else:
                mask_miss += 1
                if is_dynamic:
                    dyn_no_mask += 1
                if len(sample_bad_paths) < 15:
                    sample_bad_paths.append((f"mask缺失(roi={roi})", img_dir))

    print(f"  一级目录(=PatientId?)存在 {pid_dir_exists} / 缺失 {pid_dir_missing}"
          f"（缺失说明 xlsx 患者在磁盘找不到对应目录,需确认目录命名口径）")
    print(f"  病灶行(M/病灶N,已排除关键帧/开始/结束) 共 {total_lesion_rows}"
          f"  静态={n_static}  动态={n_dynamic}"
          f"  ← 静动判据=磁盘扩展名({{pid}}/{{pid}}/{{pid}}.nii.gz 存在即动态)")
    if prop_conflicts:
        print(f"  ⚠️ 表列「图像性质」与磁盘判据冲突的患者数 = {prop_conflicts}"
              f"（以磁盘为准;该列在官方表里本不存在,若这里非 0 说明数据换版了,请回报）")
    if mixed_patients:
        print(f"  🔴 患者级静动混合 = {len(mixed_patients)} 个"
              f"（同时有 {{pid}}.nii.gz 与含 .dcm 的 series 目录;证据链第10轮实测应为 0,"
              f"非 0 请回报这些 PatientId）:{sorted(mixed_patients)[:5]}")
    print(f"  静态图 series 子目录命中 {static_series_dir_exists} / 缺失 {static_series_dir_missing}")
    print(f"  原始影像文件命中 {img_hit} / 未命中 {img_miss}"
          f"（未命中=该行会被 loader 跳过,不构成样本）")
    print(f"  mask 命中 {mask_hit} / 未命中 {mask_miss}"
          f"（动态且无 mask ={dyn_no_mask}→ 该动态病灶整个跳过,loader 既有设计）")
    if sample_bad_paths:
        print("\n  未命中案例抽样（最多15条,人工核查是否目录结构假设有误）：")
        for reason, p in sample_bad_paths:
            print(f"     [{reason}] {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description="官方数据一次性探查（云端跑）")
    ap.add_argument("--root", default="/2026aicompetition/datasets/training/annotation",
                    help="标注 + 影像根目录（含各 PatientId 子目录）")
    ap.add_argument("--xlsx", default=None, help="标注 xlsx 路径（默认在 root 及上级自动搜）")
    ap.add_argument("--sheet", default="ROI级别", help="ROI 级别 sheet 名")
    ap.add_argument("--mask-scan-limit", type=int, default=0,
                    help="mask glob 最多扫描的目录数（0=全扫；数据极大时可设 2000 先看形态）")
    args = ap.parse_args()

    print(_LINE)
    print("官方数据一次性探查  inspect_official_data.py")
    print(f"root  = {args.root}")
    print(f"xlsx  = {args.xlsx or '(自动搜)'}")
    print(f"sheet = {args.sheet}")
    print(_LINE)

    _safe(block_env)
    _safe(lambda: block_locate(args.root, args.xlsx, args.sheet))
    for fn in (block_columns, block_hit, block_value_counts, block_parse_check,
               block_acr, block_main_vs_sub, block_malignancy_wu, block_slice):
        _safe(fn)
    _safe(lambda: block_sample_dry_run(args.root))
    _safe(lambda: block_mask(args.root, args.mask_scan_limit))

    print("\n" + _LINE)
    print("探查完成。请把本文件全文（inspect_output.txt）贴回。")
    print(_LINE)


if __name__ == "__main__":
    main()
