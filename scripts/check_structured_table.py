# -*- coding: utf-8 -*-
"""Q7 核验脚本:统计官方(新)结构化数据表的行数/临床列非空率/四头 class_counts。

只读不改。产出 JSON 报告 + 控制台摘要。
用法(云端容器内):
    python scripts/check_structured_table.py --data-root /2026aicompetition/datasets/training/annotation
    # xlsx 自动探测;多个 xlsx 则全部跑并分列
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

# 与 official_loader._FIELD_ALIASES 同源的核心列候选名(别名取自 src/dataset/official_loader.py)
HEAD_FIELDS = {
    "malignancy": ["良恶性", "恶性", "ROI良恶性"],
    "tirads": ["TI-RADS", "TIRADS", "TI-RADS分级", "甲状腺影像报告和数据系统"],
    "ete": ["甲状腺外侵犯", "甲状腺 外侵犯", "外侵犯"],
    "location": ["病灶位置", "位置", "病灶部位"],
    "t_clinical": ["T分期", "T分级", "临床T"],
    "pathology": ["病理结果-组织学", "病理"],
    "roi_name": ["RoiName", "ROI名称"],
    "patient": ["PatientId", "患者ID"],
}
OLD_VALUES = {  # 旧表口径,供对拍
    "m_rows_total": 4877,
    "ete_class_counts": [3906, 267, 67, 5],
    "malignancy_total": 3906,  # 恶性行数口径(旧表恶性3906为ETE class0来源)
}


def _resolve_columns(columns):
    """归一化匹配:去空格+全角转半角+小写。与 official_loader._resolve_columns 同法。"""
    import unicodedata

    def norm(s):
        s = unicodedata.normalize("NFKC", str(s))
        return s.replace(" ", "").replace("　", "").lower()

    table = {}
    for idx, col in enumerate(columns):
        table[norm(col)] = idx
    resolved = {}
    for field, aliases in HEAD_FIELDS.items():
        for alias in aliases:
            key = norm(alias)
            if key in table:
                resolved[field] = columns[table[key]]
                break
    return resolved


def _value_counts_series(series):
    return series.astype(str).str.strip().value_counts(dropna=False)


def analyze_xlsx(path):
    # header=1:官方 xlsx 合并单元格表头,真实列名在第2行(2026-08-13 云端核实)
    df = pd.read_excel(path, header=1)
    resolved = _resolve_columns(df.columns)
    out = {
        "xlsx": os.path.basename(path),
        "n_rows": len(df),
        "resolved_columns": {k: str(v) for k, v in resolved.items()},
        "unresolved_fields": [f for f in HEAD_FIELDS if f not in resolved],
    }

    def col(name):
        return resolved.get(name)

    # --- 行数口径 ---
    if col("roi_name") is not None:
        roi = df[col("roi_name")].astype(str).str.strip()
        out["roi_counts"] = roi.value_counts().head(10).to_dict()
        m_mask = roi == "M"
        out["m_rows"] = int(m_mask.sum())

    # --- 良恶性 ---
    if col("malignancy") is not None:
        vc = _value_counts_series(df.loc[m_mask, col("malignancy")]) if col("roi_name") else _value_counts_series(df[col("malignancy")])
        out["malignancy_counts_M"] = {str(k): int(v) for k, v in vc.items()}

    # --- ETE 4类分布(M行口径,与训练侧判据 is_main 一致) ---
    if col("ete") is not None and col("roi_name") is not None:
        ete = df.loc[m_mask, col("ete")].astype(str).str.strip()
        # 官方编码:1无 2包膜 3软组织 4血管;数字串与中文都可能出现
        ete_num = ete.str.extract(r"^([0-4])")[0]
        vc = ete_num.value_counts(dropna=False)
        out["ete_class_counts_M"] = {str(k): int(v) for k, v in vc.items()}

    # --- 位置 / T / TI-RADS / 病理 非空率(M行) ---
    for field in ("location", "t_clinical", "tirads", "pathology"):
        if col(field) is not None and col("roi_name") is not None:
            s = df.loc[m_mask, col(field)]
            nonnull = int(s.notna().sum() - (s.astype(str).str.strip() == "").sum())
            out[f"{field}_nonempty_M"] = nonnull
            out[f"{field}_nonempty_ratio_M"] = round(nonnull / max(1, int(m_mask.sum())), 4)
            vc = _value_counts_series(s).head(12)
            out[f"{field}_value_counts_M"] = {str(k): int(v) for k, v in vc.items()}

    # --- 患者数 ---
    if col("patient") is not None:
        out["n_patients"] = int(df[col("patient")].nunique())

    # --- 对拍旧值 ---
    diffs = []
    if "ete_class_counts_M" in out:
        new = [out["ete_class_counts_M"].get(str(i), 0) for i in range(1, 5)]
        old = OLD_VALUES["ete_class_counts"]
        if list(new) != old:
            diffs.append(f"ete_class_counts: old={old} new={new}")
    if "m_rows" in out and out["m_rows"] != OLD_VALUES["m_rows_total"]:
        diffs.append(f"m_rows: old={OLD_VALUES['m_rows_total']} new={out['m_rows']}")
    out["vs_old_diffs"] = diffs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="q7_structured_table_report.json")
    args = ap.parse_args()

    xlsx_files = sorted(glob.glob(os.path.join(args.data_root, "**", "*.xlsx"), recursive=True))
    if not xlsx_files:
        print(f"[FATAL] no xlsx under {args.data_root}")
        return 2

    results = []
    for p in xlsx_files:
        try:
            r = analyze_xlsx(p)
        except Exception as e:  # noqa: BLE001
            r = {"xlsx": os.path.basename(p), "error": repr(e)}
        results.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=2))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"files": results, "old_reference": OLD_VALUES}, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] {len(results)} file(s) -> {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
