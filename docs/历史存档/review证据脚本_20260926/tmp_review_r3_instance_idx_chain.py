# -*- coding: utf-8 -*-
"""R3 证据 3:instance_idx 生产链末端 —— 真链路下它到底能不能取到非 0 值。

攻击面(任务书原话):**B 路声称"同 GT 病灶 split 出的多实例=不同 idx"在真实
前向里到底成不成立**?

链路:expand_roi_samples → RoiDataset → roi_collate → _meta_rows_from_collate
      → forward_roi(roi_outputs)→ collect_roi_records → aggregate_by_roi_instance

判据:roi_outputs 里有没有 instance_idx 键;expand 后的样本里有没有。
"""
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.training.collect_roi_records import collect_roi_records  # noqa: E402


def fake_roi(lid, inst=None, is_main=1):
    d = {"lesion_id": lid, "patient_id": "P", "is_main": is_main,
         "malignancy": torch.tensor([0.0, 3.0]),
         "location_probs": [0.1] * 7, "ete_probs": [0.25] * 4,
         "ete_logits": [0.0] * 4, "t_conf": 0.7, "tirads_probs": [0.2] * 5}
    if inst is not None:
        d["instance_idx"] = inst
    return d


def lab(inst=None, gt=1):
    d = {"gt_malignancy": gt, "valid_malignancy": 1, "gt_lesion_loc": 0,
         "valid_lesion_loc": 1, "gt_extrathy_ext": 0, "valid_extrathy_ext": 1,
         "gt_clin_t_stage": 0, "valid_clin_t_stage": 1,
         "gt_tirads": 0, "valid_tirads": 1}
    if inst is not None:
        d["instance_idx"] = inst
    return d


def main():
    r = collect_roi_records([fake_roi("L1"), fake_roi("L1")], [lab(), lab()])
    print("[1] 无 instance_idx(真链路现状)→",
          [(x["lesion_id"], x["instance_idx"]) for x in r])
    r2 = collect_roi_records([fake_roi("L1", 0), fake_roi("L1", 1)], [lab(), lab()])
    print("[2] roi 侧 0/1(合同声称的路径)→",
          [(x["lesion_id"], x["instance_idx"]) for x in r2])
    r3 = collect_roi_records([fake_roi("L1"), fake_roi("L1")], [lab(0), lab(1)])
    print("[3] labels 侧 0/1(次优先路径)  →",
          [(x["lesion_id"], x["instance_idx"]) for x in r3])
    try:
        collect_roi_records([fake_roi("L1", torch.tensor([0, 1]))], [lab()])
        print("[4] batch 张量:未拦(缺陷)")
    except ValueError as e:
        print("[4] batch 张量 fail-loud ✓:", str(e)[:60], "...")

    # ── 聚合对比:同 lesion 两条独立实例,实例级 vs 帧级 ────────────────
    from src.training.metrics_official import (
        aggregate_by_roi_instance, aggregate_by_lesion)
    recs = []
    for p, inst in ((0.9975, 0), (0.0025, 1)):
        d = {**r2[len(recs)], "mal_prob": p}
        recs.append(d)
    a_inst = aggregate_by_roi_instance(recs)
    a_les = aggregate_by_lesion([{k: v for k, v in rec.items()
                                  if k != "instance_idx"} for rec in recs])
    print(f"\n[5] 双实例 0.9975/0.0025 → 实例级 {len(a_inst)} 行 "
          f"{[round(x['mal_prob'], 4) for x in a_inst]} | "
          f"帧级 {len(a_les)} 行 {[round(x['mal_prob'], 4) for x in a_les]}")
    print("    → 无 instance_idx 时两者逐位等价(单实例退化,尺子修复失效)")

    # ── 模型侧/数据面是否产这个键 ───────────────────────────────────
    m_src = open("src/models/multi_task_model.py", encoding="utf-8").read()
    i = m_src.index("def forward_roi")
    seg = m_src[i:i + 9000]
    print(f"\n[6] forward_roi 全文 'instance_idx' 出现次数: {seg.count('instance_idx')}")
    keys = sorted(set(re.findall(r'"(\w+)":', seg[seg.index("roi_outputs = []"):])))
    print("    forward_roi 产出的 roi_outputs 键:", keys)
    d_src = open("src/dataset/roi_dataset.py", encoding="utf-8").read()
    print("[7] roi_dataset.py 'instance_idx' 出现次数:", d_src.count("instance_idx"))
    print("[8] official_loader.py 'instance_idx' 出现次数:",
          open("src/dataset/official_loader.py", encoding="utf-8").read().count("instance_idx"))


if __name__ == "__main__":
    main()
