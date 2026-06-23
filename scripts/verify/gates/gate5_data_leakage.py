import os
import json


def _scan_patient_ids(data_dir):
    """扫描 data_dir 下所有 sample_* 子目录的 meta.json,抽取 patient_id 集合。"""
    ids = set()
    if not os.path.exists(data_dir):
        return None  # 目录不存在,返回 None 表示跳过
    for folder in os.listdir(data_dir):
        meta_p = os.path.join(data_dir, folder, "meta.json")
        if os.path.exists(meta_p):
            with open(meta_p, "r") as f:
                meta = json.load(f)
                if "patient_id" in meta:
                    ids.add(str(meta["patient_id"]))
    return ids


def gate5_data_leakage():
    """图像级数据泄漏校验:train↔val + holdout↔trainval 双检查。"""
    try:
        train_ids = _scan_patient_ids("data/train")
        val_ids = _scan_patient_ids("data/val")
        holdout_ids = _scan_patient_ids("data/holdout")

        if train_ids is None or val_ids is None:
            print(
                "⚠ GATE 5: 图像隔离检查跳过(本机无 data/train 或 data/val 目录)。"
            )
            return True

        # 1. train ↔ val 互斥(纯随机图像级划分,每张图独立身份)
        overlap_trainval = train_ids.intersection(val_ids)
        assert len(overlap_trainval) == 0, (
            f"CRITICAL DATA LEAKAGE (train↔val): {len(overlap_trainval)} 张图像同时出现在 train 和 val 中!"
        )
        print("✔ GATE 5: train↔val 图像级 0 重叠验证通过。")

        # 2. holdout ↔ trainval 互斥(holdout 来自 TN3K 独立 test-image/)
        if holdout_ids is not None:
            trainval_ids = train_ids | val_ids
            overlap_holdout = holdout_ids.intersection(trainval_ids)
            assert len(overlap_holdout) == 0, (
                f"CRITICAL DATA LEAKAGE (holdout): {len(overlap_holdout)} 张图像同时出现在 holdout 和 train/val 中!"
            )
            print("✔ GATE 5: holdout↔trainval 图像级 0 重叠验证通过。")
        else:
            print("⚠ GATE 5: holdout 图像隔离检查跳过(本机无 data/holdout 目录)。")

        return True
    except Exception as e:
        print(f"✘ GATE 5 Failed: {e}")
        return False
