import os
import json


def gate5_data_leakage():
    """Data Leakage & Patient ID Isolation Check."""
    try:
        train_dir = "data/train"
        val_dir = "data/val"

        if os.path.exists(train_dir) and os.path.exists(val_dir):
            train_patients = set()
            val_patients = set()

            # 扫描训练集元数据
            for folder in os.listdir(train_dir):
                meta_p = os.path.join(train_dir, folder, "meta.json")
                if os.path.exists(meta_p):
                    with open(meta_p, "r") as f:
                        meta = json.load(f)
                        if "patient_id" in meta:
                            train_patients.add(str(meta["patient_id"]))

            # 扫描验证集元数据
            for folder in os.listdir(val_dir):
                meta_p = os.path.join(val_dir, folder, "meta.json")
                if os.path.exists(meta_p):
                    with open(meta_p, "r") as f:
                        meta = json.load(f)
                        if "patient_id" in meta:
                            val_patients.add(str(meta["patient_id"]))

            overlap = train_patients.intersection(val_patients)
            # 强行断言：在工业级数据隔离流水线中，训练集与验证集之间的患者重叠必须为 0，否则立即抛错熔断！
            assert (
                len(overlap) == 0
            ), f"CRITICAL DATA LEAKAGE: {len(overlap)} overlapping patients found between train and val splits!"
            print("✔ GATE 5: Patient ID isolation verified successfully (0 overlap).")
        else:
            print(
                "⚠ GATE 5: Patient isolation check skipped (Data folders not found on this machine)."
            )
        return True
    except Exception as e:
        print(f"✘ GATE 5 Failed: {e}")
        return False
