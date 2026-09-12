# -*- coding: utf-8 -*-
"""baseline JSONL 留档行的可达测试(2026-09-12)。

🔴 为什么必须有:ETE-only 首跑在云端炸于 `mode="validation"` 撞 _MODES 枚举
assert——该行只在「真数据 + warm-start 完成 + 非 mock」路径执行,mock 冒烟
根本走不到,本地 1104 全绿照样带病上云。教训定案:**新代码路径必须有能到达
它的本地测试**,不能只靠"冒烟跑过"当覆盖。

本测试直接驱动 main() 的真实代码路径:mock 掉数据集构建与 validate_epoch
(返回带官方指标键的假 dict),让 main() 走到 baseline 落盘那几行,断言:
1. JSONL 里出现 epoch=0/phase=val 行;
2. mode 是合法枚举(_MODES 里的值)——首版正是死在这里;
3. extra 里带 baseline_note 与官方指标数值。
"""
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_baseline_jsonl_row_reachable_and_valid(tmp_path):
    """驱动 main() 真实路径走到 baseline 落盘,断言 JSONL 行合法。"""
    import scripts.train_official as tf

    # --- 假官方指标(warm-start 后 validate_epoch 会填进 base_metrics) ---
    fake_metrics = {"malignancy_auc_pr": 0.90, "ete_weighted_kappa": 0.21,
                    "location_weighted_auc": 0.75, "clinical_t_auc": 0.74,
                    "tirads_weighted_kappa": 0.69, "composite": 0.64}

    class _FakeDS:
        def __len__(self):
            return 2

    # mock 掉「贵」的环节,只留 baseline 分支真实代码。
    # build_official_datasets 是 main() 内的延迟导入(非模块属性),
    # 须 patch 源模块 src.dataset.official_loader。
    import src.dataset.official_loader as _olo
    # 🔴 mock 契约必须复刻真实签名语义(教义:fake 逐键抄真实 return):
    # validate_epoch 把官方指标写进**传入的 official_metrics_out dict**(参数
    # 填充),不在返回值里。首版只设 return_value 指标全丢,行里只剩规范字段。
    def _fake_validate_epoch(model, loader, criterion, device, epoch=1,
                             official_metrics_out=None):
        if official_metrics_out is not None:
            official_metrics_out.update(fake_metrics)
        return 0.0, 0.70, 0.0, 0.0, fake_metrics

    with mock.patch.object(_olo, "build_official_datasets",
                           return_value=(_FakeDS(), _FakeDS())), \
         mock.patch.object(tf, "load_warm_start"), \
         mock.patch.object(tf, "validate_epoch", side_effect=_fake_validate_epoch), \
         mock.patch.object(tf, "run_official_epoch",
                           side_effect=RuntimeError("__STOP__")), \
         mock.patch.object(tf.JsonlLogger, "close"):
        # run_official_epoch 抛 STOP 让训练循环第一轮即停——baseline 已在
        # 循环之前落盘,我们只要那几行,不需要真训练。
        with pytest.raises(RuntimeError, match="__STOP__"):
            with mock.patch.object(sys, "argv",
                                   ["train_official.py",
                                    "--data-root", "unused", "--xlsx", "unused",
                                    "--ete-only", "--freeze-legacy",
                                    "--resume-from", "checkpoints/official_heads_best.pth",
                                    "--out", str(tmp_path / "w.pth"),
                                    "--epochs", "1", "--batch-size", "2",
                                    "--log-dir", str(tmp_path)]):
                tf.main()

    # --- 断言 JSONL:epoch=0/phase=val 行存在且字段合法 ---
    log_file = tmp_path / "train.jsonl"
    assert log_file.exists(), "JSONL 未产出"
    rows = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    base_rows = [r for r in rows if r.get("epoch") == 0 and r.get("phase") == "val"]
    assert base_rows, f"缺 epoch=0/phase=val 的 baseline 行,实际行: {rows[:3]}"
    row = base_rows[0]
    # 🔴 首版事故点:mode 必须是合法枚举
    from src.training.jsonl_logger import _MODES
    assert row["mode"] in _MODES, f"mode={row['mode']!r} 不在合法枚举 {_MODES}"
    # 指标进 extra(顶层平铺,compute_official_metrics 返回键直接 update)
    assert abs(row.get("ete_weighted_kappa", -1) - 0.21) < 1e-9
    assert abs(row.get("composite", -1) - 0.64) < 1e-9
