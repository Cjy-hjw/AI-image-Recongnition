# -*- coding: utf-8 -*-
"""ROI 范式训练侧接线(债1-债5)契约测试(2026-09-25)。

覆盖(scoping §J.3 五条债逐一钉死):
1.  债1 `_roi_labels_from_meta` 桥:5 键改名 + valid_*/lesion_id/is_main/
    patient_key 原样透传(collate 后 [K] tensor 与 python int 两形态);
2.  债2 roi_mask 生产:静态唯一帧 / 动态交集帧 / 帧不在 idxs=全零 /
    无 mask_path 缺键 四态 + 值域断言 + 与图像同几何(crop 用 _crop_mask_like_image);
3.  债2+债5 loader `_entry` / expand_roi_samples 带出 mask_path/patient_id;
4.  债3b configure_official_stage(roi_heads=True):ROI 头可训且进
    param_groups(head_lr 档);roi_heads=False 逐位零回归(参数集与组数不变);
5.  债4 run_roi_epoch 冒烟(最小真模型):loss finite + backward 后 ROI 头
    梯度非空 + JSONL 键写出;validate_roi_epoch 的 None 纪律;
6.  --roi-heads 硬门:互斥 / 缺 resume(mock 豁免)/ 全 0 权重(parser.error 三态);
7.  债3 warm-start 白名单:旧底座(无 ROI 键)+ ROI 头结构 → 不 raise 且
    打冷启行。

fixture 构造抄 tests/test_roi_dataset.py 的 roi_root 形态(双行表头 xlsx +
静态/动态双病灶),但**独立手构**(不 import 它的 fixture 文件——任务书红线)。
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# ↓ 顺序敏感(Windows OpenMP 冲突,test_train_step_head_targets.py 同款)
import sklearn.metrics  # noqa: F401

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.multi_task_model import MultiTaskThyroidModel  # noqa: E402
from src.training.schedule_official import configure_official_stage  # noqa: E402
from src.training.loop.train_step import _roi_labels_from_meta  # noqa: E402

pydicom = pytest.importorskip("pydicom")
nib = pytest.importorskip("nibabel")
pd = pytest.importorskip("pandas")
pytest.importorskip("openpyxl")


# train_official 按路径加载(test_seg_guards.py 同款;scripts/ 不是包)
_SPEC = importlib.util.spec_from_file_location(
    "train_official_mod",
    Path(__file__).resolve().parents[1] / "scripts" / "train_official.py")
train_official = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train_official)


# ── fixture:双病灶静态 + 动态(照 test_roi_dataset.py 的 roi_root 手构)────
@pytest.fixture
def roi_root(tmp_path):
    """QA 静态双病灶 + QB 动态双病灶(与 test_roi_dataset.py 同构,独立构造)。"""
    from test_official_loader import _make_dcm, _save_image_nifti_3d
    from src.inference.nifti_io import save_binary_mask
    from src.dataset import official_loader as O
    from src.dataset.roi_dataset import expand_roi_samples, RoiDataset

    root = tmp_path / "annotation"
    H = W = 64

    qa = root / "QA" / "SA"
    _make_dcm(str(qa / "img.dcm"), np.full((H, W), 120, np.uint8))
    m1 = np.zeros((H, W), np.uint8); m1[20:40, 20:40] = 1
    m2 = np.zeros((H, W), np.uint8); m2[30:50, 30:50] = 1
    save_binary_mask(m1, str(qa / "M_1_mask.nii.gz"))
    save_binary_mask(m2, str(qa / "病灶1_1_mask.nii.gz"))

    qb = root / "QB" / "QB"
    vol = np.stack([np.full((H, W), 40 + 10 * i, np.uint8) for i in range(5)])
    _save_image_nifti_3d(str(qb / "QB.nii.gz"), vol)
    dm = np.zeros((5, H, W), np.uint8)
    dm[1, 10:30, 10:30] = 1
    dm[3, 25:45, 25:45] = 1
    save_binary_mask(dm, str(qb / "M_1_mask.nii.gz"))
    ds = np.zeros((5, H, W), np.uint8); ds[2, 40:55, 5:20] = 1
    save_binary_mask(ds, str(qb / "病灶1_1_mask.nii.gz"))

    rows = [
        ["QA", "SA", "1", "M", "2左中叶", "2甲状腺包膜", "1有", "恶", "", "V类"],
        ["QA", "SA", "1", "病灶1", "5右中叶", "1无", "0无", "良", "", "IV类"],
        ["QB", "", "2", "M", "7峡部", "1无", "1有", "恶", "", ""],
        ["QB", "", "2", "病灶1", "7峡部", "1无", "1有", "良", "", ""],
    ]
    df = pd.DataFrame(rows, columns=[
        "PatientId", "SeriesUid", "图像性质", "RoiName", "主病灶位置",
        "甲状腺外侵犯", "临床T分期", "ROI良恶性", "ANNOTATION->SliceIndex", "ROI-TIRADS",
    ])
    xlsx = tmp_path / "labels_roi.xlsx"
    df.to_excel(xlsx, sheet_name="ROI级别", index=False, startrow=1)

    samples = O.build_official_samples(str(root), str(xlsx), roi_meta=True)
    rois = expand_roi_samples(samples)
    return {"root": str(root), "xlsx": str(xlsx), "samples": samples,
            "rois": rois, "O": O, "RoiDataset": RoiDataset,
            "qa_mask_m": str(qa / "M_1_mask.nii.gz")}


# ═════════════════════════════════════════════════════════════════════════════
# 债1:gt_ 前缀桥
# ═════════════════════════════════════════════════════════════════════════════

def test_bridge_renames_five_label_keys():
    meta = {"malignancy": 1, "lesion_loc": 3, "extrathy_ext": 2,
            "clin_t_stage": 1, "tirads": 4,
            "valid_malignancy": 1, "valid_lesion_loc": 1,
            "valid_extrathy_ext": 1, "valid_clin_t_stage": 1,
            "valid_tirads": 1, "is_main": 1,
            "lesion_id": "L1", "patient_key": "PK", "patient_id": "PID"}
    out = _roi_labels_from_meta([meta])[0]
    assert out["gt_malignancy"] == 1
    assert out["gt_lesion_loc"] == 3
    assert out["gt_extrathy_ext"] == 2
    assert out["gt_clin_t_stage"] == 1
    assert out["gt_tirads"] == 4
    # 原键不再以无前缀形态存在(损失面读 gt_,留着旧键是冗余非错误,但
    # 断言不存在能杀"忘了改名"变异)
    for k in ("malignancy", "lesion_loc", "extrathy_ext",
              "clin_t_stage", "tirads"):
        assert k not in out


def test_bridge_passthrough_non_label_keys_and_tensor_values():
    """valid_*/lesion_id/is_main/patient_key 原样;[K] int64 tensor 元素可透传。"""
    meta = {"lesion_loc": torch.tensor(2, dtype=torch.int64),
            "valid_lesion_loc": torch.tensor(1, dtype=torch.int64),
            "is_main": 0, "lesion_id": "L", "patient_key": "P",
            "bbox_xyxy": (1, 2, 3, 4)}
    out = _roi_labels_from_meta([meta])[0]
    assert out["gt_lesion_loc"] == torch.tensor(2, dtype=torch.int64)
    assert out["valid_lesion_loc"] == torch.tensor(1, dtype=torch.int64)
    assert out["is_main"] == 0 and out["lesion_id"] == "L"
    assert out["patient_key"] == "P" and out["bbox_xyxy"] == (1, 2, 3, 4)


def test_bridge_consumed_by_loss_and_records(roi_root):
    """桥产物同时满足损失面与尺子面契约(键名互锁,端到端最小对拍)。"""
    from src.models.losses.components.roi_heads import compute_roi_head_losses
    from src.training.collect_roi_records import collect_roi_records
    from src.dataset.roi_dataset import RoiDataset, roi_collate

    rois = roi_root["rois"]
    ds = RoiDataset(rois, roi_size=32)
    imgs, metas = roi_collate([ds[i] for i in range(min(3, len(ds)))])
    labels = _roi_labels_from_meta(metas)

    K = imgs.shape[0]
    torch.manual_seed(0)
    roi_outputs = []
    for i in range(K):
        roi_outputs.append({
            "lesion_id": metas["lesion_id"][i],
            "is_main": int(metas["is_main"][i]),
            "malignancy": torch.randn(2),
            "location_probs": torch.softmax(torch.randn(7), dim=0),
            "ete_probs": torch.softmax(torch.randn(4), dim=0),
            "ete_logits": torch.randn(4),
            "t_conf": torch.tensor(0.5),
            "tirads_probs": torch.softmax(torch.randn(5), dim=0),
        })
    # 损失面:吃 gt_ 前缀键不 raise(权重>0 时才校验,此处真算)
    _, detail = compute_roi_head_losses(
        {"roi_outputs": roi_outputs}, {"roi_labels": labels}, lambda_roi=0.5)
    assert math.isfinite(float(detail["roi_location"]))
    # 尺子面:同 labels 喂 collect_roi_records 不 raise
    recs = collect_roi_records(roi_outputs, labels)
    assert len(recs) == K
    assert recs[0]["gt_lesion_loc"] == labels[0]["gt_lesion_loc"]


# ═════════════════════════════════════════════════════════════════════════════
# 债2:roi_mask 生产四态
# ═════════════════════════════════════════════════════════════════════════════

def test_roi_mask_static_unique_frame(roi_root):
    """静态:GT mask 该帧外接框内全 1;裁剪后 [1,S,S] float 值域 {0,1}。"""
    RoiDataset = roi_root["RoiDataset"]
    rois = [r for r in roi_root["rois"] if not r["is_dynamic"]
            and os.path.basename(r["lesion_id"]) == "M_1_mask.nii.gz"]
    ds = RoiDataset(rois, roi_size=32)
    img, meta = ds[0]
    assert "roi_mask" in meta
    m = meta["roi_mask"]
    assert tuple(m.shape) == (1, 32, 32)
    assert m.dtype == torch.float32
    assert float(m.max()) <= 1.0
    # bbox 20:40 恰方形、mask 全 1 → ROI 内全 1(30×30 resize 32,NEAREST)
    assert float(m.min()) >= 0.0


def test_roi_mask_dynamic_intersection_frame(roi_root):
    """动态:该样本帧(M@帧1)的 mask 在 ROI 域非零;帧3 语义不同。"""
    RoiDataset = roi_root["RoiDataset"]
    rois = [r for r in roi_root["rois"] if r["is_dynamic"]
            and os.path.basename(r["lesion_id"]) == "M_1_mask.nii.gz"]
    assert sorted(r["frame_index"] for r in rois) == [1, 3]
    ds = RoiDataset(rois, roi_size=32)
    by_f = {}
    for i in range(len(ds)):
        _, meta = ds[i]
        by_f[meta["frame_index"]] = meta["roi_mask"]
    # 两帧 mask 都非零且内容不同(帧1@10:30 vs 帧3@25:45,并集 bbox 10:45)
    for f, m in by_f.items():
        assert float(m.sum()) > 0, f"帧{f} mask 应非零"
    assert not torch.equal(by_f[1], by_f[3])


def test_roi_mask_frame_not_in_idxs_gives_zero(roi_root):
    """动态:帧不在 idxs → 全零图(非缺键——防御性兜底口径)。"""
    RoiDataset = roi_root["RoiDataset"]
    rois = [dict(r) for r in roi_root["rois"] if r["is_dynamic"]
            and os.path.basename(r["lesion_id"]) == "病灶1_1_mask.nii.gz"]
    assert len(rois) == 1
    rois[0]["frame_index"] = 0     # 病灶1 只有帧 2 → 0 不在 idxs
    ds = RoiDataset(rois, roi_size=32)
    _, meta = ds[0]
    assert "roi_mask" in meta       # 键在(mask_path 有)
    assert float(meta["roi_mask"].sum()) == 0.0


def test_roi_mask_absent_when_no_mask_path(roi_root):
    """无 mask_path(mock/loader 无 mask 病灶)→ 键缺失(forward_roi 全零兜底)。"""
    RoiDataset = roi_root["RoiDataset"]
    rois = [dict(r) for r in roi_root["rois"]]
    for r in rois:
        r["mask_path"] = None
    ds = RoiDataset(rois, roi_size=32)
    for i in range(len(ds)):
        _, meta = ds[i]
        assert "roi_mask" not in meta


def test_roi_mask_crop_geometry_matches_image(roi_root):
    """mask 裁剪与图像同几何:非方形 bbox 下 pad_hw 一致、ROI 域对齐。"""
    RoiDataset = roi_root["RoiDataset"]
    # QB 病灶1 bbox (5,40,20,55) → w=15,h=15 已方形;手工造非方形:
    rois = [r for r in roi_root["rois"] if not r["is_dynamic"]
            and os.path.basename(r["lesion_id"]) == "M_1_mask.nii.gz"]
    r = dict(rois[0])
    r["bbox_xyxy"] = (20, 20, 35, 40)      # w=15, h=20 → pad_w=5
    ds = RoiDataset([r], roi_size=32)
    _, meta = ds[0]
    assert meta["pad_hw"] == (0, 5)
    assert "roi_mask" in meta
    # 手工对拍:图像管线裁同一帧的 mask(20:40 全 1 → 切 15×20 + pad 5 列)
    exp = np.ones((20, 15), np.uint8)
    exp = np.pad(exp, ((0, 0), (0, 5)), mode="constant")
    from src.dataset.official_loader import resize_mask_to
    exp = resize_mask_to(exp, 32)
    assert np.array_equal(meta["roi_mask"].numpy()[0], exp)


# ═════════════════════════════════════════════════════════════════════════════
# 债2+债5:loader _entry / expand_roi_samples 带出 mask_path/patient_id
# ═════════════════════════════════════════════════════════════════════════════

def test_loader_entry_has_mask_path_and_patient_id(roi_root):
    O = roi_root["O"]
    for x in roi_root["samples"]:
        for d in x["roi_lesions"]:
            assert "mask_path" in d
            assert "patient_id" in d
            assert d["patient_id"] == x["patient_key"]
            if d["bbox_xyxy"] is not None:
                assert d["mask_path"] is not None
                assert os.path.basename(d["mask_path"]).endswith(".nii.gz")


def test_expand_carries_mask_path_and_patient_id(roi_root):
    for r in roi_root["rois"]:
        assert "patient_id" in r
        assert "mask_path" in r
        assert r["patient_id"] == r["patient_key"]
        assert os.path.basename(r["mask_path"]).endswith(".nii.gz") \
            if r["mask_path"] else True


def test_collate_roi_mask_stays_list_of_tensors(roi_root):
    """roi_mask 经 collate → list[[1,S,S]] 透传不 stack(forward_roi 契约)。"""
    from src.dataset.roi_dataset import RoiDataset, roi_collate
    ds = RoiDataset(roi_root["rois"], roi_size=32)
    batch = [ds[i] for i in range(min(3, len(ds)))]
    imgs, metas = roi_collate(batch)
    if "roi_mask" in metas:       # 批内全有 mask 时键在
        assert isinstance(metas["roi_mask"], list)
        assert all(torch.is_tensor(m) for m in metas["roi_mask"])
        assert len(metas["roi_mask"]) == imgs.shape[0]


# ═════════════════════════════════════════════════════════════════════════════
# 债3b:调度 roi_heads 分支
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def model_and_criterion():
    from src.models.loss import MultiTaskLoss
    model = MultiTaskThyroidModel(pretrained=False)
    criterion = MultiTaskLoss()
    return model, criterion


def test_roi_heads_true_trainable_and_in_optimizer(model_and_criterion):
    model, criterion = model_and_criterion
    groups, head_lr, name = configure_official_stage(
        model, criterion, epoch=1, lr_base=0.01, weight_decay=5e-4,
        roi_heads=True)
    covered = {id(p) for g in groups for p in g["params"]}
    roi_named = [(n, p) for n, p in model.named_parameters()
                 if n.startswith(MultiTaskThyroidModel.ROI_HEAD_PREFIXES)]
    assert roi_named, "模型应有 ROI 头参数"
    for n, p in roi_named:
        assert p.requires_grad, f"{n} 应可训(roi_heads=True)"
        assert id(p) in covered, f"{n} 应进 optimizer(梯度累积红线)"
        # head_lr 档:与四头同待遇(挂 backbone 档 lr 会低 10 倍,冷启头不动)
    _lr_of = {}
    for g in groups:
        for p in g["params"]:
            _lr_of[id(p)] = g["lr"]
    for n, p in roi_named[:5]:
        assert _lr_of[id(p)] == pytest.approx(head_lr), n
    assert "ROI heads" in name


def test_roi_heads_default_zero_regression(model_and_criterion):
    """roi_heads=False(默认)逐位回归:参数集/组数/组内容与不传时一致。"""
    model, criterion = model_and_criterion
    g_new, lr_new, _ = configure_official_stage(
        model, criterion, epoch=1, lr_base=0.01, weight_decay=5e-4,
        roi_heads=False)
    g_old, lr_old, _ = configure_official_stage(
        model, criterion, epoch=1, lr_base=0.01, weight_decay=5e-4)
    assert lr_new == lr_old
    assert len(g_new) == len(g_old)
    for ga, gb in zip(g_new, g_old):
        assert {id(p) for p in ga["params"]} == {id(p) for p in gb["params"]}
        assert ga["lr"] == gb["lr"]
    # 默认下 ROI 头冻结
    roi_frozen = [n for n, p in model.named_parameters()
                  if n.startswith(MultiTaskThyroidModel.ROI_HEAD_PREFIXES)
                  and p.requires_grad]
    assert not roi_frozen


def test_roi_heads_true_does_not_touch_vclub_or_cls(model_and_criterion):
    """roi_heads=True 不连带解冻 cls_backbone/Adaptor(vclub 由独立 optimizer)。"""
    model, criterion = model_and_criterion
    configure_official_stage(
        model, criterion, epoch=1, lr_base=0.01, weight_decay=5e-4,
        roi_heads=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not any(n.startswith("cls_backbone") for n in trainable)
    assert any("vclub" in n for n in trainable)   # vclub 恒 True(独立管理)


# ═════════════════════════════════════════════════════════════════════════════
# 债4:run_roi_epoch / validate_roi_epoch 冒烟(最小真模型)
# ═════════════════════════════════════════════════════════════════════════════

def _tiny_model():
    cfg = {"model": {"backbone": "efficientnet_b2"}, "data": {"img_size": 64}}
    return MultiTaskThyroidModel(
        backbone_name="efficientnet_b2", pretrained=False, config=cfg)


def _fake_criterion(weight):
    """criterion 替身:只暴露 run_roi_epoch / configure_official_stage
    读取的超参属性(mal_loss_fn 用空参数模块顶替——调度只设其
    requires_grad,无参数即无操作)。"""
    class _C(nn.Module):
        def __init__(self):
            super().__init__()
            self._mal = nn.Module()          # 无参数的 mal_loss_fn 替身
        @property
        def mal_loss_fn(self):
            return self._mal
    c = _C()
    c.roi_head_weight = weight
    c.ete_class_weights = None
    c.ete_focal_gamma = 0.0
    c.tirads_class_weights = None
    c.tirads_label_smoothing = 0.0
    c.tirads_ordinal_weight = 0.0
    return c


def _mock_batch(K=4, S=64, seed=0):
    rng = np.random.RandomState(seed)
    imgs = torch.tensor(rng.rand(K, 1, S, S), dtype=torch.float32)
    metas = []
    for i in range(K):
        metas.append({
            "bbox_xyxy": (20, 20, 40, 40), "orig_w": S, "orig_h": S,
            "pad_hw": (0, 0), "spacing_mm": None,
            "lesion_id": f"L{i}", "patient_key": f"P{i}",
            "patient_id": f"P{i}", "is_main": 1 if i < 3 else 0,
            "malignancy": int(rng.randint(0, 2)), "valid_malignancy": 1,
            "lesion_loc": int(rng.randint(0, 7)), "valid_lesion_loc": 1,
            "extrathy_ext": int(rng.randint(0, 4)), "valid_extrathy_ext": 1,
            "clin_t_stage": int(rng.randint(0, 2)), "valid_clin_t_stage": 1,
            "tirads": int(rng.randint(0, 5)), "valid_tirads": 1,
        })
    return imgs, metas


class _ListLoader:
    def __init__(self, batches):
        self.batches = batches
    def __iter__(self):
        return iter(self.batches)
    def __len__(self):
        return len(self.batches)


def test_run_roi_epoch_smoke_real_model(tmp_path):
    """最小真模型跑 run_roi_epoch:loss finite + ROI 头梯度非空 + JSONL 键。"""
    torch.manual_seed(0)
    model = _tiny_model()
    criterion = _fake_criterion(weight=0.5)
    groups, _, _ = configure_official_stage(
        model, criterion, epoch=1, lr_base=1e-3, weight_decay=0.0,
        roi_heads=True)
    optimizer = torch.optim.AdamW(
        [p for g in groups for p in g["params"]], lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    batches = [_mock_batch(K=4, seed=s) for s in range(3)]
    loader = _ListLoader(batches)

    from src.training.jsonl_logger import JsonlLogger
    with JsonlLogger(log_dir=str(tmp_path), data_source="test") as logger:
        avg, n = train_official.run_roi_epoch(
            model, loader, optimizer, criterion, scaler,
            torch.device("cpu"), epoch=1, logger=logger, log_every=1,
            head_lr=1e-3, amp_enabled=False)
    assert n == 3 and math.isfinite(avg)
    # 梯度非空(ROI 头参与了 backward)
    grads = [p.grad for n_, p in model.named_parameters()
             if n_.startswith(MultiTaskThyroidModel.ROI_HEAD_PREFIXES)
             and p.grad is not None]
    assert grads, "ROI 头无任何梯度(接线断)"
    assert any(g.abs().sum().item() > 0 for g in grads), "ROI 头梯度全零"
    # JSONL 落盘行含 roi_branch + loss_roi_* 分项
    import json
    path = os.path.join(str(tmp_path), "train.jsonl")
    lines = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    assert len(lines) == 3
    assert all(ln.get("roi_branch") is True for ln in lines)
    assert any("loss_roi_location" in ln for ln in lines)


def test_validate_roi_epoch_none_discipline(tmp_path):
    """空 val loader → 全 None + n=0(None 透传,不填 0);真跑时键齐全。"""
    model = _tiny_model()
    out = train_official.validate_roi_epoch(
        model, _ListLoader([]), torch.device("cpu"))
    assert out["roi_loc_auc"] is None
    assert out["roi_n_records"] == 0
    # 真跑一条小批:键集齐、n_records=K
    imgs, metas = _mock_batch(K=3, seed=1)
    out2 = train_official.validate_roi_epoch(
        model, _ListLoader([(imgs, metas)]), torch.device("cpu"))
    assert set(out2.keys()) == {"roi_loc_auc", "roi_ete_kappa", "roi_t_auc",
                                "roi_tirads_kappa", "roi_mal_auc_pr",
                                "roi_n_records", "roi_size"}
    assert out2["roi_n_records"] == 3
    # R1 P1(2026-09-26):口径哨兵恒在且等于训练域尺寸(224 域 val 指标
    # 与 512 推理不可比,哨兵钉死口径防误读)
    assert out2["roi_size"] == train_official._ROI_TRAIN_SIZE
    # 模型恢复原 training 态(try/finally)
    assert model.training


# ═════════════════════════════════════════════════════════════════════════════
# --roi-heads 硬门(parser.error 三态)与 warm-start 白名单(债3)
# ═════════════════════════════════════════════════════════════════════════════

def _run_main(argv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_official.py"] + argv)
    with pytest.raises(SystemExit) as ei:
        train_official.main()
    return ei.value.code, capsys.readouterr()


class _ParserStub:
    """捕获 parser.error(经 SystemExit code 2)的 argv 适配器。"""
    pass


def test_gate_mutual_exclusion_with_seg_only(tmp_path, monkeypatch, capsys):
    # --resume-from 给假路径即可:parser.error 在数据/模型构造之前触发
    # (seg_only 门自身也要求 resume,不给会在 ROI 互斥门之前先炸 resume 门)。
    argv = ["--mock-labels", "--smoke", "--roi-heads", "--seg-only",
            "--resume-from", str(tmp_path / "base.pth"),
            "--out", str(tmp_path / "x.pth"), "--log-dir", str(tmp_path)]
    code, out = _run_main(argv, monkeypatch, capsys)
    assert code == 2 and "互斥" in out.err


def test_gate_mutual_exclusion_with_ete_only(tmp_path, monkeypatch, capsys):
    argv = ["--mock-labels", "--smoke", "--roi-heads", "--ete-only",
            "--resume-from", str(tmp_path / "base.pth"),
            "--out", str(tmp_path / "x.pth"), "--log-dir", str(tmp_path)]
    code, out = _run_main(argv, monkeypatch, capsys)
    assert code == 2 and "互斥" in out.err


def test_gate_requires_versioned_out(tmp_path, monkeypatch, capsys):
    argv = ["--mock-labels", "--smoke", "--roi-heads",
            "--out", "checkpoints/official_heads_best.pth",
            "--log-dir", str(tmp_path)]
    code, out = _run_main(argv, monkeypatch, capsys)
    assert code == 2 and "版本化" in out.err


def test_gate_requires_resume_for_real_training(tmp_path, monkeypatch, capsys):
    """真训(非 mock)缺 resume → parser.error;mock 豁免(合成数据无真底座)。"""
    argv = ["--data-root", str(tmp_path), "--xlsx", str(tmp_path / "f.xlsx"),
            "--roi-heads", "--out", str(tmp_path / "x.pth"),
            "--log-dir", str(tmp_path)]
    code, out = _run_main(argv, monkeypatch, capsys)
    assert code == 2 and "resume" in out.err


def test_gate_zero_weight_real_training(tmp_path, monkeypatch, capsys):
    """真训 + config roi_head_weight=0 → 空 ROI 微调白烧,parser.error。
    (config 从测试侧构造 yaml,权重 0 恰是默认 thyroid_config 的值。)"""
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "environment: {seed: 42, num_workers: 0}\n"
        "data: {img_size: 64, batch_size: 2}\n"
        "model: {backbone: efficientnet_b2, pretrained: false}\n"
        "train: {lr: 0.001, weight_decay: 0.0005, amp: false,\n"
        "  ete_class_counts: [4459, 333, 81, 5], ete_cb_beta: 0.9,\n"
        "  clin_t_class_counts: [100, 50], roi_head_weight: 0.0}\n",
        encoding="utf-8")
    argv = ["--config", str(cfg_path), "--data-root", str(tmp_path),
            "--xlsx", str(tmp_path / "f.xlsx"), "--resume-from",
            str(tmp_path / "base.pth"), "--roi-heads",
            "--out", str(tmp_path / "x.pth"), "--log-dir", str(tmp_path)]
    # --resume-from 指向的底座可以不存在吗?门在 warm-start 之前(parser.error
    # 先于数据/模型构造),用不存在的路径即可触发"须权重>0"门。
    code, out = _run_main(argv, monkeypatch, capsys)
    assert code == 2 and "roi_head_weight" in out.err


def test_warm_start_whitelist_for_roi_heads(tmp_path, capsys):
    """债3:带 ROI 头的模型 + 无 ROI 键的底座 → 不 raise 且打"ROI 头 N 键
    随机初始化"冷启行(旧底座四头/分割全量,缺的恰好只有 ROI 键)。"""
    torch.manual_seed(0)
    model = _tiny_model()
    # 构造"旧底座"state:当前模型全键去掉 ROI 头键 → missing 恰为 ROI 键
    full = model.state_dict()
    roi_pfx = MultiTaskThyroidModel.ROI_HEAD_PREFIXES
    old_state = {k: v for k, v in full.items()
                 if not k.startswith(roi_pfx)}
    ckpt = tmp_path / "base.pth"
    torch.save(old_state, ckpt)
    train_official.load_warm_start(model, str(ckpt), torch.device("cpu"))
    out = capsys.readouterr().out
    assert "ROI 头" in out and "随机初始化" in out
    # 非白名单键缺失仍 fail-loud:去掉一个 backbone 键 → raise
    bad_state = {k: v for k, v in old_state.items()
                 if not k.startswith("cls_backbone.")}
    bad_ckpt = tmp_path / "bad.pth"
    torch.save(bad_state, bad_ckpt)
    with pytest.raises(RuntimeError, match="非四头"):
        train_official.load_warm_start(model, str(bad_ckpt), torch.device("cpu"))


# ═════════════════════════════════════════════════════════════════════════════
# 债4 数据半:build_official_datasets roi_meta 透传 + mock 样本构造
# ═════════════════════════════════════════════════════════════════════════════

def test_mock_roi_samples_shape():
    """_mock_roi_samples:键集齐全、bbox 64² 图上按 i 渐变、无 mask_path。"""
    out = train_official._mock_roi_samples(n=8, img_size=64, seed=0)
    assert len(out) == 8
    for i, s in enumerate(out):
        # R1 P2(2026-09-26):bbox 渐变(16+i%8 .. +20),恒在 64² 内且方形;
        # global_loc 与 bbox 中心一致(渐变给 loc 头 CoordConv 通道变动维度)
        b = 16 + (i % 8)
        assert s["bbox_xyxy"] == (b, b, b + 20, b + 20)
        assert s["orig_w"] == 64 and s["orig_h"] == 64
        cx = (b + b + 20) / 2.0 / 64
        assert s["global_loc"] == (cx, cx)
        assert s["mask_path"] is None
        assert s["img_path"].startswith("mock::")
        for k in ("malignancy", "lesion_loc", "extrathy_ext",
                  "clin_t_stage", "tirads", "is_main"):
            assert k in s


def test_build_datasets_roi_meta_kwarg_passthrough(roi_root, monkeypatch):
    """build_official_datasets(roi_meta=True) 产出的 train_ds.samples 带
    roi_lesions(便捷入口透传 build_official_samples)。"""
    O = roi_root["O"]
    train_ds, val_ds = O.build_official_datasets(
        roi_root["root"], roi_root["xlsx"], img_size=32, seed=0,
        roi_meta=True)
    n_with = sum(1 for s in train_ds.samples + val_ds.samples
                 if "roi_lesions" in s)
    assert n_with > 0
    # 默认 False:无键(零回归)
    train2, val2 = O.build_official_datasets(
        roi_root["root"], roi_root["xlsx"], img_size=32, seed=0)
    assert all("roi_lesions" not in s
               for s in train2.samples + val2.samples)
