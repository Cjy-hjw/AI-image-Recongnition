"""scripts/data/format_thyroidxl.py 脚本逻辑单元测试。

现有 test_multisource_seg_smoke.py 是纯 loader 契约测试(loader.demo 接口),够不到
format_thyroidxl 的 _build_meta/convert/_self_check 这些脚本函数。本文件直接单测它们。

优先覆盖 5 个修复点(按优先级):
  1. 【阻塞1·最高】_build_meta 的 label_mask.malignancy 是**有效标志**非标签取值:
     良性样本(has_conclusion=True)必须 malignancy=1(有效),标签取值才存 label_malignancy。
     这是本轮最严重真 bug("曾写 malignancy:int(malignancy)" 致良性样本被误当无监督)的唯一直接锁点。
  2. 【高2】全黑 mask 判据:has_mask = mask 非 None **且** 有前景像素;全黑 mask 不写 mask.png。
  3. 【footgun】dest 指向活数据根(data / data_root)必须 raise ValueError 且信息含"受保护"。
  4. 【撞名】image_id 撞名(同 id 不同 patient)必须 raise ValueError 含"撞名"。
  5. 【_self_check】合法目录通过;违反契约(缺键)raise AssertionError。

纯合成:monkeypatch ThyroidXLAdapter 为一个假类(不读真实 ThyroidXL / 不连 A800 / 不用 GPU),
CPU 秒级。真实 ThyroidXLAdapter.get_image/get_mask 的"真实文件加载"路径由 A800 集成测覆盖,
本文件不依赖之;convert 的写盘/自检逻辑被完整走到。
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import scripts.data.format_thyroidxl as fmt


# ═════════════════════════════════════════════════════════════════════════════
# 假适配器工厂(monkeypatch 掉真实 ThyroidXLAdapter,让 convert 不依赖真实数据跑通)
# ═════════════════════════════════════════════════════════════════════════════

def _make_fake_adapter(samples, get_mask=None, get_image=None):
    """造一个冒充 ThyroidXLAdapter 的假类。

    samples: [{image_id, patient_id, malignancy, has_conclusion, ...}]
    get_mask: 可选,返回该样本的 mask 数组或 None(决定 has_mask 判据)。
    get_image: 可选,返回该样本图像数组(默认 32x32 灰)。
    """
    class _FakeAdapter:
        def __init__(self, _data_root):  # 匹配真实 ThyroidXLAdapter(data_root) 签名,合成测试不读盘
            self.samples = samples

        def get_labels(self, sm):
            return {"malignancy": sm.get("malignancy", 0)}

        def get_task_mask(self, sm):
            return {"malignancy": 1 if sm.get("has_conclusion") else 0,
                    "mask": 1 if sm.get("has_mask_path") else 0}

        def get_image(self, sm):
            if get_image is not None:
                return get_image(sm)
            return np.full((32, 32), 128, dtype=np.uint8)

        def get_mask(self, sm):
            if get_mask is not None:
                return get_mask(sm)
            return None
    return _FakeAdapter


def _multi_patient_samples(n=6):
    """造 n 个跨患者样本(确保 train/val 两侧都不为空,否则 _self_check 报 val 目录不存在)。"""
    return [
        {"image_id": f"id_{i}", "patient_id": f"p{i}",
         "malignancy": i % 2, "has_conclusion": True, "has_mask_path": True}
        for i in range(n)
    ]


def _run_convert(monkeypatch, tmp_path, samples, get_mask=None, get_image=None,
                 dest=None, data_root=None):
    """monkeypatch 假适配器 + 跑 convert,返回 counts。"""
    monkeypatch.setattr(fmt, "ThyroidXLAdapter",
                        _make_fake_adapter(samples, get_mask=get_mask, get_image=get_image))
    if dest is None:
        dest = str(tmp_path / "dest")
    if data_root is None:
        data_root = str(tmp_path / "fake_root")
    return fmt.convert(data_root=data_root, dest=dest, val_ratio=0.15, seed=42)


# ═════════════════════════════════════════════════════════════════════════════
# 1.【阻塞1·最高】_build_meta 的 label_mask.malignancy 是有效标志,非标签取值
# ═════════════════════════════════════════════════════════════════════════════

def test_build_meta_benign_valid_malignancy():
    """【阻塞1】良性样本且有 Conclusion → label_mask.malignancy 必须 =1(有效标志)。

    曾写 `malignancy:int(malignancy)` 把良性=0 塞进 label_mask,loader 读成 valid_malignancy=0,
    导致良恶性头只见恶性样本、良性零监督——本轮最严重真 bug。这里直接锁死:
    has_conclusion=True 时 malignancy 有效标志必须为 1,即使标签取值是 0。
    """
    meta = fmt._build_meta(image_id="x", patient_id="p", malignancy=0,
                           has_mask=True, has_conclusion=True)
    # 有效标志 =1(良性但有结论 = 有效!),与标签取值 0 是两个独立字段。
    assert meta["label_mask"]["malignancy"] == 1, \
        "良性+有结论样本 label_mask.malignancy 必须=1(有效标志),不能被标签值 0 污染"
    # 标签取值存在 label_malignancy(=0/1 取值),证明"取值"与"有效标志"独立。
    assert meta["label_malignancy"] == 0, "标签取值应存 0(良性),放在 label_malignancy"


def test_build_meta_no_conclusion_invalid_malignancy():
    """【阻塞1】对照:has_conclusion=False → label_mask.malignancy 必须 =0(无良恶性标注)。"""
    meta = fmt._build_meta(image_id="x", patient_id="p", malignancy=0,
                           has_mask=True, has_conclusion=False)
    assert meta["label_mask"]["malignancy"] == 0, "无 Conclusion 时 malignancy 有效标志应为 0"


def test_build_meta_label_mask_exact_7_keys():
    """【阻塞1】label_mask 恰好 7 键(mask+malignancy+5概念),缺任一键 loader 落默认1成坑。"""
    meta = fmt._build_meta(image_id="x", patient_id="p", malignancy=1,
                           has_mask=True, has_conclusion=True)
    assert set(meta["label_mask"].keys()) == set(fmt._ALL_LABEL_MASK_KEYS)
    assert len(meta["label_mask"]) == 7, f"label_mask 应恰好 7 键,实得 {len(meta['label_mask'])}"
    # mask 键跟 has_mask 走(阻塞1 的 7 键之一)。
    assert meta["label_mask"]["mask"] == 1


# ═════════════════════════════════════════════════════════════════════════════
# 2.【高2】全黑 mask 判据:has_mask = 非 None **且** 有前景像素
# ═════════════════════════════════════════════════════════════════════════════

def test_all_black_mask_not_written(monkeypatch, tmp_path):
    """【高2】get_mask 返回全零数组(文件存在但全黑)→ convert 判 has_mask=False:
    不写 mask.png,且 meta label_mask.mask=0。全黑 mask 是真监督漏洞,必须双重防线锁死。"""
    def _black_mask(sm):
        return np.zeros((32, 32), dtype=np.float32)  # 全黑,无前景像素
    counts = _run_convert(monkeypatch, tmp_path, _multi_patient_samples(6), get_mask=_black_mask)
    for split in ("train", "val"):
        split_dir = os.path.join(tmp_path, "dest", split)
        for sdir in os.listdir(split_dir):
            sd = os.path.join(split_dir, sdir)
            if not os.path.isdir(sd):
                continue
            assert not os.path.exists(os.path.join(sd, "mask.png")), \
                f"{sdir}: 全黑 mask 不应写 mask.png"
            with open(os.path.join(sd, "meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
            assert meta["label_mask"]["mask"] == 0, f"{sdir}: 全黑 mask 的 label_mask.mask 应为 0"
    # 计数口径:全部判为无 mask。
    assert counts["train"]["no_mask"] + counts["val"]["no_mask"] == 6


def test_all_black_mask_build_meta_no_mask():
    """【高2】has_mask=False 时 _build_meta 的 label_mask.mask 必须=0(与 convert 判据一致)。"""
    meta = fmt._build_meta(image_id="x", patient_id="p", malignancy=0,
                           has_mask=False, has_conclusion=True)
    assert meta["label_mask"]["mask"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# 3.【footgun】dest 指向活数据根必须拒,防不可逆删训练集
# ═════════════════════════════════════════════════════════════════════════════

def test_footgun_dest_equals_data_root(monkeypatch, tmp_path):
    """【footgun】dest == data_root(受保护集合含 data_root)→ 必须 raise ValueError 含"受保护"。

    裸 rmtree 若误传 --dest data_root 会不可逆删数据;footgun 检查在 convert 开头、
    adapter 构造之后,故仍需 monkeypatch 假适配器让构造不崩。
    """
    r = str(tmp_path / "data")
    monkeypatch.setattr(fmt, "ThyroidXLAdapter",
                        _make_fake_adapter(_multi_patient_samples(6)))
    with pytest.raises(ValueError) as exc:
        fmt.convert(data_root=r, dest=r)  # dest 指向 data_root → 受保护
    assert "受保护" in str(exc.value)


# ═════════════════════════════════════════════════════════════════════════════
# 4.【撞名】image_id 撞名(同 id 不同 patient)必须 fail-fast raise
# ═════════════════════════════════════════════════════════════════════════════

def test_image_id_collision_raises(monkeypatch, tmp_path):
    """【撞名】两个同 image_id 不同 patient 的样本 → convert 必须 raise ValueError 含"撞名"。

    撞名检测在写盘前 fail-fast(零残留),train/test 两侧同 stem 会写进同一 sample_dir 覆盖前者。
    """
    samples = [
        {"image_id": "dup", "patient_id": "p0", "malignancy": 0,
         "has_conclusion": True, "has_mask_path": True},
        {"image_id": "dup", "patient_id": "p1", "malignancy": 1,
         "has_conclusion": True, "has_mask_path": True},
        {"image_id": "ok", "patient_id": "p2", "malignancy": 0,
         "has_conclusion": True, "has_mask_path": True},
    ]
    monkeypatch.setattr(fmt, "ThyroidXLAdapter", _make_fake_adapter(samples))
    with pytest.raises(ValueError) as exc:
        fmt.convert(data_root=str(tmp_path / "fake_root"), dest=str(tmp_path / "dest"))
    assert "撞名" in str(exc.value)


# ═════════════════════════════════════════════════════════════════════════════
# 5.【_self_check】合法目录通过;违反契约(缺键)raise AssertionError
# ═════════════════════════════════════════════════════════════════════════════

def _write_case(split_dir, name, meta, with_mask_png=False):
    """写一个 sample_xxx/ 目录(meta 由调用方给)。"""
    sd = os.path.join(split_dir, name)
    os.makedirs(sd, exist_ok=True)
    with open(os.path.join(sd, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    if with_mask_png:
        import cv2
        cv2.imwrite(os.path.join(sd, "mask.png"), np.full((8, 8), 255, np.uint8))


def _valid_meta(image_id="a", mask_valid=0):
    """构造满足 _self_check 契约 1/2 的合法 meta(mask:0 无 mask.png)。"""
    return {
        "image_id": image_id, "patient_id": "p",
        "label_malignancy": 0,
        "label_mask": {"mask": mask_valid, "malignancy": 1,
                       "composition": 0, "echogenicity": 0, "shape": 0,
                       "margin": 0, "foci": 0},
    }


def test_self_check_valid_passes(tmp_path):
    """【_self_check】合法目录(合法 meta,无泄漏,计数对齐)→ 应通过不抛异常。"""
    dest = str(tmp_path / "dest")
    _write_case(os.path.join(dest, "train"), "sample_a", _valid_meta("a", mask_valid=0))
    _write_case(os.path.join(dest, "val"), "sample_b", _valid_meta("b", mask_valid=0))
    fmt._self_check(dest, patient_leak=set(), expected_total=2)  # 不抛异常即通过


def test_self_check_missing_label_mask_key_raises(tmp_path):
    """【_self_check】label_mask 缺键(如缺 malignancy)→ 必须 raise AssertionError。"""
    dest = str(tmp_path / "dest")
    bad = _valid_meta("a")
    del bad["label_mask"]["malignancy"]  # 违反契约1:7 键必须写全
    _write_case(os.path.join(dest, "train"), "sample_a", bad)
    _write_case(os.path.join(dest, "val"), "sample_b", _valid_meta("b"))
    with pytest.raises(AssertionError) as exc:
        fmt._self_check(dest, patient_leak=set(), expected_total=2)
    assert "缺键" in str(exc.value) or "label_mask" in str(exc.value)