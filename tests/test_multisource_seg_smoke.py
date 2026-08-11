"""TN3K + ThyroidXL 分割混训数据契约冒烟测试。

锁死 3 个高危契约(sonnet 对抗审定位):
1. 无 mask 样本 valid_mask=0 —— ThyroidXL 38% 样本无分割 mask,若 valid_mask 误为 1,
   全零 mask 会被当有效分割监督训练,静默拖垮主域。
2. 源级 label_mask 屏蔽概念 —— TN3K 概念是伪标签须屏蔽(源级 label_mask 全 0);
   ThyroidXL 无概念标注也须屏蔽。验证 loader 优先级链
   `meta.label_mask > source_label_mask > 默认1` 真的把 valid_* 压成 0。
3. 有 mask 样本 valid_mask=1 且分割有梯度 —— 正常样本不能被误屏蔽。

纯合成数据:tmp_path 造 2 个 mini 源目录,照 format_thyroidxl 的 meta 格式写 label_mask。
不读真实 ThyroidXL / 不连 A800 / 不用 GPU,CPU 秒级跑完。用真实 loader,不 mock。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch

from src.dataset.loader import ThyroidDataset, MultiSourceThyroidDataset
from src.models.losses.components.segmentation import compute_segmentation_loss
from src.models.losses.segmentation import DiceLoss


# ═════════════════════════════════════════════════════════════════════════════
# 合成数据工厂(照 format_thyroidxl._build_meta 的 label_mask 格式)
# ═════════════════════════════════════════════════════════════════════════════

def _make_sample(root, name, with_mask=True, meta=None):
    """造单个 sample_xxx/ 目录:必有 image.png+meta.json,可选 mask.png。

    image: 随机 64x64 灰度;mask: 二值随机图(含前景,保证 >127 二值化后非全零)。
    meta 由调用方给;给 None 时给默认心智 meta,便于单样本行为测试。
    """
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    cv2.imwrite(os.path.join(d, "image.png"), img)
    if with_mask:
        binarized = (img > 128).astype(np.uint8) * 255
        # 兜底:若随机图恰全 False,手工画一块前景,保证 mask 非空(区分于"无 mask")
        if binarized.sum() == 0:
            binarized[16:48, 16:48] = 255
        cv2.imwrite(os.path.join(d, "mask.png"), binarized)
    if meta is None:
        meta = {"patient_id": name, "label_mask": {"mask": 1 if with_mask else 0}}
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    return d


def _thyroidxl_meta(image_id, patient_id, with_mask=True):
    """照 format_thyroidxl._build_meta:label_mask 6 键写全(5 概念恒 0 + mask/malignancy)。

    与真实产品脚本结构一致——这正是本 smoke 要锁的格式契约。
    """
    return {
        "image_id": image_id,
        "patient_id": patient_id,
        "label_malignancy": 1,
        "label_composition": 0,
        "label_echogenicity": 0,
        "label_shape": 0,
        "label_margin": 0,
        "label_foci": [0, 0, 0, 0],
        "label_authenticity": 1,
        "is_duplicate": 0,
        "is_tampered": 0,
        "label_mask": {
            "mask": 1 if with_mask else 0,
            "malignancy": 1,
            "composition": 0,
            "echogenicity": 0,
            "shape": 0,
            "margin": 0,
            "foci": 0,
        },
    }


def _make_tn3k_source(root, n=3):
    """TN3K 源:目录含真实 mask + 源级屏蔽概念(伪标签不当下游监督)。"""
    os.makedirs(root, exist_ok=True)
    for i in range(n):
        _make_sample(root, f"sample_{i:03d}", with_mask=True,
                     meta={"patient_id": f"p{i}", "label_composition": i % 4})
    return root


def _make_thyroidxl_source(root, n=4, with_mask_ids=(0, 1)):
    """ThyroidXL 源:部分样本无 mask(有 mask→写 mask.png+label_mask.mask=1;
    无 mask→不写 mask.png+label_mask.mask=0),meta 全 6 键。

    with_mask_ids 默认前 2 个有 mask、后 2 个无,模拟 38% 无 mask 的形态。
    """
    os.makedirs(root, exist_ok=True)
    for i in range(n):
        has_mask = i in with_mask_ids
        _make_sample(root, f"sample_{i:03d}", with_mask=has_mask,
                     meta=_thyroidxl_meta(i, f"tlx_p{i}", with_mask=has_mask))
    return root


# ═════════════════════════════════════════════════════════════════════════════
# 契约1:无 mask 样本 valid_mask=0(ThyroidXL 38% 无 mask · 静默拖垮主域的高危点)
# ═════════════════════════════════════════════════════════════════════════════

class TestNoMaskValidMaskZero:
    """直接试 ThyroidDataset 单源路径,照 loader.py:96 的 mask 优先级链。"""

    def test_nomask_meta_mask0_valid_mask_is_zero(self, tmp_path):
        """无 mask.png 且 meta label_mask.mask=0 → loader 的 labels['valid_mask']==0。

        这是 ThyroidXL 无 mask 样本的落盘形态(format_thyroidxl 不写 mask.png +
        meta.mask=0 双重防线)。若 valid_mask 误为 1,全零 GT 会带队分割损失。
        """
        meta = _thyroidxl_meta("img1", "p1", with_mask=False)  # label_mask.mask=0
        _make_sample(str(tmp_path / "src"), "sample_001", with_mask=False, meta=meta)
        ds = ThyroidDataset(str(tmp_path / "src"), img_size=64)
        _, mask_t, labels = ds[0]
        assert labels["valid_mask"] == 0
        # 无 mask 样本 mask 张量为全零(loader.py:63-64 兜底)——一旦 valid_mask=1 会污染
        assert mask_t.sum().item() == 0.0

    def test_nomask_default_path_defaults_to_valid(self, tmp_path):
        """反向红线对照:缺省(无 meta mask/无 source)仍 valid_mask=1(旧行为)。

        用以证明上面 valid_mask=0 是「显示写 mask:0」的结果,而非 loader 缺省反了。
        """
        # 无 mask.png + meta 不含 label_mask → 落默认 1(loader.py:96)
        _make_sample(str(tmp_path / "src"), "sample_001", with_mask=False,
                     meta={"patient_id": "p1"})
        ds = ThyroidDataset(str(tmp_path / "src"), img_size=64)
        _, _, labels = ds[0]
        assert labels["valid_mask"] == 1, "缺省路径应保持 valid_mask=1(向后兼容)"


# ═════════════════════════════════════════════════════════════════════════════
# 契约3:有 mask 样本 valid_mask=1(正常样本不被误屏蔽)
# ═════════════════════════════════════════════════════════════════════════════

class TestHasMaskValidMaskOne:
    def test_with_mask_meta_mask1_valid_mask_is_one(self, tmp_path):
        """有 mask.png 且 meta label_mask.mask=1 → valid_mask==1。

        正常甲状腺样本不能被「无 mask 屏蔽」逻辑误伤——这是混训的对照组。
        """
        meta = _thyroidxl_meta("img2", "p2", with_mask=True)  # label_mask.mask=1
        _make_sample(str(tmp_path / "src"), "sample_001", with_mask=True, meta=meta)
        ds = ThyroidDataset(str(tmp_path / "src"), img_size=64)
        _, mask_t, labels = ds[0]
        assert labels["valid_mask"] == 1
        assert mask_t.sum().item() > 0.0, "有 mask 样本 mask 张量应非全零"


# ═════════════════════════════════════════════════════════════════════════════
# 契约2a:TN3K 源级 label_mask 屏蔽概念(valid_composition 等压 0)
# ═════════════════════════════════════════════════════════════════════════════

class TestSourceLevelMaskConcepts:
    def test_tn3k_source_masks_concepts(self, tmp_path):
        """TN3K 源:source_label_mask 屏蔽概念 → valid_composition/echogenicity/...==0。

        TN3K 概念是伪标签(非真金),混训时须整体屏蔽其梯度。规则:优先级链
        `meta.label_mask > source_label_mask > 默认1` 中,未逐样本写 meta 的样本
        落到 source_label_mask 的 0。
        """
        path = _make_tn3k_source(str(tmp_path / "tn3k"), n=3)
        # TN3K 源级:屏蔽全部 6 项概念(含 mask 保留为 1——TN3K 有真 mask)
        source_label_mask = {
            "composition": 0, "echogenicity": 0, "shape": 0,
            "margin": 0, "foci": 0, "malignancy": 0,
        }
        ds = ThyroidDataset(path, img_size=64, source_label_mask=source_label_mask)
        for i in range(len(ds)):
            _, _, labels = ds[i]
            for t in ("composition", "echogenicity", "shape", "margin", "foci", "malignancy"):
                assert labels[f"valid_{t}"] == 0, f"样本{i} 的 valid_{t} 应被源级屏蔽"
            assert labels["valid_mask"] == 1, "TN3K 有真 mask,valid_mask 应保留 1"


# ═════════════════════════════════════════════════════════════════════════════
# 契约2b:ThyroidXL 源逐样本 meta 写全 6 键(label_mask.mask + 5 概念全 0)
# → valid_composition 等概念被屏蔽,且与源级是否声明无关
# ═════════════════════════════════════════════════════════════════════════════

class TestThyroidXLSampleMetaMaskConcepts:
    def test_thyroidxl_meta_full_keys_concepts_zeroed(self, tmp_path):
        """ThyroidXL 逐样本 meta label_mask 写全 6 键(5 概念=mask=按真值)。

        → valid_composition == 0。因 5 概念占位恒 0,源级 label_mask 即便不声明
        (缺省 1),meta 的 label_mask.mask/foci 等键优先把 valid 压成 0。
        """
        path = _make_thyroidxl_source(str(tmp_path / "tlx"), n=4)
        # 模拟真实接入:任务侧只把 ThyroidXL 源丢给 MultiSource,不带源级概念屏蔽
        ds = ThyroidDataset(path, img_size=64)
        for i in range(len(ds)):
            _, _, labels = ds[i]
            for t in ("composition", "echogenicity", "shape", "margin", "foci"):
                assert labels[f"valid_{t}"] == 0, (
                    f"样本{i} valid_{t} 应=0(meta 6 键全写且概念占位 0)")
        # 掩膜与 malignancy 按真值(默认 with_mask_ids=(0,1):idx0/idx1 有 mask)
        _, _, labels0 = ds[0]
        assert labels0["valid_mask"] == 1
        assert labels0["valid_malignancy"] == 1

    def test_thyroidxl_nomask_and_withmask_mix(self, tmp_path):
        """同一 ThyroidXL 源内混搭有/无 mask → valid_mask 逐样本精确。

        前2个有 mask=1,后2个无 mask=0。这是 38% 无 mask 的真实形态,int 级精确校验。
        """
        path = _make_thyroidxl_source(str(tmp_path / "tlx_mix"), n=4, with_mask_ids=(0, 1))
        ds = ThyroidDataset(path, img_size=64)
        v = [ds[i][2]["valid_mask"] for i in range(len(ds))]
        assert v == [1, 1, 0, 0], f"valid_mask 逐样本应为 [1,1,0,0],实得 {v}"
        # 无 mask 样本 mask 张量全零(防假监督前置校验)
        for i in (2, 3):
            assert ds[i][1].sum().item() == 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 多源装配:TN3K + ThyroidXL 混训 + get_sample_weights
# ═════════════════════════════════════════════════════════════════════════════

class TestMultiSourceAssembly:
    def test_mixed_assembly_validity(self, tmp_path):
        """TN3K + ThyroidXL 两源装配成 MultiSourceThyroidDataset,逐源 valid_* 正确。

        - TN3K 源:概念全屏蔽(源级),mask/真分割 保留 1。
        - ThyroidXL 源:概念屏蔽(meta),mask 逐样本按真值。
        这是混训数据契约的端到端拼装验证。
        """
        tn3k = _make_tn3k_source(str(tmp_path / "tn3k"), n=3)
        tlx = _make_thyroidxl_source(str(tmp_path / "tlx"), n=4, with_mask_ids=(0, 1))
        tn3k_source_mask = {
            "composition": 0, "echogenicity": 0, "shape": 0,
            "margin": 0, "foci": 0, "malignancy": 0,
        }
        mds = MultiSourceThyroidDataset([
            {"name": "tn3k", "path": tn3k, "label_mask": tn3k_source_mask, "ratio": 0.7},
            {"name": "tlx", "path": tlx, "ratio": 0.3},
        ], img_size=64)
        assert len(mds) == 7

        # TN3K 块(idx 0..2)
        for i in range(3):
            _, _, la = mds[i]
            assert la["valid_mask"] == 1
            assert la["valid_composition"] == 0
        # ThyroidXL 块(idx 3..6)
        v = [mds[i][2]["valid_mask"] for i in range(3, 7)]
        assert v == [1, 1, 0, 0], f"ThyroidXL 块 valid_mask 应为 [1,1,0,0],实得 {v}"
        for i in range(3, 7):
            assert mds[i][2]["valid_composition"] == 0

    def test_sample_weights_by_ratio(self, tmp_path):
        """get_sample_weights 长度==总样本数,且按源 ratio 均摊。

        source b 的样本权重 0.3 -> 各 source 的总权重 = ratio(供 WeightedRandomSampler)。
        """
        a = _make_tn3k_source(str(tmp_path / "a"), n=3)
        b = _make_thyroidxl_source(str(tmp_path / "b"), n=4)
        mds = MultiSourceThyroidDataset([
            {"name": "a", "path": a, "ratio": 0.7},
            {"name": "b", "path": b, "ratio": 0.3},
        ], img_size=64)
        w = mds.get_sample_weights()
        assert len(w) == 7, f"权重长度应=总样本数7,实得 {len(w)}"
        assert abs(sum(w[:3]) - 0.7) < 1e-6   # 源 a 总权重=1.0-哨兵 0.7
        assert abs(sum(w[3:]) - 0.3) < 1e-6   # 源 b 总权重=0.3
        assert abs(w[0] - 0.7 / 3) < 1e-6     # 逐样本=ratio/n


# ═════════════════════════════════════════════════════════════════════════════
# 契约红线:valid_mask=0 的样本在分割损失里零梯度(推理侧/交付侧硬约束)
# ═════════════════════════════════════════════════════════════════════════════

class TestSegZeroGradientWhenInvalid:
    """compute_segmentation_loss 好构造(只需 bce_fn/dice_fn/输出),故直接验零梯度。

    这一层锁的是「loader valid_mask=0 → 分割损失不吐梯度」的端到端链路:
    loader 给全零 GT + valid_mask=0,loss 必须对 seg logits 零梯度,否则无 mask 样本
    仍会通过 BCE/Dice 拖住分割头。
    """

    H = 32

    def test_valid_mask0_zero_grad(self):
        """batch 全 valid_mask=0 → 分割 logits 零梯度(loss 近似 0)。

        腐蚀场景:ThyroidXL 无 mask 样本拼成 batch 时不泄漏监督。用_eps 分母,值非严格0
        但梯度必须为 0。
        """
        torch.manual_seed(0)
        bce_fn = torch.nn.BCEWithLogitsLoss()
        dice_fn = DiceLoss()
        pred = torch.randn(4, 1, self.H, self.H, requires_grad=True)
        gt = (torch.rand(4, 1, self.H, self.H) > 0.5).float()
        loss = compute_segmentation_loss(bce_fn, dice_fn,
                                         {"mask": pred}, {"mask": gt,
                                                          "valid_mask": torch.zeros(4)})
        loss.backward()
        if pred.grad is not None:
            assert pred.grad.abs().sum().item() < 1e-6, (
                f"全 invalid 时 seg 梯度应≈0,实得 {pred.grad.abs().sum().item()}")
        else:
            # loss 未参与 pred → backward 不碰 pred,grad 为 None 同样合格(零梯度)
            pass

    def test_valid_mask1_preserves_grad(self):
        """对照组:valid_mask=1 的样本 must 保留分割梯度(不被误屏蔽)。"""
        torch.manual_seed(0)
        bce_fn = torch.nn.BCEWithLogitsLoss()
        dice_fn = DiceLoss()
        pred = torch.randn(4, 1, self.H, self.H, requires_grad=True)
        gt = (torch.rand(4, 1, self.H, self.H) > 0.5).float()
        loss = compute_segmentation_loss(bce_fn, dice_fn,
                                         {"mask": pred}, {"mask": gt,
                                                          "valid_mask": torch.ones(4)})
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum().item() > 1e-6, "valid_mask=1 必须保留分割梯度"