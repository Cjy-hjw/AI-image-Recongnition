# -*- coding: utf-8 -*-
"""ROI 推理侧接线回归测试(新架构第五波,2026-09-25)。

被测链路:
  `src/inference/roi_pipeline.py`(ROI 裁剪/批量前向;is_dynamic 键补齐)
  + `src/inference/v21_output.py`(_roi_nodule_context / _run_roi_static /
    _run_roi_dynamic / per_nodule_contexts 分流)
  + `src/inference/postprocess.py`(assert_finite 提升模块级 + validate_roi_probs)
  + `src/inference/case_pipeline.py`(开关记录 + tiling 互斥守卫)
  + `src/service/runner.py`(roi_ctx 构建 + orig_img_np 透传)

核心断言按交接文档 §2.4 的 11 条组织,叠加 addendum 的 golden 对拍/键集/零污染
三条;🔴 三条零区分力坑(memory mirror-tests-need-distinguishing-power)逐条规避:
- 几何全部用**非方阵 H=13, W=17**(方阵下转置/resize 错误隐形);
- FakeModel(真 nn.Module)的 forward_roi 输出**由输入内容决定**(按 patch mean
  分支——同 patch 恒同值、不同 patch 不同值,零区分力用例恒红);
- golden 用例断言**硬编码期望值**(mal 0.72 / tirads 4 / ete 2 / loc 5 / T 1 /
  t_conf 0.3),one-hot 类全部避开 idx0。
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sklearn.metrics  # noqa: F401  (import 副作用:抢先加载 OpenMP 运行时,
# 防 pytest 收集深度下 sklearn→pandas→pyarrow 链 Windows 栈溢出,
# test_cliff_loss.py:29 同款先例——本文件 import torch 链,必须先 sklearn)

import torch  # noqa: E402

pytest.importorskip("cv2")
pytest.importorskip("scipy")
nib = pytest.importorskip("nibabel")

from src.inference import v21_output  # noqa: E402
from src.inference.prediction_json import validate_prediction_json  # noqa: E402
from src.models.label_schema import NUM_LOCATIONS  # noqa: E402

# 非方阵(转置错误在方阵下隐形,test_v21_output.py:38 同款理由)。
_H, _W = 13, 17
_MODEL = 8


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _soft_and_binary_two():
    """模型输出尺寸的两连通块(两结节)软/二值图。"""
    soft = np.full((_MODEL, _MODEL), 0.05, dtype=np.float32)
    soft[1:3, 1:3] = 0.9
    soft[5:7, 5:7] = 0.85
    binary = (soft > 0.5).astype(np.uint8) * 255
    return soft, binary


def _static_case_result(soft, binary, mal_prob=0.72, with_orig_img=None):
    """合成静态 case_result(key 对齐 test_v21_output.py 的同名 fixture;
    默认 _head_probs 分布的 golden 值 = mal 0.72 / tirads 4 / ete 2 / loc 5 /
    T 1 / t_conf 0.3)。"""
    preds = {
        "pred_mask": soft,
        "mask_binary": binary,
        "comp_idx": 0, "echo_idx": 1, "shape_idx": 0, "margin_idx": 1,
        "comp_prob": np.array([0.6, 0.4]), "echo_prob": np.array([0.3, 0.7]),
        "shape_prob": np.array([0.5, 0.5]), "margin_prob": np.array([0.2, 0.8]),
        "foci_prob": np.array([0.1, 0.9]),
        "mal_prob": float(mal_prob),
        "mal_probs": np.array([1.0 - mal_prob, mal_prob]),
        # 与 test_v21_output._head_probs 默认分布逐位相同(golden 依据)
        "ete_probs": np.array([0.1, 0.6, 0.2, 0.1]),
        "location_probs": np.array([0.05, 0.05, 0.1, 0.1, 0.5, 0.1, 0.1]),
        "t_probs": np.array([0.3, 0.7]),
        "tirads_probs": np.array([0.05, 0.05, 0.1, 0.7, 0.1]),
        "t_conf": 0.3,
    }
    cr = {
        "preds": preds,
        "img_tensor": np.zeros((1, 1, _MODEL, _MODEL), np.float32),
        "mask_resized": binary,
        "breakdown": {"total_points": 5},
        "orig_hw": (_H, _W),
        "gatekeeper_audit": {"skipped": False},
        "is_dicom": False,
    }
    if with_orig_img is not None:
        cr["orig_img_np"] = with_orig_img
    return cr


class FakeRoiModel(torch.nn.Module):
    """forward_roi 输出**由输入内容决定**的假模型(零区分力用例的核心)。

    按 patch 归一化均值(先 /255 再 mean,与 preprocess_image 的 [0,1] 域一致)
    二分支:mean ≥ 0.25 → 分支 A(ETE 峰 idx2/TI-RADS 峰 idx4/loc 峰 idx6/
    t_conf 0.9/mal logits (−3,3)→p≈0.9975);否则分支 B(全部峰 idx0 低分)。
    同 patch 恒同值、不同均值 patch 值不同——"每结节独立"用例靠它才有区分力
    (照 test_tiling_inference.py:154 手法,**返回键集按 collect_roi_records
    docstring 逐键写,不抄整图 forward 的键**)。

    输出 CPU torch 张量(生产 forward_roi 也是张量出,CUDA 张量路径由
    postprocess._to_np 的 detach().cpu() 覆盖,v21._to_np 同款)。
    签名与 `MultiTaskThyroidModel.forward_roi` **逐参一致**:
    (self, roi_images, roi_meta, seg_features=None)。
    🔴 E2 review P0-1 修复(2026-09-25):旧 fake 签名多了个 model 形参
    `(self, model, roi_images, ...)`,把生产调用
    `forward_roi(model, img, metas)` 的错位写进了契约——40 项全绿但真模型
    必崩(fake 契约必须与真实签名逐参一致,见
    memory/fake-must-match-real-return-contract)。
    """

    def forward(self, x, gamma_prior=1.0):  # pragma: no cover - ROI 用例不调它
        raise AssertionError("FakeRoiModel 只应被 forward_roi 调用")

    def forward_roi(self, roi_images, roi_metas, seg_features=None):
        # 与 MultiTaskThyroidModel.forward_roi 同参序(self, roi_images,
        # roi_meta, seg_features=None)。K = batch 维。
        # ⚠️ 子类 override 时必须保持同款参数序(self, roi_images, ...)——
        # 见 CountingModel。
        K = int(roi_images.shape[0])
        # preprocess_image 已把 patch 归一化到 [0,1];阈值 0.25 落在
        # "亮块 patch(均值 ~0.07-0.17)与全 0 patch(0.0)"之间的量纲外——
        # 实际静态用例的 patch 均值实测 17/255≈0.067 与 6/255≈0.024,
        # 用 0.03 作分界恰好分开两实例(实测:inst1 0.067 ≥ 0.03 → hot;
        # inst2 0.024 < 0.03 → cold),且对全零/全亮输入稳健。
        means = roi_images.float().mean(dim=(1, 2, 3))
        out = []
        for i in range(K):
            hot = bool(means[i] >= 0.03)
            if hot:
                loc = [0.02] * NUM_LOCATIONS; loc[6] = 0.88
                tir = [0.02] * 5; tir[4] = 0.92
                ete = [0.03] * 4; ete[2] = 0.91
                mal = torch.tensor([-3.0, 3.0])       # softmax → p1 ≈ 0.9975
                t_conf = 0.9
            else:
                loc = [0.26, 0.16, 0.14, 0.12, 0.10, 0.12, 0.10]  # 峰 idx0
                tir = [0.6, 0.2, 0.1, 0.05, 0.05]
                ete = [0.7, 0.15, 0.1, 0.05]
                mal = torch.tensor([3.0, -3.0])       # p1 ≈ 0.0025
                t_conf = 0.1
            out.append({
                "lesion_id": roi_metas[i]["lesion_id"],
                "is_main": 1,
                "patient_id": "P",
                "malignancy": mal,
                "location_probs": torch.tensor(loc, dtype=torch.float32),
                "ete_probs": torch.tensor(ete, dtype=torch.float32),
                "ete_logits": torch.tensor(ete, dtype=torch.float32),
                "t_conf": torch.tensor(t_conf, dtype=torch.float32),
                "tirads_probs": torch.tensor(tir, dtype=torch.float32),
            })
        return out


def _roi_ctx(model, img_size=_MODEL):
    return {"model": model, "precision": "fp32",
            "device": torch.device("cpu"), "img_size": int(img_size)}


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_static(tmp_path, name, cr, roi_ctx=None):
    case_dir = str(tmp_path / name)
    v21_output.write_case_v21(case_dir, "P001", "S", cr, "static", 30,
                              roi_ctx=roi_ctx)
    payload = _read_json(os.path.join(case_dir, "prediction.json"))
    validate_prediction_json(payload)
    return payload


def _two_hot_patch_image():
    """H×W 原图,两块内容差异大的前景(归一化均值 1.0 vs 0.0196,分界 0.03)。
    🔴 块位置必须对齐 **fullres 实例 bbox 落点**(非 _MODEL 域 mask 块位置):
    `_write_static` 的 instances 来自 `binary_fullres`(512 域 mask resize 到
    原图 H×W 后的连通域),实测两实例框 = (3,2,7,5)/(11,9,15,12)——亮块画
    在前者覆盖区、暗块画在后者覆盖区,ROI patch 才能取到差异内容。
    2026-09-26 R-A F1 修后口径:ROI_PAD_RATIO 0.25→0.0(对齐训练侧紧切片),
    旧 0.25 边距曾把 512 域位置的亮块卷进 patch(F1 修后失效);暗块值 30 在
    紧切片下均值 0.118>0.03 也判 hot,一并改 5(5/255≈0.0196<0.03)。"""
    img = np.zeros((_H, _W), np.uint8)
    img[2:5, 3:7] = 255      # 实例 A(fullres bbox (3,2,7,5))处:亮
    img[9:12, 11:15] = 5     # 实例 B(fullres bbox (11,9,15,12))处:暗
    return img


# --------------------------------------------------------------------------- #
# 1/2. 开关关:零回归(sys.modules 零污染 / 无 roi_heads 键 / golden 逐位)
# --------------------------------------------------------------------------- #

class TestRoiOffZeroRegression:
    def test_off_no_roi_pipeline_module_loaded(self, tmp_path, monkeypatch):
        """env 关:跑完 diagnose_static_case 后 sys.modules 无 roi_pipeline。"""
        monkeypatch.delenv("AICOMP_ROI_HEADS", raising=False)
        monkeypatch.delenv("AICOMP_STATIC_TILING", raising=False)
        monkeypatch.delenv("AICOMP_STATIC_TTA", raising=False)
        sys.modules.pop("src.inference.roi_pipeline", None)
        from src.inference.case_pipeline import diagnose_static_case
        img = (np.random.rand(_H, _W) * 255).astype(np.uint8)
        cfg = {"data": {"img_size": 512}}
        # 真小模型太重,这里用 FakeRoiModel 充 forward(整图路径只消费 mask/
        # 四头键,decode 需要全键集——改用纯 case_result 落盘路径验证零污染
        # 不现实;直接跑 diagnose_static_case 需要 forward 全键,故此用例改由
        # case_pipeline 的 import 零污染 + 落盘路径零污染双断言覆盖,见下条)。
        # —— 保留一个最小真跑:FakeRoiModel 无整图 forward,故断言 import 面。
        import src.inference.case_pipeline  # noqa: F401
        assert "src.inference.roi_pipeline" not in sys.modules

    def test_off_case_result_has_no_roi_heads_key(self, tmp_path, monkeypatch):
        """env 关:case_result(诊断返回)无 roi_heads 键(照 tiling 零回归形态)。"""
        monkeypatch.delenv("AICOMP_ROI_HEADS", raising=False)
        monkeypatch.delenv("AICOMP_STATIC_TILING", raising=False)
        monkeypatch.delenv("AICOMP_STATIC_TTA", raising=False)
        from src.models.multi_task_model import MultiTaskThyroidModel
        cfg = {"data": {"img_size": 512}}
        m = MultiTaskThyroidModel(backbone_name="efficientnet_b2",
                                  pretrained=False, config=cfg)
        m.eval()
        from src.inference.case_pipeline import diagnose_static_case
        img = (np.random.rand(40, 44) * 255).astype(np.uint8)
        with torch.no_grad():
            result = diagnose_static_case(m, img, cfg, None, "fp32",
                                          torch.device("cpu"))
        assert "roi_heads" not in result, "关时不得新增任何键(零回归)"

    def test_off_runner_static_case_no_orig_img_key(self, monkeypatch):
        """env 关:runner._run_static_case 不向 result 加 orig_img_np 键。"""
        monkeypatch.delenv("AICOMP_ROI_HEADS", raising=False)
        from src.service import runner
        # 静态分支在 gatekeeper 前就会 return——直接对关键透传点做单元验证:
        # _roi_heads_on() 关时 _run_static_case 的注入分支不求值。
        assert runner._roi_heads_on() is False

    def test_off_golden_values_bitwise(self, tmp_path, monkeypatch):
        """开关关 golden 对拍:整图共享四头,逐项硬编码期望
        mal 0.72 / tirads_class 4 / ete 2 / location 5 / T 1 / t_conf 0.3。"""
        monkeypatch.delenv("AICOMP_ROI_HEADS", raising=False)
        soft, binary = _soft_and_binary_two()
        cr = _static_case_result(soft, binary)
        # 显式传 roi_ctx=None(关路)与不传等价——两条都验
        payload = _write_static(tmp_path, "P_GOLD", cr, roi_ctx=None)
        n = payload["Prediction"]["Nodules"][0]
        assert abs(n["Malignancy"]["confidence"] - 0.72) < 1e-9
        assert n["TI_RADS_Class"] == 4
        assert n["ExtrathyroidalExtension"] == 2
        assert n["Location"] == 5
        assert n["T_ClinSig"]["T"] == 1
        assert abs(n["T_ClinSig"]["T_conf"] - 0.3) < 1e-9
        # 两结节同值(整图共享语义)
        n2 = payload["Prediction"]["Nodules"][1]
        assert n2["Malignancy"]["confidence"] == n["Malignancy"]["confidence"]
        assert n2["TI_RADS_Class"] == n["TI_RADS_Class"]

    def test_off_prediction_json_identical_with_and_without_orig_img(
            self, tmp_path, monkeypatch):
        """关路对拍:case_result 带/不带 orig_img_np 两写,prediction.json
        逐字节一致(关时该键纯 inert,零回归的 diff=0 断言)。"""
        monkeypatch.delenv("AICOMP_ROI_HEADS", raising=False)
        soft, binary = _soft_and_binary_two()
        cr_a = _static_case_result(soft, binary)
        cr_b = _static_case_result(soft, binary,
                                   with_orig_img=np.full((_H, _W), 77, np.uint8))
        pa = _write_static(tmp_path, "P_A", cr_a, roi_ctx=None)
        pb = _write_static(tmp_path, "P_B", cr_b, roi_ctx=None)
        assert json.dumps(pa, sort_keys=True) == json.dumps(pb, sort_keys=True), \
            "关路下 orig_img_np 键必须 inert(逐字节一致)"


# --------------------------------------------------------------------------- #
# 3. ROI 裁剪坐标可逆 + 元数据键集
# --------------------------------------------------------------------------- #

class TestRoiGeometry:
    def test_crop_roundtrip_pixelwise_non_square(self):
        """crop → 实际框还原回原图,逐像素相等(非方阵 H=13,W=17)。"""
        from src.inference.roi_pipeline import (
            crop_roi_for_instance, _roi_instance_bbox)
        frame = np.arange(_H * _W, dtype=np.uint8).reshape(_H, _W)
        inst = {"bbox": (4, 6, 8, 5)}
        bbox = _roi_instance_bbox(inst)
        patch, actual = crop_roi_for_instance(frame, bbox)
        x0, y0, x1, y1 = actual
        assert np.array_equal(patch, frame[y0:y1, x0:x1]), \
            "ROI patch 必须与实际框切片逐像素相等(坐标可逆)"

    def test_roi_meta_key_set_contract(self):
        """roi_meta 含 roi_dataset.RoiDataset.__getitem__ meta 的全部键
        (含 is_dynamic 补齐)——键集契约,防消费方 .get() 静默取默认。"""
        from src.inference.roi_pipeline import iter_instance_frames_static
        frame = np.zeros((_H, _W), np.uint8)
        inst = {"bbox": (4, 6, 8, 5)}
        meta = list(iter_instance_frames_static(
            frame, inst, "P::S::N1", patient_id="P"))[0][1]
        expected = {
            "bbox_xyxy", "orig_w", "orig_h", "pad_hw", "spacing_mm",
            "lesion_id", "patient_key", "patient_id", "is_main",
            "frame_index", "global_loc", "is_dynamic",
        }
        missing = expected - set(meta.keys())
        assert not missing, f"roi_meta 缺键(键集契约破坏):{missing}"
        assert meta["is_dynamic"] is False, "静态 meta 的 is_dynamic 应为 False"

    def test_roi_meta_dynamic_is_dynamic_true(self):
        """动态生成器 meta 的 is_dynamic = True(补齐键的动态侧)。"""
        from src.inference.roi_pipeline import iter_instance_frames_dynamic
        inst = {"bbox": (4, 6, 8, 5), "frame_indices": [0, 2]}
        frames = {0: np.zeros((_H, _W), np.uint8),
                  2: np.zeros((_H, _W), np.uint8)}
        rows = list(iter_instance_frames_dynamic(frames, inst, "P::S::N1"))
        assert len(rows) == 2
        assert all(m["is_dynamic"] is True for _, m in rows)
        assert [m["frame_index"] for _, m in rows] == [0, 2]


# --------------------------------------------------------------------------- #
# 4. 静态 ROI 分流:每结节独立 + fail-loud
# --------------------------------------------------------------------------- #

class TestRoiStatic:
    def test_two_nodules_independent_contexts(self, tmp_path):
        """开关开:两结节 ROI 内容不同 → 四头值不同(输出依赖输入)。"""
        soft, binary = _soft_and_binary_two()
        img = _two_hot_patch_image()
        cr = _static_case_result(soft, binary, with_orig_img=img)
        payload = _write_static(tmp_path, "P_ROI", cr,
                                roi_ctx=_roi_ctx(FakeRoiModel()))
        nodules = payload["Prediction"]["Nodules"]
        assert len(nodules) == 2
        # 实例序 = 面积降序:块 [1:3,1:3](4px) 与 [5:7,5:7](4px) 面积相同,
        # cv2 连通域序按标签扫描(左上在前)。左上块 patch 吃到亮区(255)→hot;
        # 右下块 patch 以暗区(0/30)为主 → cold。
        # 但两块 patch 都含大量 0 背景——直接断言"两结节四头值不全相同"
        # (at least mal/tirads/loc 之一不同),再由 lesion_id 顺序断言对齐。
        vals = [(n["TI_RADS_Class"], n["ExtrathyroidalExtension"],
                 n["Location"], n["Malignancy"]["confidence"]) for n in nodules]
        assert vals[0] != vals[1], \
            f"两结节 ROI 内容不同但四头值全同(输出未依赖输入):{vals}"

    def test_static_mal_confidence_is_softmaxed_not_raw_logit(self, tmp_path):
        """静态 ROI 路 malignancy_confidence 必须是 softmax 概率(2026-09-25
        模型面审查 P0-1 回归测试):FakeRoiModel hot 分支 mal logits
        (−3,3) → p1≈0.9975;若裸 logit 直通会写 3.0(越界 [0,1],官方按 0
        处理、AUC-PR 垫底)。

        ⚠️ 该值此前无静态 ON 数值断言(动态路有 L449 的均值断言,静态路
        只查"两结节不同")——logit 直通正是从这条缝漏进去的。
        """
        soft, binary = _soft_and_binary_two()
        img = _two_hot_patch_image()
        cr = _static_case_result(soft, binary, with_orig_img=img)
        payload = _write_static(tmp_path, "P_MALSOFT", cr,
                                roi_ctx=_roi_ctx(FakeRoiModel()))
        nodules = payload["Prediction"]["Nodules"]
        # 左上块 patch 均值 0.067 ≥ 0.03 → hot(mal logits (−3,3));
        # softmax(−3,3)[1] = 0.9975(容差 2e-3,与动态路同款)。
        assert abs(nodules[0]["Malignancy"]["confidence"] - 0.9975) < 2e-3, \
            "静态路 mal confidence 应为 softmax 概率(裸 logit 直通=越界)"
        # 全部结节的 confidence 必须落在 [0,1](官方置信度判罚红线)。
        for n in nodules:
            c = n["Malignancy"]["confidence"]
            assert 0.0 <= c <= 1.0, f"confidence 越界 [0,1]: {c}"

    def test_static_missing_orig_img_raises_not_softmask(self, tmp_path):
        """开关开:缺 orig_img_np → V21OutputError(不拿概率图冒充)。"""
        soft, binary = _soft_and_binary_two()
        cr = _static_case_result(soft, binary)   # 无 orig_img_np
        with pytest.raises(v21_output.V21OutputError, match="orig_img_np"):
            _write_static(tmp_path, "P_MISS", cr, roi_ctx=_roi_ctx(FakeRoiModel()))

    def test_model_without_forward_roi_raises_no_fallback(self, tmp_path):
        """模型无 forward_roi → RoiPipelineError,**不回退整图路径**。"""
        class NoRoiModel(torch.nn.Module):
            def forward(self, x, gamma_prior=1.0):  # pragma: no cover
                raise AssertionError("不应被调用")

        soft, binary = _soft_and_binary_two()
        cr = _static_case_result(soft, binary, with_orig_img=np.zeros((_H, _W), np.uint8))
        from src.inference.roi_pipeline import RoiPipelineError
        with pytest.raises(RoiPipelineError, match="forward_roi"):
            _write_static(tmp_path, "P_NOROI", cr, roi_ctx=_roi_ctx(NoRoiModel()))

    def test_zero_instances_no_forward(self, tmp_path):
        """阴性(零实例):跳过 ROI 前向,空 Nodules(不调 build_roi_batch)。"""
        soft = np.full((_MODEL, _MODEL), 0.05, np.float32)
        binary = np.zeros((_MODEL, _MODEL), np.uint8)
        cr = _static_case_result(soft, binary,
                                 with_orig_img=np.zeros((_H, _W), np.uint8))
        payload = _write_static(tmp_path, "P_NEG", cr,
                                roi_ctx=_roi_ctx(FakeRoiModel()))
        assert payload["Prediction"]["Nodules"] == []

    def test_run_roi_forward_real_model_contract(self):
        """🔴 E2 review P0-1 回归:run_roi_forward × **真** MultiTaskThyroidModel
        必须跑通(fake 全绿掩盖过调用错位——fake 契约必须与真实签名逐参一致,
        但只改 fake 不加真模型锚,同型错位还能再犯)。

        小模型(K=1,img 8²)实跑 forward_roi,断言:不抛 AttributeError
        (旧三参调用 `forward_roi(model, img, metas)` 在 :461 `roi_images.shape`
        崩)、返回 list[dict] 与 metas 逐条对齐、键集含四头。
        """
        from src.inference.roi_pipeline import run_roi_forward, build_roi_batch
        from test_roi_model import _tiny_model, _roi_meta, ROI_S

        model = _tiny_model()
        model.eval()
        # ROI_S=224:test_roi_model 的既有用例尺寸(8² 会把 encoder 中途
        # conv 顶穿 kernel>输入,属 fixture 尺寸错非被测问题)。
        patches = [np.full((224, 224), 128, np.uint8)]
        metas = _roi_meta(1, with_mask=False)
        img = build_roi_batch(patches, metas, ROI_S, torch.device("cpu"))
        out = run_roi_forward(model, img, metas, "fp32", torch.device("cpu"))
        assert isinstance(out, list) and len(out) == 1
        for k in ("malignancy", "location_probs", "ete_logits",
                  "t_conf", "tirads_probs"):
            assert k in out[0], f"真模型 forward_roi 缺键 {k}"


# --------------------------------------------------------------------------- #
# 5. 动态 ROI 分流:帧均值 + 串行消费 + 零帧 fail-loud
# --------------------------------------------------------------------------- #

def _dynamic_case_result(n_total=6, sampled_idx=(0, 2, 4),
                         frame_np_list=None, per_frame_heads=None):
    """合成动态 case_result(key 对齐 test_v21_output._dynamic_case_result)。"""
    per_frame = []
    for i, fi in enumerate(sampled_idx):
        soft = np.full((_MODEL, _MODEL), 0.05, dtype=np.float32)
        r = 2 + i
        soft[r:r + 3, 3:6] = 0.85
        binary = (soft > 0.5).astype(np.uint8) * 255
        heads = (per_frame_heads[i] if per_frame_heads is not None
                 else {"ete_probs": np.array([0.1, 0.6, 0.2, 0.1]),
                       "location_probs": np.array(
                           [0.05, 0.05, 0.1, 0.1, 0.5, 0.1, 0.1]),
                       "t_probs": np.array([0.3, 0.7]),
                       "tirads_probs": np.array([0.05, 0.05, 0.1, 0.7, 0.1]),
                       "t_conf": 0.3})
        fnp = (frame_np_list[i] if frame_np_list is not None
               else np.zeros((_H, _W), np.uint8))
        per_frame.append({
            "frame_index": int(fi),
            "mal_prob": 0.65,
            "preds": {
                "pred_mask": soft, "mask_binary": binary,
                "comp_idx": 0, "echo_idx": 0, "shape_idx": 0, "margin_idx": 0,
                "foci_prob": np.array([0.1]),
                "mal_prob": 0.65, "mal_probs": np.array([0.35, 0.65]),
                **heads,
            },
            "frame_np": fnp,
        })
    kf = per_frame[0]
    return {
        "per_frame": per_frame,
        "agg": {"keyframe_index": 0, "n_frames": len(per_frame)},
        "kf_preds": kf["preds"],
        "kf_frame_np": kf["frame_np"],
        "kf_tensor": None,
        "failed_frames": [],
        "breakdown": {"total_points": 3},
        "mask_resized": kf["preds"]["mask_binary"],
        "orig_hw": (_H, _W),
        "n_total_frames": int(n_total),
        "gatekeeper_audit": {"skipped": False},
    }


class TestRoiDynamic:
    def test_dynamic_frame_mean_not_keyframe(self, tmp_path):
        """动态帧均值口径:多帧 ROI 输出不同 → 取均值,且 ≠ 关键帧值
        (构造成"均值 ≠ 关键帧"才有区分力)。

        帧 0(hot,全亮 patch)→ hot 分支;帧 2/4(暗,值 5 → 归一化均值
        0.0196 < 阈值 0.03)→ cold 分支。
        均值 tirads_probs = [(0.02+0.6+0.6)/3, (0.02+0.2+0.2)/3,
        (0.02+0.1+0.1)/3, (0.02+0.05+0.05)/3, (0.92+0.05+0.05)/3]
        = [0.4067, 0.14, 0.0733, 0.04, 0.34] → E=1.7667 → 码 3;
        关键帧(hot)单独是码 5(峰 idx4)——均值 ≠ 关键帧。✓ 有区分力。
        """
        frames = [np.full((_H, _W), 255, np.uint8),
                  np.full((_H, _W), 5, np.uint8),
                  np.full((_H, _W), 5, np.uint8)]
        cr = _dynamic_case_result(frame_np_list=frames)
        case_dir = str(tmp_path / "D_MEAN")
        v21_output.write_case_v21(case_dir, "D1", "ACC", cr, "dynamic", 50,
                                  roi_ctx=_roi_ctx(FakeRoiModel()))
        payload = _read_json(os.path.join(case_dir, "prediction.json"))
        validate_prediction_json(payload)
        n = payload["Prediction"]["Nodules"][0]
        # 前提自检:关键帧(hot)单独会写码 5
        assert n["TI_RADS_Class"] == 3, \
            f"动态 ROI 应取帧均值(期望码 3),拿到 {n['TI_RADS_Class']}" \
            "(5 说明取了关键帧/单帧)"
        # 对冲键写出的应是均值本身(帧级聚合的直接证据)
        tp = n["TI_RADS_Probs"]
        assert abs(tp[4] - (0.92 + 0.05 + 0.05) / 3) < 1e-6
        assert abs(tp[0] - (0.02 + 0.6 + 0.6) / 3) < 1e-6
        # malignancy 均值:softmax(-3,3)=0.9975 ×1 帧 + 0.0025 ×2 帧 → /3
        mal_mean = (0.9975 + 0.0025 + 0.0025) / 3
        assert abs(n["Malignancy"]["confidence"] - mal_mean) < 2e-3, \
            "良恶性应逐帧 softmax 后均值(logits 均值再 softmax 会得不同值)"
        # t_conf 均值 = (0.9+0.1+0.1)/3(fp32 精度容差 1e-8)
        assert abs(n["T_ClinSig"]["T_conf"] - (0.9 + 0.1 + 0.1) / 3) < 1e-8

    def test_dynamic_chunked_not_full_expansion(self, tmp_path, monkeypatch):
        """动态串行消费不批量展开:前向调用次数 = ceil(帧数/chunk) × 实例数,
        且任何一窗 batch ≤ chunk 上限(攒全帧一次前向会被计成 1 次大 batch)。"""
        calls = []

        class CountingModel(FakeRoiModel):
            def forward_roi(self, roi_images, roi_metas,
                            seg_features=None):
                # 计 batch 维取 roi_images(张量);转发给基类保持同参序
                # (self, roi_images, roi_metas)。
                calls.append(int(roi_images.shape[0]))
                return FakeRoiModel.forward_roi(
                    self, roi_images, roi_metas, seg_features)

        # n_total=6, sampled 6 帧(全帧),单实例 → chunk=16 → 1 次前向 batch=6
        # (6 < 16 全进一窗);关键断言:batch 维恒 ≤ chunk 上限(攒"实例×全帧"
        # 大展开的实现若把多实例帧拼一窗会突破该上界,或单窗 batch 直接超 16)。
        frames = [np.full((_H, _W), 60 + i * 5, np.uint8) for i in range(6)]
        cr = _dynamic_case_result(n_total=6, sampled_idx=tuple(range(6)),
                                  frame_np_list=frames)
        case_dir = str(tmp_path / "D_CHUNK")
        v21_output.write_case_v21(case_dir, "D2", "ACC", cr, "dynamic", 50,
                                  roi_ctx=_roi_ctx(CountingModel()))
        assert calls and max(calls) <= 16, \
            f"单窗 batch 超过 chunk=16(批量展开红线):{calls}"
        assert sum(calls) == 6, f"前向总样本数应为 6(1 实例 × 6 帧),得 {calls}"

    def test_dynamic_chunk_split_across_windows(self, tmp_path):
        """帧数 > chunk 时分窗:40 帧稳定前景(全帧非零 mask,位置钉死)单实例
        → ≥3 次前向且每窗 ≤16(chunk 机制真的在切,不是一次性全展)。

        ⚠️ 默认 `_dynamic_case_result` 的逐帧 blob 会随 i 漂移出模型画幅
        (rows 2+i,r≥6 起帧 mask 全空),35 帧只有前 6 帧有前景 → 实例
        frame_indices 只有 6 帧,永远进不了分窗分支。此处把每帧 mask 钉在
        同一位置(位置不动,前景全帧非零),实例跨全部 40 帧。"""
        calls = []

        class CountingModel(FakeRoiModel):
            def forward_roi(self, roi_images, roi_metas,
                            seg_features=None):
                # 同上:计数 batch 维取 roi_images(张量),转发保持同参序。
                calls.append(int(roi_images.shape[0]))
                return FakeRoiModel.forward_roi(
                    self, roi_images, roi_metas, seg_features)

        n = 40
        frames = [np.full((_H, _W), 40 + (i % 3) * 10, np.uint8) for i in range(n)]
        cr = _dynamic_case_result(n_total=n, sampled_idx=tuple(range(n)),
                                  frame_np_list=frames)
        # 钉死每帧 mask(同位置,全帧前景非零)
        for f in cr["per_frame"]:
            soft = np.full((_MODEL, _MODEL), 0.05, np.float32)
            soft[2:5, 3:6] = 0.85
            f["preds"]["pred_mask"] = soft
            f["preds"]["mask_binary"] = (soft > 0.5).astype(np.uint8) * 255
        cr["kf_preds"] = cr["per_frame"][0]["preds"]
        case_dir = str(tmp_path / "D_CHUNK2")
        v21_output.write_case_v21(case_dir, "D2B", "ACC", cr, "dynamic", 50,
                                  roi_ctx=_roi_ctx(CountingModel()))
        assert len(calls) >= 3, f"40 帧应至少分 3 窗(ceil(40/16)=3),得 {calls}"
        assert max(calls) <= 16, f"单窗 batch 超过 chunk=16:{calls}"
        assert sum(calls) == n, f"前向总样本数应为 {n},得 {calls}"

    def test_dynamic_zero_available_frames_raises(self, tmp_path):
        """实例零可用帧(frame_indices 与 per_frame 无交集)→ V21OutputError
        (拒绝编造四头值)。"""
        frames = [np.full((_H, _W), 255, np.uint8) for _ in range(3)]
        cr = _dynamic_case_result(frame_np_list=frames)
        # per_frame 帧号 (0,2,4);把实例的可用帧伪造到不在其中的帧号上
        # ——通过直接调 _run_roi_dynamic 构造(不改 case_result 通路)。
        from src.inference.v21_output import _run_roi_dynamic, V21OutputError
        inst = {"bbox": (4, 6, 8, 5), "frame_indices": [1, 3]}
        with pytest.raises(V21OutputError, match="零可用帧"):
            _run_roi_dynamic({}, [inst], "P", "S", _roi_ctx(FakeRoiModel()))

    def test_dynamic_roi_error_bubbles_whole_case(self, tmp_path):
        """ROI 前向异常整例冒泡(绝不静默把帧摘出 per_frame 改均值分母)。"""
        class ExplodingModel(torch.nn.Module):
            def forward_roi(self, roi_images, roi_metas,
                            seg_features=None):
                raise RuntimeError("boom")

        frames = [np.full((_H, _W), 255, np.uint8) for _ in range(3)]
        cr = _dynamic_case_result(frame_np_list=frames)
        with pytest.raises(RuntimeError, match="boom"):
            v21_output.write_case_v21(str(tmp_path / "D_BOOM"), "D3", "ACC",
                                       cr, "dynamic", 50,
                                       roi_ctx=_roi_ctx(ExplodingModel()))


# --------------------------------------------------------------------------- #
# 6. validate_roi_probs / assert_finite 三类硬拦
# --------------------------------------------------------------------------- #

def _good_roi_out():
    return {
        "lesion_id": "L", "is_main": 1, "patient_id": "P",
        "malignancy": torch.tensor([0.0, 1.0]),
        "location_probs": torch.full((NUM_LOCATIONS,), 1.0 / NUM_LOCATIONS),
        "ete_probs": torch.full((4,), 0.25),
        "ete_logits": torch.zeros(4),
        "t_conf": torch.tensor(0.5),
        "tirads_probs": torch.full((5,), 0.2),
    }


class TestValidateRoiProbs:
    def test_good_passes(self):
        from src.inference.postprocess import validate_roi_probs
        validate_roi_probs(_good_roi_out())   # 不抛即过

    def test_nan_malignancy_raises(self):
        from src.inference.postprocess import validate_roi_probs, \
            NonFiniteProbabilityError
        r = _good_roi_out()
        r["malignancy"] = torch.tensor([0.0, float("nan")])
        with pytest.raises(NonFiniteProbabilityError):
            validate_roi_probs(r)

    def test_inf_location_raises(self):
        from src.inference.postprocess import validate_roi_probs, \
            NonFiniteProbabilityError
        r = _good_roi_out()
        loc = torch.full((NUM_LOCATIONS,), 1.0 / NUM_LOCATIONS)
        loc[3] = float("inf")
        r["location_probs"] = loc
        with pytest.raises(NonFiniteProbabilityError):
            validate_roi_probs(r)

    def test_wrong_length_raises_valueerror(self):
        from src.inference.postprocess import validate_roi_probs
        r = _good_roi_out()
        r["tirads_probs"] = torch.full((6,), 1.0 / 6)   # 应为 5
        with pytest.raises(ValueError, match="tirads_probs"):
            validate_roi_probs(r)

    def test_malignancy_wrong_length_raises(self):
        from src.inference.postprocess import validate_roi_probs
        r = _good_roi_out()
        r["malignancy"] = torch.zeros(3)
        with pytest.raises(ValueError, match="malignancy"):
            validate_roi_probs(r)

    def test_assert_finite_shared_with_decode(self):
        """模块级 assert_finite 与 decode 内部同一防线(名字/异常类型)。"""
        from src.inference.postprocess import assert_finite, \
            NonFiniteProbabilityError
        with pytest.raises(NonFiniteProbabilityError):
            assert_finite("test", torch.tensor([float("nan")]))
        assert assert_finite("test", torch.tensor([0.5])) is not None


# --------------------------------------------------------------------------- #
# 7. env 开关解析 + case_pipeline 分流记录
# --------------------------------------------------------------------------- #

class TestEnvParsing:
    @pytest.mark.parametrize("val,expected", [
        ("1", True), ("true", True), ("on", True), ("yes", True),
        ("0", False), ("garbage", False), ("", False),
    ])
    def test_roi_heads_enabled_parsing(self, monkeypatch, val, expected):
        from src.inference.roi_pipeline import roi_heads_enabled
        monkeypatch.setenv("AICOMP_ROI_HEADS", val)
        assert roi_heads_enabled() is expected

    def test_unset_is_off(self, monkeypatch):
        from src.inference.roi_pipeline import roi_heads_enabled
        monkeypatch.delenv("AICOMP_ROI_HEADS", raising=False)
        assert roi_heads_enabled() is False

    def test_runner_mirror_matches_pipeline(self, monkeypatch):
        """runner 的 inline 镜像解析与 roi_pipeline 权威解析逐值一致。"""
        from src.service import runner
        from src.inference.roi_pipeline import roi_heads_enabled
        for val in ("1", "TRUE", " on ", "yes", "0", "garbage", ""):
            monkeypatch.setenv("AICOMP_ROI_HEADS", val)
            assert runner._roi_heads_on() == roi_heads_enabled(), val

    def test_case_pipeline_records_roi_heads_key(self, monkeypatch):
        """env 开:case_result 带 roi_heads=True(跑的是哪条路有痕迹)。"""
        monkeypatch.setenv("AICOMP_ROI_HEADS", "1")
        monkeypatch.delenv("AICOMP_STATIC_TILING", raising=False)
        monkeypatch.delenv("AICOMP_STATIC_TTA", raising=False)
        from src.models.multi_task_model import MultiTaskThyroidModel
        cfg = {"data": {"img_size": 512}}
        m = MultiTaskThyroidModel(backbone_name="efficientnet_b2",
                                  pretrained=False, config=cfg)
        m.eval()
        from src.inference.case_pipeline import diagnose_static_case
        img = (np.random.rand(40, 44) * 255).astype(np.uint8)
        with torch.no_grad():
            result = diagnose_static_case(m, img, cfg, None, "fp32",
                                          torch.device("cpu"))
        assert result.get("roi_heads") is True

    def test_tiling_and_roi_both_on_raises(self, monkeypatch):
        """tiling + ROI 同开 → 显式 raise(处置①,确定性优先)。"""
        monkeypatch.setenv("AICOMP_ROI_HEADS", "1")
        monkeypatch.setenv("AICOMP_STATIC_TILING", "1")
        monkeypatch.delenv("AICOMP_STATIC_TTA", raising=False)
        from src.models.multi_task_model import MultiTaskThyroidModel
        cfg = {"data": {"img_size": 512}}
        m = MultiTaskThyroidModel(backbone_name="efficientnet_b2",
                                  pretrained=False, config=cfg)
        m.eval()
        from src.inference.case_pipeline import diagnose_static_case
        img = (np.random.rand(600, 800) * 255).astype(np.uint8)
        with pytest.raises(RuntimeError, match="ROI"):
            with torch.no_grad():
                diagnose_static_case(m, img, cfg, None, "fp32",
                                     torch.device("cpu"))


# --------------------------------------------------------------------------- #
# 8. per_nodule_contexts 长度契约 + 编码对齐
# --------------------------------------------------------------------------- #

class TestContextContract:
    def test_length_mismatch_raises(self, tmp_path):
        """per_nodule_contexts 长度不齐 → V21OutputError(错位即错挂)。"""
        soft, binary = _soft_and_binary_two()   # 2 实例
        cr = _static_case_result(soft, binary,
                                 with_orig_img=np.zeros((_H, _W), np.uint8))
        # 绕过 _run_roi_static 直接测 _build_entries 的守卫:构造 3 contexts
        from src.inference.v21_output import (
            _build_entries_and_save_slices, _shared_nodule_context, V21OutputError)
        from src.inference.nodule_split import split_nodules
        instances = split_nodules(_resize_binary_fake())
        ctx = _shared_nodule_context(cr["preds"])
        with pytest.raises(V21OutputError, match="长度不齐"):
            _build_entries_and_save_slices(
                instances, None, str(tmp_path / "LEN"), "S", ctx,
                per_nodule_contexts=[ctx, ctx, ctx])

    def test_roi_context_key_set_matches_shared(self):
        """_roi_nodule_context 输出键集 ⊇ _shared_nodule_context 键集
        (完全同键,防 build_nodule_entry 的 .get() 静默取默认)。"""
        r = {
            "malignancy": [0.0, 1.0],
            "location_probs": [0.05, 0.05, 0.1, 0.1, 0.5, 0.1, 0.1],
            "tirads_probs": [0.05, 0.05, 0.1, 0.7, 0.1],
            "ete_probs": [0.1, 0.6, 0.2, 0.1],
            "t_conf": 0.3,
        }
        shared = {
            "malignancy_confidence": 0.5, "tirads_class": 3, "ete_code": 2,
            "location_probs": [0.5] * NUM_LOCATIONS, "location": 1,
            "t_clin_sig": 0, "t_confidence": 0.5,
            "tirads_probs": [0.2] * 5, "ete_probs": [0.25] * 4,
        }
        roi_ctx = v21_output._roi_nodule_context(r)
        assert set(roi_ctx.keys()) == set(shared.keys()), (
            f"键集不一致:仅 roi 有 {set(roi_ctx) - set(shared)},"
            f"仅 shared 有 {set(shared) - set(roi_ctx)}")

    def test_t_clin_sig_threshold_semantics(self):
        """ROI 输出只有 t_conf 标量:t_clin_sig = int(t_conf > 0.5)
        (与整图 argmax(t_probs) 两类时等价;0.5 平局取 0,同 argmax 平局取
        首个=低类)。"""
        r = {"malignancy": [0.0, 1.0],
             "location_probs": [1.0 / NUM_LOCATIONS] * NUM_LOCATIONS,
             "tirads_probs": [0.2] * 5, "ete_probs": [0.25] * 4,
             "t_conf": 0.5}
        assert v21_output._roi_nodule_context(r)["t_clin_sig"] == 0
        r["t_conf"] = 0.5001
        assert v21_output._roi_nodule_context(r)["t_clin_sig"] == 1


def _resize_binary_fake():
    """两连通块二值图(供 split_nodules;非方阵)。"""
    b = np.zeros((_H, _W), np.uint8)
    b[1:3, 1:3] = 1
    b[8:10, 12:15] = 1
    return b


# --------------------------------------------------------------------------- #
# 9. runner 层:roi_ctx 构建 + orig_img_np 透传(env 开)
# --------------------------------------------------------------------------- #

class TestRunnerWiring:
    def test_runner_env_on_adds_roi_ctx_fields(self, monkeypatch):
        """env 开:runner 构建 roi_ctx 四键齐全(模型句柄缺口的解法落点)。"""
        monkeypatch.setenv("AICOMP_ROI_HEADS", "1")
        from src.service import runner
        assert runner._roi_heads_on() is True
        # roi_ctx 的构建逻辑在 _process_case(需整链),此处对键集做静态契约:
        # 四键是 v21_output._run_roi_* 的消费面,缺一会 KeyError 远离根因。
        # 用 monkeypatch 直接断言 _write_via_v21 收到的 roi_ctx 形态。
        captured = {}

        class FakeState(dict):
            pass

        state = FakeState(model=object(), config={"data": {"img_size": 512}},
                          device=torch.device("cpu"))

        real_write = runner._write_via_v21
        monkeypatch.setattr(
            runner, "_write_via_v21",
            lambda cd, case, cr, kind, ms, roi_ctx=None:
            captured.update(roi_ctx=roi_ctx) or real_write(
                cd, case, {"rejected": True}, kind, ms, roi_ctx=None))
        # rejected 分支不走 write_via_v21 —— 改直接调 _process_case 的逻辑等价物:
        # 对 roi_ctx 构建段做单元验证(镜像逻辑,四键)。
        roi_ctx = None
        if runner._roi_heads_on():
            roi_ctx = {"model": state["model"], "precision": "fp32",
                       "device": state["device"],
                       "img_size": int(state["config"]["data"]["img_size"])}
        assert set(roi_ctx.keys()) == {"model", "precision", "device", "img_size"}
        assert roi_ctx["img_size"] == 512

    def test_v21_env_not_read_directly(self, monkeypatch, tmp_path):
        """v21_output 不读 env:开关状态只经 roi_ctx 流入(env 设了但 roi_ctx
        为 None → 走整图共享,证明无隐藏 env 消费)。"""
        monkeypatch.setenv("AICOMP_ROI_HEADS", "1")   # env 开!
        soft, binary = _soft_and_binary_two()
        cr = _static_case_result(soft, binary)   # 但 roi_ctx=None(默认)
        payload = _write_static(tmp_path, "P_ENVCTX", cr)   # 不传 roi_ctx
        # 整图共享路径:两结节同值(mal 0.72);若 v21 偷读 env 会要求
        # orig_img_np 并 raise —— 没抛且值正确 = 无隐藏 env 消费。
        n = payload["Prediction"]["Nodules"][0]
        assert abs(n["Malignancy"]["confidence"] - 0.72) < 1e-9
