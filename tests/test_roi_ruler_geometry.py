# -*- coding: utf-8 -*-
"""M2 坐标几何测试债(E2 review 实锤,docs/新架构文件级改动清单_20260925.md
§J.7 表 "M2 轴翻转零杀" 行):唯一几何测试此前是**循环定义**(自己造输入
自己断言,坐标类改动零区分力)—— 轴翻转(x/y 互换)变异当年 SURVIVED。

本文件补"真 `split_nodules` + 逐轴几何断言":

- 用**非对称** GT blob(x/y 幅面刻意不同,互换后 bbox 落不进 GT 半开区间)
  跑真 `split_nodules`,断言每个输出实例的 bbox 各轴落在 GT mask 的
  **半开区间**内(口径参照 `official_loader._roi_lesion_meta` 的
  `bbox_xyxy` 半开语义:x1/y1 = 最大非零列/行 + 1;以及
  `roi_pipeline._roi_instance_bbox` 的 (x, y, x+w, y+h) 转换);
- 静态 2D 与动态 3D 两条路都测(实例 bbox 前 4 项同为 xywh);
- 变异自证:x/y 轴互换注入进实例 bbox → 测试必须 RED(轴翻转正是当年
  SURVIVED 的变异);恢复后 GREEN。注入前后的输出摘要见测试内注释与本
# MUTATION EVIDENCE(2026-09-25,本文件落地轮实跑,注入 = 实例 bbox
# 前 4 项 (x,y) 轴互换,正是当年 SURVIVED 的那个变异):
#   bbox 单翻转 → 2D 1 红 + 3D 2 红(blobA 的 80/30 互换后仍是图内合法
#   框,centroid/mask 两条天然免疫 bbox-only 变异);
#   bbox+mask+centroid 全翻转 → 2D 3 条全红;
#   收编 = TestMutationAxisSwap 以 monkeypatch 正式钉死该变异,每轮 CI
#   自动重放,不依赖落地轮的一次性证据。
#   非对称 blob(x 幅 80、y 幅 30)是区分力来源:对称 blob 上互换后
#   断言仍成立 = 零区分力(当年测试债的根因)。
"""

import numpy as np
import pytest

import sklearn.metrics  # noqa: F401  (import 副作用:抢先加载 OpenMP 运行时,
# 防 pytest 收集深度下 sklearn→pandas→pyarrow 链 Windows 栈溢出,
# test_cliff_loss.py / test_roi_records.py 同款)

from src.inference.nodule_split import split_nodules



def _assert_bbox_within_gt(bbox, gt_xyxy, ctx):
    """实例 bbox (x,y,w,h) → 半开 (x0,y0,x1,y1) 后须与 GT 半开框相等。

    相等(而非"落在内")的原因:GT blob 是**单连通域**,`split_nodules`
    的外接框定义与手算半开框是同一几何对象 —— 相等断言既验轴序(互换必
    越出),又验半开/闭开 off-by-one(x1 少 1 会红)。"""
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    got = (x, y, x + w, y + h)
    assert got == gt_xyxy, (
        f"{ctx}: 实例半开框 {got} ≠ GT 手算半开框 {gt_xyxy}"
        " —— 轴序/半开口径漂移(x/y 互换或 x1 少 1 都落在这里)"
    )


class TestGeometry2D:
    def test_asymmetric_blobs_bbox_matches_gt_halfopen(self):
        """两个**非对称**不连通 blob(各轴幅面刻意不同)→ 逐实例 bbox
        各轴与 GT 半开框逐项相等(轴翻转变异在此必红)。"""
        m = np.zeros((512, 512), np.uint8)
        # blob A:行 100-130(y 幅 30)、列 60-140(x 幅 80)—— x/y 非对称
        m[100:130, 60:140] = 1
        # blob B:行 300-420(y 幅 120)、列 350-390(x 幅 40)
        m[300:420, 350:390] = 1
        insts = split_nodules(m, min_area_ratio=0)
        assert len(insts) == 2
        # GT 手算半开框(独立基准,不 import 被测模块):
        #   blob A:cols 60..139、rows 100..129 → (60, 100, 140, 130)
        #   blob B:cols 350..389、rows 300..419 → (350, 300, 390, 420)
        # 逐实例对号(按面积:blob B 120×40=4800 > blob A 30×80=2400)
        areas = sorted([i["area"] for i in insts], reverse=True)
        assert areas == [4800, 2400]
        by_area = {i["area"]: i for i in insts}
        big, small = by_area[4800], by_area[2400]
        _assert_bbox_within_gt(big["bbox"], (350, 300, 390, 420), "blobB")
        _assert_bbox_within_gt(small["bbox"], (60, 100, 140, 130), "blobA")

    def test_centroid_inside_gt_halfopen(self):
        """质心 (cx, cy) 须落在 GT 半开框内(cx∈[x0,x1)、cy∈[y0,y1))——
        轴翻转会把质心投到错误轴上,非对称 blob 上必红。"""
        m = np.zeros((512, 512), np.uint8)
        m[100:130, 60:140] = 1
        inst = split_nodules(m, min_area_ratio=0)[0]
        cx, cy = inst["centroid"]
        # 手算:均匀矩形 → cx=(60+139)/2=99.5、cy=(100+129)/2=114.5
        assert 60 <= cx < 140, f"cx={cx} 落 GT x 半开区间外(轴翻转指纹)"
        assert 100 <= cy < 130, f"cy={cy} 落 GT y 半开区间外(轴翻转指纹)"
        assert cx == pytest.approx(99.5)
        assert cy == pytest.approx(114.5)

    def test_instance_mask_nonzero_within_gt_halfopen(self):
        """实例 mask 的非零像素全部落在 GT 半开框内(内容级轴校验:
        bbox 对但 mask 内容翻转时此条红)。"""
        m = np.zeros((512, 512), np.uint8)
        m[100:130, 60:140] = 1
        inst = split_nodules(m, min_area_ratio=0)[0]
        ys, xs = np.nonzero(inst["mask"])
        assert ys.min() >= 100 and ys.max() <= 129
        assert xs.min() >= 60 and xs.max() <= 139


class TestGeometry3D:
    def test_3d_bbox_xywh_and_z_closed(self):
        """3D 实例 bbox = (x,y,w,h,z0,z1),前 4 项 xywh 与 2D 同口径,
        z 闭区间;xy 须与 GT 空间半开框一致(轴翻转在此必红)。"""
        v = np.zeros((8, 512, 512), np.uint8)
        # 非对称:x 幅 80(cols 60-139)、y 幅 30(rows 100-129)、z 帧 2-5
        v[2:6, 100:130, 60:140] = 1
        inst = split_nodules(v, min_area_ratio=0)[0]
        x, y, w, h, z0, z1 = inst["bbox"]
        # xy 半开口径:与 2D 同款手算
        _assert_bbox_within_gt((x, y, w, h), (60, 100, 140, 130), "3D xy")
        # z 闭区间(契约:z1 闭,即 z∈{2,3,4,5})
        assert (z0, z1) == (2, 5), "z 段须为闭区间 (2,5)"
        assert inst["frame_indices"] == [2, 3, 4, 5]

    def test_3d_two_blobs_axis_alignment(self):
        """两个空间分离的非对称 3D blob:逐实例 xy 框对号入座(面积降序),
        互换实例/轴翻转都必红。"""
        v = np.zeros((8, 512, 512), np.uint8)
        v[2:6, 100:130, 60:140] = 1      # A:area 4 帧×30×80=9600,z 2-5
        v[2:5, 300:420, 350:390] = 1     # B:area 3 帧×120×40=14400,z 2-4
        insts = split_nodules(v, min_area_ratio=0)
        assert len(insts) == 2
        assert [i["area"] for i in insts] == [14400, 9600]
        _assert_bbox_within_gt(insts[0]["bbox"][:4], (350, 300, 390, 420), "3D B")
        _assert_bbox_within_gt(insts[1]["bbox"][:4], (60, 100, 140, 130), "3D A")
        assert insts[0]["bbox"][4:] == (2, 4) and insts[1]["bbox"][4:] == (2, 5)


class TestMutationAxisSwap:
    """M2 变异杀测收编:轴翻转(x/y 互换)当年在旧几何测试上 SURVIVED,
    此处 monkeypatch 注入进 `_split_2d`/`_split_3d` 的实例 bbox,断言
    几何测试的**断言层**必须抓住它 —— 每轮 CI 自动重放该变异。"""

    def test_axis_swap_bbox_is_caught(self, monkeypatch):
        import src.inference.nodule_split as ns

        real_2d = ns._split_2d

        def mutated_2d(arr, min_area_ratio, max_nodules):
            out = real_2d(arr, min_area_ratio, max_nodules)
            for inst in out:
                x, y, w, h = inst["bbox"]
                inst["bbox"] = (y, x, h, w)
            return out

        monkeypatch.setattr(ns, "_split_2d", mutated_2d)
        m = np.zeros((512, 512), np.uint8)
        m[100:130, 60:140] = 1   # 非对称:x 幅 80、y 幅 30
        inst = ns.split_nodules(m, min_area_ratio=0)[0]
        # 期望框 = GT 手算半开;轴翻转产出 (100,60,80,30) → 半开
        # (100,60,180,90) ≠ (60,100,140,130),断言必须死(红 = 变异
        # 被抓住,测试本身须通过 = "变异被杀"是被断言的行为)。
        # 变异杀测语义:注入的轴翻转必须**被上面的几何断言口径识破**
        # (产出 ≠ GT 手算框 → 测试通过 = 区分力在;若产出 == GT 框,
        # 说明非对称 blob 退化成对称/断言口径失去轴序敏感性 = M2 复发)。
        x, y, w, h = inst["bbox"]
        assert (x, y, x + w, y + h) != (60, 100, 140, 130), (
            f"轴翻转变异未被识破(产出仍 == GT 框 {(x, y, x + w, y + h)})——"
            "几何断言失去轴序区分力(M2 测试债复发)"
        )

    def test_axis_swap_3d_bbox_is_caught(self, monkeypatch):
        import src.inference.nodule_split as ns

        real_3d = ns._split_3d

        def mutated_3d(arr, min_area_ratio, max_nodules, close_gap_frames):
            out = real_3d(arr, min_area_ratio, max_nodules, close_gap_frames)
            for inst in out:
                x, y, w, h = inst["bbox"][:4]
                inst["bbox"] = (y, x, h, w) + tuple(inst["bbox"][4:])
            return out

        monkeypatch.setattr(ns, "_split_3d", mutated_3d)
        v = np.zeros((8, 512, 512), np.uint8)
        v[2:6, 100:130, 60:140] = 1
        inst = ns.split_nodules(v, min_area_ratio=0)[0]
        x, y, w, h = inst["bbox"][:4]
        assert (x, y, x + w, y + h) != (60, 100, 140, 130), (
            "3D 轴翻转变异未被识破(产出仍 == GT 框)—— 前四项 xywh "
            "轴序区分力丢失(M2 复发)"
        )
