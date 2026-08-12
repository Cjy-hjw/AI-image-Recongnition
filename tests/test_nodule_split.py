"""`src/inference/nodule_split.py` 回归测试。

**`TestFrameGapClosing::test_gap_at_first_frame_survives` 是本文件最重要的一条** ——
它锁死一个开发期实测发现的真实 bug:`scipy.ndimage.binary_closing` 的腐蚀步骤默认
`border_value=0`(数组外视作背景),会把**贴着首帧/末帧边界**的前景整个吃掉。实测
(帧0,帧2) 有、帧1 漏检的结节,默认 closing 后变成 (帧1,帧2)——帧 0 凭空消失、
帧 1 凭空出现。结节出现在视频首/末帧极其常见,该行为等于系统性丢帧,且产出的
mask 完全合法、看不出任何异常。
"""

import numpy as np
import pytest

from src.inference.nodule_split import (
    crop_nodule,
    detection_confidence,
    split_nodules,
)


def _two_nodule_mask(h=200, w=300):
    """大结节 40×60=2400px、小结节 20×30=600px、单像素噪点各一。"""
    m = np.zeros((h, w), np.uint8)
    m[20:60, 30:90] = 255
    m[120:140, 200:230] = 255
    m[5, 5] = 255
    return m


class TestSplit2D:
    def test_separates_instances_and_drops_noise(self):
        nodules = split_nodules(_two_nodule_mask())
        assert len(nodules) == 2, "单像素噪点应被 min_area_ratio 滤除"
        assert [n["area"] for n in nodules] == [2400, 600], "必须按面积降序"

    def test_bbox_is_xywh(self):
        """bbox 顺序错位(如返回 y,x)会让下游 crop 裁到完全错误的区域,而形状仍合法。"""
        n = split_nodules(_two_nodule_mask())[0]
        assert n["bbox"] == (30, 20, 60, 40)

    def test_instance_masks_are_binary_full_size_and_disjoint(self):
        m = _two_nodule_mask()
        a, b = split_nodules(m)
        for n in (a, b):
            assert n["mask"].shape == m.shape, "实例 mask 必须是原图尺寸(规范硬约束)"
            assert set(n["mask"].flatten().tolist()) <= {0, 1}
        assert int((a["mask"] & b["mask"]).sum()) == 0, "实例间不得重叠"

    def test_negative_case_returns_empty_list(self):
        """规范:阴性病例输出空文件夹/空数组。空列表是合法结果,不是错误。"""
        assert split_nodules(np.zeros((100, 100), np.uint8)) == []

    def test_diagonal_touching_counted_as_one(self):
        """connectivity=8:斜向相邻属同一结节。4-邻域会把阶梯状边界切成多块、虚增结节数。"""
        m = np.zeros((50, 50), np.uint8)
        m[10:20, 10:20] = 1
        m[20:30, 20:30] = 1  # 仅对角接触
        assert len(split_nodules(m, min_area_ratio=0)) == 1

    def test_min_area_ratio_scales_with_image_size(self):
        """相对阈值:同样 100px 的斑点,在小图上保留、在大图上滤除。"""
        small = np.zeros((100, 100), np.uint8)
        small[0:10, 0:10] = 1
        assert len(split_nodules(small)) == 1  # 100px vs 阈值 1

        large = np.zeros((2000, 2000), np.uint8)
        large[0:10, 0:10] = 1
        assert split_nodules(large) == []  # 100px vs 阈值 400

    def test_max_nodules_truncates_by_area(self):
        m = np.zeros((200, 200), np.uint8)
        for i in range(5):  # 面积递增,间隔 ≥2px 保证不连通
            m[i * 40: i * 40 + 5 + i * 3, 0:20] = 1
        kept = split_nodules(m, min_area_ratio=0, max_nodules=2)
        assert len(kept) == 2
        assert kept[0]["area"] > kept[1]["area"], "截断应保留面积最大的"

    def test_accepts_0_255_and_arbitrary_nonzero(self):
        m = np.zeros((100, 100), np.float32)
        m[10:30, 10:30] = 0.7
        assert split_nodules(m)[0]["area"] == 400


class TestSplit3D:
    def test_separate_nodules_not_merged(self):
        v = np.zeros((4, 50, 50), np.uint8)
        v[0, 2:8, 2:8] = 1
        v[1, 40:46, 40:46] = 1  # 时间相邻但空间分离
        assert len(split_nodules(v, min_area_ratio=0)) == 2

    def test_bbox_has_z_range(self):
        v = np.zeros((6, 50, 50), np.uint8)
        v[2:5, 10:20, 30:40] = 1
        n = split_nodules(v, min_area_ratio=0)[0]
        assert n["bbox"] == (30, 10, 10, 10, 2, 4), "3D bbox 为 (x,y,w,h,z0,z1),z1 闭区间"
        assert n["frame_indices"] == [2, 3, 4]

    def test_area_accumulates_across_frames(self):
        v = np.zeros((5, 50, 50), np.uint8)
        v[1:4, 10:20, 10:20] = 1  # 3 帧 × 100px
        assert split_nodules(v, min_area_ratio=0)[0]["area"] == 300


class TestFrameGapClosing:
    """时间维缝合:分割在个别帧漏检不应把一个结节拆成两个。"""

    def test_gap_at_first_frame_survives(self):
        """**回归锁**:帧0 有、帧1 漏检、帧2 有。

        `binary_closing` 默认 border_value=0 会吃掉贴边界的帧0,产出 [1,2] ——
        既丢了真实帧 0,又虚构了帧 1。正确行为是 [0,2]。
        """
        v = np.zeros((8, 100, 100), np.uint8)
        v[0, 10:30, 10:30] = 1
        v[2, 10:30, 10:30] = 1

        nodules = split_nodules(v)
        assert len(nodules) == 1, "跨 1 帧的缺口应被缝合为单个结节"
        assert nodules[0]["frame_indices"] == [0, 2], "帧0 不得被边界腐蚀吃掉"
        assert nodules[0]["area"] == 800, "缝合只影响标签归属,不得把虚构帧写进 mask"

    def test_gap_at_last_frame_survives(self):
        v = np.zeros((6, 50, 50), np.uint8)
        v[3, 5:15, 5:15] = 1
        v[5, 5:15, 5:15] = 1
        n = split_nodules(v)
        assert len(n) == 1 and n[0]["frame_indices"] == [3, 5]

    def test_closing_disabled_splits_instances(self):
        v = np.zeros((8, 100, 100), np.uint8)
        v[0, 10:30, 10:30] = 1
        v[2, 10:30, 10:30] = 1
        assert len(split_nodules(v, close_gap_frames=0)) == 2

    def test_gap_wider_than_span_not_bridged(self):
        """缺口 3 帧 > close_gap_frames=1 的跨度,不应缝合(避免把两次独立出现粘一起)。"""
        v = np.zeros((10, 50, 50), np.uint8)
        v[1, 5:15, 5:15] = 1
        v[5, 5:15, 5:15] = 1
        assert len(split_nodules(v, close_gap_frames=1)) == 2


class TestSplitGuards:
    @pytest.mark.parametrize("arr", [np.zeros(5, np.uint8), np.zeros((2, 3, 4, 5), np.uint8)])
    def test_bad_ndim_raises(self, arr):
        with pytest.raises(ValueError, match="2D|3D|ndim"):
            split_nodules(arr)


class TestCropNodule:
    def test_pads_by_ratio_per_axis(self):
        """边距按宽/高各自计算:细长结节不应被短边的边距撑成方形。"""
        img = np.zeros((200, 300), np.uint8)
        _, box = crop_nodule(img, (30, 20, 60, 40), pad_ratio=0.25)
        assert box == (15, 10, 105, 70)  # pad_x=15, pad_y=10

    def test_clamps_to_image_bounds(self):
        img = np.zeros((100, 100), np.uint8)
        patch, box = crop_nodule(img, (0, 0, 20, 20))
        assert box == (0, 0, 25, 25)
        assert patch.shape == (25, 25)

    def test_accepts_3d_bbox_ignoring_z(self):
        img = np.zeros((100, 100), np.uint8)
        _, box = crop_nodule(img, (10, 10, 20, 20, 2, 5))
        assert box == (5, 5, 35, 35)

    def test_bbox_fully_outside_raises(self):
        with pytest.raises(ValueError, match="无交集|空"):
            crop_nodule(np.zeros((50, 50), np.uint8), (100, 100, 10, 10))

    def test_non_2d_image_raises(self):
        with pytest.raises(ValueError, match="2D"):
            crop_nodule(np.zeros((3, 50, 50), np.uint8), (1, 1, 5, 5))


class TestDetectionConfidence:
    def test_is_region_mean(self):
        p = np.zeros((10, 10))
        p[2:5, 2:5] = 0.8
        m = np.zeros((10, 10), np.uint8)
        m[2:5, 2:5] = 1
        assert detection_confidence(p, m) == pytest.approx(0.8)

    def test_mean_not_max(self):
        """用最大值会让 confidence 几乎恒为 1.0,失去区分度 —— AP 排序会退化成随机。"""
        p = np.zeros((10, 10))
        p[0, 0] = 1.0
        p[0, 1] = 0.2
        m = np.zeros((10, 10), np.uint8)
        m[0, 0:2] = 1
        assert detection_confidence(p, m) == pytest.approx(0.6)

    def test_empty_instance_returns_zero(self):
        assert detection_confidence(np.ones((5, 5)), np.zeros((5, 5), np.uint8)) == 0.0

    def test_shape_mismatch_raises(self):
        """静默广播会得到一个数值上合法但语义无意义的 confidence。"""
        with pytest.raises(ValueError, match="形状不一致"):
            detection_confidence(np.zeros((5, 5)), np.zeros((6, 6), np.uint8))

    def test_3d_instance_averages_all_frames(self):
        """规范:动态数据的 confidence 代表整个序列(病灶级汇总,不按帧重复计数)。"""
        p = np.zeros((4, 10, 10))
        p[0, 0:2, 0:2] = 1.0
        p[1, 0:2, 0:2] = 0.5
        m = np.zeros((4, 10, 10), np.uint8)
        m[0:2, 0:2, 0:2] = 1
        assert detection_confidence(p, m) == pytest.approx(0.75)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_probability_rejected(self, bad):
        """QA_VP §1.7:`np.clip(nan, 0, 1)` **返回 nan 而非钳位** —— NaN 会一路流进
        prediction.json,`json.dumps` 写成字面量 `NaN`(非标准 JSON),严格解析器
        拒绝**整份**文件 ⇒ 该样本零输出,比预测错更严重。"""
        p = np.full((5, 5), bad)
        m = np.ones((5, 5), np.uint8)
        with pytest.raises(ValueError, match="非有限"):
            detection_confidence(p, m)

    def test_non_finite_outside_instance_is_ignored(self):
        """只校验实例区域内的值:区域外的 NaN(背景噪声)不该让整个结节失败。"""
        p = np.full((5, 5), np.nan)
        p[1:3, 1:3] = 0.8
        m = np.zeros((5, 5), np.uint8)
        m[1:3, 1:3] = 1
        assert detection_confidence(p, m) == pytest.approx(0.8)
