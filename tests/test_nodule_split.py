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
    hysteresis_binarize,
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


class TestMinAreaEnv:
    """零训练四件套④(2026-09-10):min_area_ratio 三层优先级。

    显式传参 > env(None 默认时) > DEFAULT(env 未设时)。
    🔴 静态生产靠默认参吃 env(动态生产显式传 DEFAULT 钉死,不走 env)。
    """

    def test_no_env_none_equals_default(self, monkeypatch):
        monkeypatch.delenv("AICOMP_MIN_AREA_RATIO", raising=False)
        large = np.zeros((2000, 2000), np.uint8)
        large[0:10, 0:10] = 1  # 100px,默认阈值 400 → 滤除
        assert split_nodules(large) == []

    def test_env_none_uses_env(self, monkeypatch):
        monkeypatch.setenv("AICOMP_MIN_AREA_RATIO", "2e-5")  # 2000²×2e-5=80px
        large = np.zeros((2000, 2000), np.uint8)
        large[0:10, 0:10] = 1  # 100px > 80 → 现在保留
        assert len(split_nodules(large)) == 1

    def test_explicit_param_beats_env(self, monkeypatch):
        monkeypatch.setenv("AICOMP_MIN_AREA_RATIO", "2e-5")
        large = np.zeros((2000, 2000), np.uint8)
        large[0:10, 0:10] = 1
        # 显式传默认值 → env 不生效 → 滤除(动态生产钉死语义的单元级验证)
        assert split_nodules(large, min_area_ratio=1e-4) == []

    @pytest.mark.parametrize("raw", ["garbage", "-1e-4", "", "inf", "nan",
                                     "1e400"])
    def test_invalid_env_falls_back_to_default(self, monkeypatch, raw):
        """🔴 inf/nan 是 A 路审查 P2 补的反例:inf 曾经只挡负值不挡非有限值,
        int(inf*h*w) 抛 OverflowError 炸整例(docstring 自述"非法回默认"被
        自己的实现破坏)。"""
        monkeypatch.setenv("AICOMP_MIN_AREA_RATIO", raw)
        large = np.zeros((2000, 2000), np.uint8)
        large[0:10, 0:10] = 1
        assert split_nodules(large) == []

    def test_env_zero_disables_filter(self, monkeypatch):
        """env=0 = 完全不过滤(与显式传 0 同义)。🔴 0 语义与 max_nodules
        相反:min_area 的 0=禁用过滤(合法),max_nodules 的 0=归零实例数
        (灾难,见 TestMaxNodulesEnv 的非法集)。"""
        monkeypatch.setenv("AICOMP_MIN_AREA_RATIO", "0")
        m = np.zeros((100, 100), np.uint8)
        m[5, 5] = 1  # 单像素
        assert len(split_nodules(m)) == 1


class TestMaxNodulesEnv:
    """阶段1(2026-09-13,方向A):max_nodules 三层优先级。

    显式传参 > env(None 默认时) > DEFAULT(env 未设时)。
    静/动生产调用点均靠默认参吃 env(与 min_area 不同——动态不钉死,分池
    放开是扫参决策对象)。
    """

    @staticmethod
    def _many_blobs(n=25):
        """25 个互不连通的等面积 blob(超过 DEFAULT_MAX_NODULES=20)。"""
        m = np.zeros((500, 500), np.uint8)
        for i in range(n):
            y, x = (i // 5) * 100 + 40, (i % 5) * 100 + 40
            m[y:y + 20, x:x + 20] = 1
        return m

    def test_no_env_none_equals_default(self, monkeypatch):
        monkeypatch.delenv("AICOMP_MAX_NODULES", raising=False)
        assert len(split_nodules(self._many_blobs(), min_area_ratio=0)) == 20

    def test_env_none_uses_env(self, monkeypatch):
        monkeypatch.setenv("AICOMP_MAX_NODULES", "30")
        assert len(split_nodules(self._many_blobs(), min_area_ratio=0)) == 25

    def test_explicit_param_beats_env(self, monkeypatch):
        monkeypatch.setenv("AICOMP_MAX_NODULES", "30")
        # 显式传 20 → env 不生效(扫参确定性)
        assert len(split_nodules(self._many_blobs(), min_area_ratio=0,
                                 max_nodules=20)) == 20

    @pytest.mark.parametrize("raw", ["garbage", "0", "-5", "20.5", "12.9",
                                     "inf", "nan", "1e400", "", "101",
                                     "1e308", "10000"])
    def test_invalid_env_falls_back_to_default(self, monkeypatch, raw):
        """🔴 非正/非整数/超上界(_MAX_NODULES_CEILING=100)必须挡:
        20.5/12.9 是策略选择(拒绝→20,截断→12/20,断言区分两者——审查实测
        "20.5"对截断型变异零区分力,12.9 截断会返回 12 必然 RED);101/1e308/
        10000 是"静默关掉磁盘兜底"的合法大整数(1e308 通过 isfinite/正数/
        整数三关,nodules[:1e308] 不炸);0 是归零实例数的灾难值(合规空输出、
        静默全零分)。"""
        monkeypatch.setenv("AICOMP_MAX_NODULES", raw)
        assert len(split_nodules(self._many_blobs(), min_area_ratio=0)) == 20

    def test_env_ceiling_allows_decision_values(self, monkeypatch):
        """上界不能拦住放开决策的候选值(30/50/100 须合法生效)。"""
        monkeypatch.setenv("AICOMP_MAX_NODULES", "100")
        assert len(split_nodules(self._many_blobs(n=25), min_area_ratio=0)) == 25


class TestSplit2DMisc:
    """通用 2D 行为(非 env;原在 TestMinAreaEnv 内,TestMaxNodulesEnv 插入
    后归位到独立类,按类名选跑不再漏)。"""

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

    def test_sparse_roundtrip_restores_fullsize_mask(self):
        """🔴 2026-09-09 稀疏存储回归锚:restore 与旧全尺寸实现**逐位一致**。

        shape 断言抓不住 origin 偏移错/内容错位(restore 出的全零或平移图 shape
        也对)——**前景体素总数 == area** 才是能抓的断言(审查 P2-5)。
        """
        from src.inference.nodule_split import restore_instance_mask
        v = np.zeros((6, 50, 60), np.uint8)
        v[2:5, 10:20, 30:40] = 1    # 实例 A:贴内部
        v[0, 45:49, 0:5] = 1        # 实例 B:贴三面边界(锚定 origin 偏移)
        nodules = split_nodules(v, min_area_ratio=0)
        assert len(nodules) == 2
        merged = np.zeros_like(v)
        for n in nodules:
            assert "mask_sparse" in n and "mask" not in n, \
                "3D 实例须稀疏存储(mask_sparse),不应携带全尺寸 mask 键"
            full = restore_instance_mask(n["mask_sparse"], v.shape)
            assert full.shape == v.shape
            assert full.dtype == np.uint8
            assert int(full.sum()) == n["area"], \
                "restore 后前景体素总数须 == area(shape 对但 origin 错位时此断言才红)"
            assert set(np.unique(full).tolist()) <= {0, 1}
            merged |= full
        # 与旧全尺寸实现逐位一致:所有实例按位或后 == 原输入二值图
        assert np.array_equal(merged, (v != 0).astype(np.uint8)), \
            "restore 往返不得丢/挪任何前景体素"
        # 边界实例的 bbox 与 restore 前景位置吻合(防 origin 偏移)
        b = next(n for n in nodules if n["bbox"][4] == 0)  # z0=0 的贴边界实例
        fb = restore_instance_mask(b["mask_sparse"], v.shape)
        assert np.array_equal(
            fb[0, 45:49, 0:5],
            np.ones((4, 5), np.uint8)), "贴边界实例的 restore 前景位置须与原输入一致"


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


def _dual_seed_soft(h=16, w=40):
    """双核 + 低概率桥 + 孤立低概率块。seed=0.9,桥/孤立=0.4。"""
    s = np.zeros((h, w), np.float32)
    s[4:12, 2:8] = 0.9       # 左核 8×6
    s[4:12, 32:38] = 0.9     # 右核 8×6
    s[7:9, 8:32] = 0.4       # 桥
    s[0:3, 18:22] = 0.4      # 孤立,无 seed
    return s


class TestHysteresisBinarize:
    """阶段3 手段4(2026-09-16):滞回阈值。核心不变量=实例候选集由 seed 决定。"""

    def test_identity_when_grow_ge_seed(self):
        s = _dual_seed_soft()
        ident = (s > 0.5).astype(np.uint8) * 255
        assert np.array_equal(hysteresis_binarize(s, 0.5, 0.5), ident)
        assert np.array_equal(hysteresis_binarize(s, 0.5, 0.8), ident)

    def test_value_domain_is_0_or_255(self):
        out = hysteresis_binarize(_dual_seed_soft(), 0.5, 0.3)
        assert out.dtype == np.uint8
        assert set(np.unique(out).tolist()) <= {0, 255}

    def test_isolated_low_prob_does_not_nucleate(self):
        """核心断言:不含 seed 的 grow 连通域整块丢弃。
        变异'全保留 grow'必须让本测试变红——那是与已死实验(全局降阈值)
        的唯一区别。"""
        s = _dual_seed_soft()
        out = hysteresis_binarize(s, 0.5, 0.3) != 0
        assert not out[0:3, 18:22].any(), "孤立低概率块不应成核"

    def test_seed_region_expands(self):
        s = np.zeros((20, 20), np.float32)
        s[6:14, 6:14] = 0.4
        s[8:12, 8:12] = 0.9
        seed_only = hysteresis_binarize(s, 0.5, 0.5) != 0
        grown = hysteresis_binarize(s, 0.5, 0.3) != 0
        assert seed_only.sum() == 16
        assert grown.sum() == 64
        assert np.all(seed_only <= grown)

    def test_dual_seed_bridge_stays_two_instances(self):
        """Opus P1-2 RED 用例:低阈值桥连两核,切开后仍是两个实例。
        变异'整块 grow 并给一个 seed'必须让本测试变红。"""
        s = _dual_seed_soft()
        binary = hysteresis_binarize(s, 0.5, 0.3)
        nodules = split_nodules(binary, min_area_ratio=0)
        assert len(nodules) == 2, (
            f"桥连后实例数={len(nodules)}(期望 2)——seed 裁决失效会合并成 1")
        bridge_mid = binary[7:9, 18:22] != 0
        assert not bridge_mid.all(), "桥接带应被切开,不能整条保留"

    def test_contact_boundary_sentinel_keeps_outer_contour(self):
        """🔴 哨兵回归锁(Gate2 2026-09-16 P1-3):多 seed 域的**外轮廓**不得被
        当接触面删光。

        `minimum_filter` 前把 owner 的 0 换成 int32 最大值的哨兵,防的是"域外
        0 把 mn 拉到 0 → 每个 grow 像素都满足 mx!=mn → 整域外轮廓被误删"。
        审查实测该变异(删哨兵)在窄桥 fixture 上零差异,但在 **grow 域大于
        seed 域(有真正外轮廓)**的场景误删 32px(80→48)而全套测试仍绿。
        本测试用外环 grow 包双 seed 的构型钉死它:变异后外轮廓整圈被吃,
        前景跌破下限即红。
        """
        s = np.zeros((20, 20), np.float32)
        s[5:15, 5:15] = 0.4     # 外环 grow(无 seed 的外轮廓在此)
        s[8:12, 8] = 0.9        # 左 seed 竖条
        s[8:12, 12] = 0.9       # 右 seed 竖条
        binary = hysteresis_binarize(s, 0.5, 0.3)
        fg = binary != 0
        # 正确行为:seed 全保留 + 外轮廓大部分保留。变异(删哨兵)后外轮廓
        # 整圈被删,前景跌到 48,断言红。
        assert fg.sum() >= 72 + 8, (
            f"前景={fg.sum()}:两条 seed 竖条(4×4×2=32)必须完整保留,外轮廓"
            "被哨兵缺失误删时前景跌破此下限(实测变异值 48)")
        assert fg[8:12, 8].all(), "左 seed 竖条完整保留"
        assert fg[8:12, 12].all(), "右 seed 竖条完整保留"

    def test_single_pixel_seed_not_eroded_by_boundary(self):
        """🔴 1px 核不变量(Gate2 2026-09-16 P2-4):接触面判据显式排除 seed,
        核再窄也不被削空。N5 变异(boundary 不排除 seed)在两核相距 24px 的
        旧 fixture 上逐位零差异,本测试把两核贴到接触面上钉死该不变量。"""
        s = np.zeros((5, 9), np.float32)
        s[2, 2] = 0.9   # 左核 1px
        s[2, 6] = 0.9   # 右核 1px
        s[1:4, 1:8] = np.maximum(s[1:4, 1:8], 0.4)  # grow 覆盖两核
        binary = hysteresis_binarize(s, 0.5, 0.3)
        assert binary[2, 2] != 0, "1px 左核必须保留"
        assert binary[2, 6] != 0, "1px 右核必须保留"

    def test_3d_isolated_and_identity(self):
        s2 = _dual_seed_soft()
        s3 = np.stack([np.zeros_like(s2), s2, np.zeros_like(s2)])
        ident = (s3 > 0.5).astype(np.uint8) * 255
        assert np.array_equal(hysteresis_binarize(s3, 0.5, 0.5), ident)
        out = hysteresis_binarize(s3, 0.5, 0.3)
        assert out.shape == s3.shape
        assert not (out[1, 0:3, 18:22] != 0).any()
        nodules = split_nodules(out, min_area_ratio=0, close_gap_frames=0)
        assert len(nodules) == 2

    def test_bad_ndim_raises(self):
        with pytest.raises(ValueError, match="2D"):
            hysteresis_binarize(np.zeros(8), 0.5, 0.3)


class TestStaticShrinkZ2:
    """Z2 实例级收缩(2026-09-22,新架构线零训练第二波,默认关)。

    设计规格 = docs/新架构线_零训练Z1Z2设计.md Z2 节(预注册):
    ① 默认关逐位零回归;② 收缩语义:bbox 不外扩、面积单调不增;
    ③ 负例:腐蚀到空的实例不得消失(空实例=丢检测=AP 负涨);
    ④ K 越预注册域 {1,2} fail-loud;⑤ 动态(3D)路径零触及。
    env 哨兵隔离由 tests/conftest.py 的 autouse fixture 统一删除,
    本类用 monkeypatch 自设(同 TestMinAreaEnv 模式)。
    """

    def test_default_off_bitwise_identical(self, monkeypatch):
        monkeypatch.delenv("AICOMP_STATIC_SHRINK", raising=False)
        m = np.zeros((200, 200), np.uint8)
        m[20:40, 30:50] = 255
        m[100:106, 100:106] = 255
        base = split_nodules(m, min_area_ratio=0)
        off = split_nodules(m, min_area_ratio=0)
        assert [n["area"] for n in off] == [n["area"] for n in base]
        assert [n["area"] for n in off] == [400, 36]

    def test_shrink_k1_reduces_area_keeps_count(self, monkeypatch):
        monkeypatch.setenv("AICOMP_STATIC_SHRINK", "1")
        m = np.zeros((200, 200), np.uint8)
        m[20:40, 30:50] = 255
        m[100:106, 100:106] = 255
        nodules = split_nodules(m, min_area_ratio=0)
        assert [n["area"] for n in nodules] == [324, 16]  # 18×18 / 4×4

    def test_shrink_k2(self, monkeypatch):
        monkeypatch.setenv("AICOMP_STATIC_SHRINK", "1")
        monkeypatch.setenv("AICOMP_STATIC_SHRINK_K", "2")
        m = np.zeros((200, 200), np.uint8)
        m[20:40, 30:50] = 255
        m[100:106, 100:106] = 255
        nodules = split_nodules(m, min_area_ratio=0)
        assert [n["area"] for n in nodules] == [256, 4]  # 16×16 / 2×2

    def test_bbox_monotone_no_expand(self, monkeypatch):
        """收缩语义 2:收缩后 bbox ⊆ 原 bbox(外扩=方向反了,变成膨胀)。"""
        monkeypatch.setenv("AICOMP_STATIC_SHRINK", "1")
        m = np.zeros((200, 200), np.uint8)
        m[20:40, 30:50] = 255
        shrunk = split_nodules(m, min_area_ratio=0)[0]
        monkeypatch.setenv("AICOMP_STATIC_SHRINK", "0")
        orig = split_nodules(m, min_area_ratio=0)[0]
        x0s, y0s, ws, hs = shrunk["bbox"]
        x0o, y0o, wo, ho = orig["bbox"]
        assert x0s >= x0o and y0s >= y0o, "左/上角点不得外移"
        assert x0s + ws <= x0o + wo and y0s + hs <= y0o + ho, "右/下边界不得外移"

    def test_one_pixel_instance_not_eroded_to_empty(self, monkeypatch):
        """🔴 收缩语义 3 负例(设计文档测试规格 3):1px 实例腐蚀后为空
        必须回退保留实例——空实例=丢检测=AP 负涨,比偏大更糟。"""
        monkeypatch.setenv("AICOMP_STATIC_SHRINK", "1")
        m = np.zeros((200, 200), np.uint8)
        m[50, 50] = 255
        m[100:130, 100:130] = 255
        nodules = split_nodules(m, min_area_ratio=0)
        areas = sorted(n["area"] for n in nodules)
        assert areas == [1, 784]  # 1px 原样保留,30×30 → 28×28

    def test_garbage_env_falls_back_off(self, monkeypatch):
        monkeypatch.setenv("AICOMP_STATIC_SHRINK", "garbage")
        m = np.zeros((200, 200), np.uint8)
        m[20:40, 30:50] = 255
        nodules = split_nodules(m, min_area_ratio=0)
        assert [n["area"] for n in nodules] == [400]  # 逐位=关

    @pytest.mark.parametrize("bad_k", ["0", "3", "-1", "10"])
    def test_k_out_of_domain_fail_loud(self, monkeypatch, bad_k):
        """K 域 {1,2} 是预注册扫参域(设计文档 Z2 算法规格 2),越域
        = 临场发明配方,照 tiling tile≠512 先例 raise 不静默钳。"""
        monkeypatch.setenv("AICOMP_STATIC_SHRINK", "1")
        monkeypatch.setenv("AICOMP_STATIC_SHRINK_K", bad_k)
        m = np.zeros((50, 50), np.uint8)
        m[10:30, 10:30] = 255
        with pytest.raises(ValueError, match="SHRINK_K"):
            split_nodules(m, min_area_ratio=0)

    def test_k_garbage_string_defaults_to_1(self, monkeypatch):
        monkeypatch.setenv("AICOMP_STATIC_SHRINK", "1")
        monkeypatch.setenv("AICOMP_STATIC_SHRINK_K", "not-a-number")
        m = np.zeros((200, 200), np.uint8)
        m[20:40, 30:50] = 255
        nodules = split_nodules(m, min_area_ratio=0)
        assert [n["area"] for n in nodules] == [324]  # 回默认 k=1

    def test_3d_zero_touch(self, monkeypatch):
        """设计文档:动态不做(4.7% 占比,3D 收缩语义未定)——3D 输入
        在 shrink 开/关两种 env 下必须逐位一致。"""
        m3 = np.zeros((5, 100, 100), np.uint8)
        m3[1:4, 20:40, 20:40] = 1
        monkeypatch.setenv("AICOMP_STATIC_SHRINK", "1")
        on = split_nodules(m3, min_area_ratio=0)
        monkeypatch.setenv("AICOMP_STATIC_SHRINK", "0")
        off = split_nodules(m3, min_area_ratio=0)
        assert on[0]["area"] == off[0]["area"] == 3 * 400

    def test_shrink_reranks_by_new_area(self, monkeypatch):
        """收缩后面积序重排:两个原本面积悬殊的实例,腐蚀等量像素后
        大者仍居首(面积降序是 _split_2d 出口契约,收缩必须维持不变量)。"""
        monkeypatch.setenv("AICOMP_STATIC_SHRINK", "1")
        m = np.zeros((200, 300), np.uint8)
        m[10:50, 10:50] = 255    # 40×40=1600
        m[60:76, 200:216] = 255  # 16×16=256
        nodules = split_nodules(m, min_area_ratio=0)
        assert [n["area"] for n in nodules] == [38 * 38, 14 * 14]
        assert nodules[0]["area"] >= nodules[1]["area"]

    def test_mask_value_domain_and_full_size(self, monkeypatch):
        """实例 mask 仍须全尺寸、值域 {0,1}(下游 save_binary_mask 契约)。"""
        monkeypatch.setenv("AICOMP_STATIC_SHRINK", "1")
        m = np.zeros((100, 100), np.uint8)
        m[20:50, 20:50] = 255
        n = split_nodules(m, min_area_ratio=0)[0]
        assert n["mask"].shape == m.shape
        assert set(np.unique(n["mask"]).tolist()) <= {0, 1}

    def test_centroid_matches_shrunk_mask(self, monkeypatch):
        """centroid 按收缩后 mask 重算(残留旧 centroid 会让 crop_nodule
        裁偏,形状仍合法查不出来)。"""
        monkeypatch.setenv("AICOMP_STATIC_SHRINK", "1")
        m = np.zeros((100, 100), np.uint8)
        m[20:60, 20:60] = 255
        n = split_nodules(m, min_area_ratio=0)[0]
        ys, xs = np.nonzero(n["mask"])
        assert n["centroid"] == (pytest.approx(xs.mean(), abs=1e-6),
                                 pytest.approx(ys.mean(), abs=1e-6))
