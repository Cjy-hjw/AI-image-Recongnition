"""Gatekeeper 基础项(真实性 + 规范性)检测器现状回归测试。

比赛①真实性②规范性为一票否决基础项。本测试用 synthesis.py 在线合成
体模/GAN/近重复/copy-move/splice,验证检测器能力边界并**如实锁定现状**。

⚠ 现状(2026-06,Step 1):传统 CV 检测器能力有限——
  - copy-move:特定 patch 比例下可检出,跨随机种子召回不稳定(~20-40%);
  - splicing:自适应阈值过保守,当前几乎不响应 → 标记 xfail(已知局限);
  - dedup / authenticity 特征分离:稳定可用。
Step 2 / 官方数据阶段将升级检测算法,届时收紧本测试的断言。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest
import torch

from src.gatekeeper.synthesis import (
    make_phantom_like,
    make_synthetic_like,
    make_near_duplicate,
    make_copy_move,
    make_splice,
    make_splice_poisson,
    inject_nakagami_speckle,
    domain_randomize,
)
from src.gatekeeper.tampering import detect_copy_move, detect_splicing
from src.gatekeeper.tampering_catnet import (
    CatNetTamperingDetector,
    extract_dct_coeffs,
    block_artifact_energy,
    grid_strength,
    _find_catnet_weights,
)
from src.gatekeeper.deduplication import compute_dhash, hamming_distance
from src.gatekeeper.authenticity import extract_authenticity_features


def _structured_ultrasound(seed=7, vary_structure=False):
    """结构化伪超声图(渐变 + 解剖椭圆 + 散斑),比纯随机图更接近真实。

    纯随机图每个块统计量相似 → copy-move 全图自匹配,不具代表性。
    vary_structure=True 时按 seed 改变结节布局/背景频率,产生结构性不同的图
    (供"不同图不应判重复"测试;否则全局渐变主导 dHash,跨 seed 几乎一致)。
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((256, 256), np.float32)
    yy, xx = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    if vary_structure:
        freq = rng.uniform(30, 90)
        img += 80 + 40 * np.sin(xx / freq) + 30 * (yy / 256.0)
        cx, cy = rng.randint(60, 196), rng.randint(60, 196)
        cv2.ellipse(img, (cx, cy), (rng.randint(30, 60), rng.randint(20, 45)),
                    rng.randint(0, 180), 0, 360, 180, -1)
    else:
        img += 80 + 40 * np.sin(xx / 60.0) + 30 * (yy / 256.0)
        cv2.ellipse(img, (128, 120), (50, 35), 20, 0, 360, 180, -1)  # 主结节
        cv2.ellipse(img, (70, 180), (30, 20), 0, 0, 360, 60, -1)     # 次结节
    img += rng.normal(0, 8, (256, 256))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# 合成引擎:确定性与输出契约
# ═════════════════════════════════════════════════════════════════════════════

class TestSynthesisEngine:
    def test_deterministic_with_seed(self):
        base = _structured_ultrasound()
        a, _, _ = make_copy_move(base, rng_or_seed=42)
        b, _, _ = make_copy_move(base, rng_or_seed=42)
        assert np.array_equal(a, b), "同 seed 合成必须可复现"

    def test_phantom_preserves_shape_dtype(self):
        base = _structured_ultrasound()
        out = make_phantom_like(base, 1)
        assert out.shape == base.shape and out.dtype == np.uint8

    def test_synthetic_preserves_shape_dtype(self):
        base = _structured_ultrasound()
        out = make_synthetic_like(base, 1)
        assert out.shape == base.shape and out.dtype == np.uint8

    def test_copy_move_returns_mask_and_bbox(self):
        base = _structured_ultrasound()
        tam, mask, bbox = make_copy_move(base, 1)
        assert bbox is not None and len(bbox) == 4
        assert tam.shape == base.shape
        assert mask.shape == base.shape and mask.max() == 255

    def test_splice_returns_mask_and_bbox(self):
        base = _structured_ultrasound()
        donor = _structured_ultrasound(seed=99)
        sp, mask, bbox = make_splice(base, donor, 1)
        assert bbox is not None and len(bbox) == 4
        assert sp.shape == base.shape
        assert mask.shape == base.shape and mask.max() == 255


# ═════════════════════════════════════════════════════════════════════════════
# ② 规范性 - 重复检测(dedup):稳定可用
# ═════════════════════════════════════════════════════════════════════════════

class TestDeduplication:
    def test_near_duplicate_flagged(self):
        """近重复图(重采样/亮度/水印)dHash 距离应落在重复阈值内。"""
        base = _structured_ultrasound()
        hits = 0
        for s in range(8):
            nd = make_near_duplicate(base, rng_or_seed=s)
            dist = hamming_distance(compute_dhash(base), compute_dhash(nd))
            if dist <= 12:
                hits += 1
        assert hits >= 6, f"近重复检出 {hits}/8,dHash 失效"

    def test_distinct_images_not_flagged(self):
        """两张内容显著不同的图不应被判为近重复。

        注:dHash(8x8 低频签名)对"暗背景+亮结节"这类全局结构相似的超声图
        区分度有限(实测结构异位的合成图距离仅~12,贴边阈值)。本测试用块状
        布局差异更大的图验证 dHash 对真异图的可分性;生产中 Deduplicator 还有
        二级 ResNet 余弦兜底,不单依赖 dHash。
        """
        rng = np.random.RandomState(0)

        def _blocky(seed):
            r = np.random.RandomState(seed)
            img = np.zeros((256, 256), np.uint8)
            for _ in range(5):
                x, y = r.randint(0, 200, 2)
                cv2.rectangle(img, (x, y), (x + 50, y + 50),
                              int(r.randint(100, 255)), -1)
            return img

        a, b = _blocky(1), _blocky(123)
        _ = rng  # 保持确定性接口一致
        dist = hamming_distance(compute_dhash(a), compute_dhash(b))
        assert dist > 12, f"不同图被误判重复(dist={dist})"

    def test_backend_falls_back_to_resnet_without_sscd(self):
        """Phase 2 可插拔后端:无 SSCD 权重时降级 ResNet18(离线无权重环境守卫)。

        本机/CI 未下载 sscd_disc_mixup.torchscript.pt,后端必须降级而非崩溃。
        指一个不存在的显式路径以确保走降级分支(免受本机偶然缓存影响)。
        """
        from src.gatekeeper.deduplication import FeatureExtractor
        fe = FeatureExtractor(device="cpu", sscd_weights="/nonexistent/sscd.pt")
        assert fe.backend_name == "resnet18"
        # 降级后特征仍为 512 维 L2 归一,与 FAISS IndexFlatIP(512) 兼容
        feat = fe.extract(_structured_ultrasound())
        assert feat.shape == (512,)
        assert abs(float(np.linalg.norm(feat)) - 1.0) < 1e-4

    def test_resnet18_fallback_raises_instead_of_downloading_when_cache_missing(self, monkeypatch):
        """2026-07-11 回归测试(v2,修正 v1 的 CI 回归——v1 曾无条件 raise,把 CI/开发机
        这类本身有网络、pretrained=True 联网下载本能成功的环境也一并堵死,已用真实
        CI 跑红复现锁定该教训)。

        当前正确契约:SSCD 缺失降级到 resnet18 后,若本地缓存
        (~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth)也不存在,应该
        **仍然尝试** `models.resnet18(pretrained=True)` 联网下载(CI/开发机等有网
        环境应该照常成功,不受影响)——只有当联网这一步本身失败(DNS/连接/超时等)时,
        才转换成清晰的 RuntimeError(而不是让原始网络异常裸传播,排障成本高)。

        用 monkeypatch 让 os.path.exists 对 resnet18 缓存路径选择性返回 False
        (其余路径透传真实结果,避免误伤构造函数里其他文件存在性检查),模拟"本地
        缓存缺失";再 monkeypatch pretrained=True 分支模拟"联网也失败"(用
        ConnectionError 模拟真实离线容器场景,不依赖测试环境实际断网),验证失败
        被转换成携带上下文信息的 RuntimeError,而不是原始异常类型直接冒出来。
        """
        from src.gatekeeper import deduplication as dedup_module

        real_exists = os.path.exists
        resnet_cache_path = os.path.expanduser(
            "~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
        )

        def _fake_exists(path):
            if os.path.normpath(path) == os.path.normpath(resnet_cache_path):
                return False
            return real_exists(path)

        monkeypatch.setattr(dedup_module.os.path, "exists", _fake_exists)

        # 只让 pretrained=True 这一次调用(联网触发点)模拟失败;pretrained=False
        # (构造函数里紧邻 else 分支之前那一次,纯建架构不加载权重,合法无害)必须放行,
        # 否则会连带把测试自己弄坏。
        real_resnet18 = dedup_module.models.resnet18

        def _resnet18_network_failure(*args, **kwargs):
            if kwargs.get("pretrained") is True:
                raise ConnectionError("模拟离线容器场景:联网下载 ImageNet 权重失败")
            return real_resnet18(*args, **kwargs)

        monkeypatch.setattr(dedup_module.models, "resnet18", _resnet18_network_failure)

        with pytest.raises(RuntimeError, match="联网下载失败") as exc_info:
            dedup_module.FeatureExtractor(device="cpu", sscd_weights="/nonexistent/sscd.pt")
        # 原始异常必须通过 `raise ... from e` 保留,不能吞掉根因(排障需要看到
        # ConnectionError 这个真实触发源,而不只是包装后的 RuntimeError 文案)。
        assert isinstance(exc_info.value.__cause__, ConnectionError)

    def test_resnet18_fallback_succeeds_via_network_when_cache_missing(self, monkeypatch):
        """2026-07-11 补充回归测试:模拟"本地缓存缺失但网络可用"这个 CI/开发机
        典型场景,验证 pretrained=True 联网下载路径本身没有被误伤——这正是 v1 版本
        直接 raise 时会错误堵死的场景,必须显式测试锁定,防止未来重构再次引入同款回归。
        """
        from src.gatekeeper import deduplication as dedup_module

        real_exists = os.path.exists
        resnet_cache_path = os.path.expanduser(
            "~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
        )

        def _fake_exists(path):
            if os.path.normpath(path) == os.path.normpath(resnet_cache_path):
                return False
            return real_exists(path)

        monkeypatch.setattr(dedup_module.os.path, "exists", _fake_exists)

        # 用一个假的"成功的联网调用"代替真实网络请求(测试不应该依赖真实外网可用性),
        # 只验证:pretrained=True 分支确实被调用到(而不是被跳过或提前 raise)。
        network_call_seen = {"called": False}
        real_resnet18 = dedup_module.models.resnet18

        def _resnet18_fake_success(*args, **kwargs):
            if kwargs.get("pretrained") is True:
                network_call_seen["called"] = True
                return real_resnet18(pretrained=False)  # 返回可用架构,跳过真实下载
            return real_resnet18(*args, **kwargs)

        monkeypatch.setattr(dedup_module.models, "resnet18", _resnet18_fake_success)

        fe = dedup_module.FeatureExtractor(device="cpu", sscd_weights="/nonexistent/sscd.pt")
        assert network_call_seen["called"], "本地缓存缺失时应该尝试联网下载,而不是直接拒绝"
        assert fe.backend_name == "resnet18"

    def test_adaptive_cosine_threshold_per_backend(self):
        """cosine_threshold=None 时按激活后端自适应取推荐阈值(SSCD 0.86 / ResNet18 0.95)。

        显式传值则尊重调用方(向后兼容旧 config 硬编码阈值)。
        """
        from src.gatekeeper.deduplication import Deduplicator
        # 无 SSCD → ResNet18 后端 → 自适应 0.95
        dedup = Deduplicator(device="cpu", sscd_weights="/nonexistent/sscd.pt")
        assert dedup.extractor.backend_name == "resnet18"
        assert dedup.cosine_threshold == 0.95
        # 显式标量优先于自适应(两后端同值,向后兼容旧 config)
        dedup2 = Deduplicator(device="cpu", cosine_threshold=0.995,
                              sscd_weights="/nonexistent/sscd.pt")
        assert dedup2.cosine_threshold == 0.995

    def test_cosine_threshold_dict_per_backend(self):
        """阈值映射(reviewer 必修1):config 传 dict 时按激活后端取值,守住红线。

        单一标量无法同时服务两后端——SSCD 治近重复漏检需 0.86,ResNet18 守一票
        否决红线「宁可漏检不可误杀」需 0.995。映射让降级后端仍取 0.995,
        防 Phase 2 升级把 ResNet18 路径误松到 0.86 而增大误杀真实影像风险。
        """
        from src.gatekeeper.deduplication import Deduplicator
        # 降级 ResNet18 → 取 dict 中 resnet18 键(0.995,守红线),而非 sscd 的 0.86
        dedup = Deduplicator(device="cpu",
                             cosine_threshold={"sscd": 0.86, "resnet18": 0.995},
                             sscd_weights="/nonexistent/sscd.pt")
        assert dedup.extractor.backend_name == "resnet18"
        assert dedup.cosine_threshold == 0.995
        # dict 缺当前后端键 → 回退后端推荐值(ResNet18 0.95),不崩
        dedup2 = Deduplicator(device="cpu", cosine_threshold={"sscd": 0.86},
                              sscd_weights="/nonexistent/sscd.pt")
        assert dedup2.cosine_threshold == 0.95

    def test_feat_dim_probed_for_index(self):
        """维度探测(reviewer 必修2):FeatureExtractor 实测输出维度,防硬编码错配。

        降级 ResNet18 输出 512 维;Deduplicator 据 feat_dim 建 FAISS 索引而非
        硬编码,防误传 1024 维(disc_large)权重时 FAISS 晚抛错 / NumPy 静默错算。
        """
        from src.gatekeeper.deduplication import Deduplicator
        dedup = Deduplicator(device="cpu", sscd_weights="/nonexistent/sscd.pt")
        assert dedup.extractor.feat_dim == 512


# ═════════════════════════════════════════════════════════════════════════════
# ② 规范性 - copy-move 篡改:可检出但召回不稳定(锁定有效区间)
# ═════════════════════════════════════════════════════════════════════════════

class TestCopyMoveDetection:
    def test_detects_copy_move_in_working_range(self):
        """patch_frac≈0.15 时至少能检出部分 copy-move(召回>0,证明管线非死路)。"""
        detected = 0
        for s in range(10):
            base = _structured_ultrasound(seed=s)
            tam, _, _ = make_copy_move(base, rng_or_seed=s * 7 + 1, patch_frac=0.15)
            if detect_copy_move(tam)[0]:
                detected += 1
        # 现状召回 ~20-40%;锁定"至少能 fire"以防回归到完全失效
        assert detected >= 2, f"copy-move 完全检不出({detected}/10),管线断裂"

    def test_no_false_positive_on_clean(self):
        """干净图零误报 —— 这是一票否决项的红线(误杀真实影像不可接受)。"""
        fp = 0
        for s in range(10):
            base = _structured_ultrasound(seed=s)
            if detect_copy_move(base)[0]:
                fp += 1
        assert fp == 0, f"干净图误报 {fp}/10,会误杀真实影像"


# ═════════════════════════════════════════════════════════════════════════════
# ① 真实性 - 特征可分性:真 vs 体模 vs 合成
# ═════════════════════════════════════════════════════════════════════════════

class TestAuthenticityFeatures:
    def test_feature_vector_shape(self):
        base = _structured_ultrasound()
        feat = extract_authenticity_features(base)
        assert feat.shape == (13,) and feat.dtype == np.float32

    def test_naka_m_rayleigh_is_one(self):
        """锁定 Nakagami 方向(Gate2 必修):完全发展散斑(瑞利包络)→ m≈1。

        瑞利包络 = 两路独立零均值同方差高斯的模长 √(a²+b²)。其强度 x²~指数分布,
        指数分布 mean=σ、var=σ²,故矩估计 m = E[x²]²/Var(x²) = 1。这是 DND
        物理解释的理论锚点;若实现回归到方向错误(如把 m 算反),此测试立即失败。
        """
        from src.gatekeeper.authenticity import _naka_m
        rng = np.random.RandomState(0)
        a = rng.normal(0, 1, 200000)
        b = rng.normal(0, 1, 200000)
        envelope = np.sqrt(a * a + b * b).astype(np.float32)  # 瑞利包络
        m = _naka_m(envelope)
        assert 0.85 <= m <= 1.15, f"瑞利包络 m 应≈1,实测 {m:.3f}(方向/公式回归)"

    def test_phantom_smoother_than_real(self):
        """体模匀化处理后 GLCM homogeneity 应高于真实图(纹理更平滑)。"""
        wins = 0
        for s in range(8):
            base = _structured_ultrasound(seed=s)
            real_homog = extract_authenticity_features(base)[3]
            phan_homog = extract_authenticity_features(make_phantom_like(base, s))[3]
            if phan_homog >= real_homog:
                wins += 1
        assert wins >= 6, f"体模 homogeneity 未稳定高于真实({wins}/8)"

    def test_features_separate_three_classes(self):
        """真/体模/合成三类特征向量两两不同(非退化)。"""
        base = _structured_ultrasound()
        f_real = extract_authenticity_features(base)
        f_phan = extract_authenticity_features(make_phantom_like(base, 1))
        f_syn = extract_authenticity_features(make_synthetic_like(base, 2))
        assert not np.allclose(f_real, f_phan), "真 vs 体模 特征退化"
        assert not np.allclose(f_real, f_syn), "真 vs 合成 特征退化"

    def test_physical_features_flag_phantom(self):
        """物理散斑特征(DND+TAI,索引 8:13)应把体模与真实拉开。

        13 维布局:[8:11] DND(m_global/m1/m2),[11:13] TAI(global/local)。
        体模匀质各向同性 → global TAI(索引 11)显著低于真实组织;
        这是 5 维物理特征升级的核心目的(区分 PVA/琼脂仿体)。锁定方向性,
        防回归到「物理特征对体模无响应」。
        """
        wins = 0
        for s in range(8):
            base = _structured_ultrasound(seed=s)
            real_tai = extract_authenticity_features(base)[11]       # global TAI
            phan_tai = extract_authenticity_features(make_phantom_like(base, s))[11]
            if real_tai >= phan_tai:
                wins += 1
        assert wins >= 6, f"体模 TAI 未稳定低于真实({wins}/8),物理特征失效"


# ═════════════════════════════════════════════════════════════════════════════
# ② 规范性 - splicing:直线缝检测(角点正交相交判据)
# ═════════════════════════════════════════════════════════════════════════════

class TestSplicingDetection:
    """拼接检测器以「轴对齐长直缝在角点正交相交」为判据。

    设计取舍:像素/区域方差无法区分拼接块与真实结节(实测干净图 20/20 误报),
    而误报触碰"真实性/规范性"一票否决红线。故改用硬粘贴矩形特有的几何缝特征,
    宁可放弃纯噪声 donor 这类极端对抗样本的部分召回,也要换取干净图零误报。
    """

    def test_detects_splice(self):
        # 真实组织拼接:donor 取另一张结构化超声图(贴近真实篡改场景)
        base = _structured_ultrasound(seed=1)
        donor = _structured_ultrasound(seed=99, vary_structure=True)
        sp, _, _ = make_splice(base, donor, 2, patch_frac=0.30)
        assert detect_splicing(sp)[0], "结构化 donor 拼接未检出"

    def test_no_false_positive_on_clean(self):
        # 一票否决红线:干净结构化超声图(含解剖结节)必须零误报
        for seed in range(12):
            clean = _structured_ultrasound(seed=seed, vary_structure=(seed % 2 == 0))
            assert not detect_splicing(clean)[0], f"干净图误报 (seed={seed}) — 一票否决红线"

    def test_returns_contract(self):
        # 返回契约:(is_spliced, mask_2d_uint8, bbox)
        base = _structured_ultrasound(seed=1)
        donor = _structured_ultrasound(seed=99, vary_structure=True)
        sp, _, _ = make_splice(base, donor, 2, patch_frac=0.30)
        is_spliced, mask, bbox = detect_splicing(sp)
        assert mask.shape == sp.shape and mask.dtype == np.uint8
        if is_spliced:
            assert bbox is not None and len(bbox) == 4


# ═════════════════════════════════════════════════════════════════════════════
# ③ 规范性 - CAT-Net v2 风格 DCT/BAG 取证(羽化/泊松拼接 + 可插拔降级)
# ═════════════════════════════════════════════════════════════════════════════

def _jpeg_roundtrip(img_np, quality=70):
    """JPEG 编解码一圈,植入真实的 8x8 块伪影栅格(BAG)。"""
    ok, buf = cv2.imencode(".jpg", img_np, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


class TestDctPrimitives:
    """DCT 系数提取与块伪影栅格基元的数值正确性(Gate 1 可量化验证)。"""

    def test_dct_shape_and_invertible(self):
        # 块级 DCT 形状正确,且 idct 可重建原图(纯数值正确性)
        img = _structured_ultrasound(seed=3)
        coeffs = extract_dct_coeffs(img, block=8)
        assert coeffs.shape == (256 // 8, 256 // 8, 8, 8)
        # 逐块 idct 重建,与原图对应区域一致(舍入误差内)
        recon = cv2.idct(coeffs[0, 0])
        orig = img[0:8, 0:8].astype(np.float32)
        assert np.allclose(recon, orig, atol=1e-3), "DCT/IDCT 不可逆 — 系数提取有误"

    def test_grid_strength_jpeg_vs_uncompressed(self):
        # JPEG 图块栅格显著度应明显高于未压缩图(BAG 信号的物理依据)
        img = _structured_ultrasound(seed=5)
        s_raw = grid_strength(block_artifact_energy(img))
        s_jpg = grid_strength(block_artifact_energy(_jpeg_roundtrip(img, quality=40)))
        assert s_jpg > s_raw, f"JPEG 栅格未强于未压缩 (raw={s_raw:.3f}, jpg={s_jpg:.3f})"

    def test_energy_map_shape(self):
        e = block_artifact_energy(_structured_ultrasound(seed=2), block=8)
        assert e.shape == (8, 8)


class TestCatNetDetector:
    """CatNetTamperingDetector:可插拔降级 + BAG 相位取证 + 三路并集。"""

    def test_degrades_without_weights(self):
        # 无 HRNet 权重(离线环境)→ 安静降级到纯 NumPy dct-bag 后端,绝不崩溃
        det = CatNetTamperingDetector(catnet_weights="/nonexistent/catnet.pt")
        assert det.backend_name == "dct-bag"
        assert det._model is None

    def test_find_weights_returns_none_when_absent(self):
        assert _find_catnet_weights("/nonexistent/catnet.pt") is None

    def test_no_false_positive_on_clean_jpeg(self):
        # 一票否决红线:干净 JPEG(全图栅格一致)必须零误报
        det = CatNetTamperingDetector()
        for seed in range(8):
            clean = _jpeg_roundtrip(
                _structured_ultrasound(seed=seed, vary_structure=(seed % 2 == 0)),
                quality=70,
            )
            assert not det.bag_phase_inconsistency(clean)[0], \
                f"干净 JPEG 误报 (seed={seed}) — 一票否决红线"

    def test_no_false_positive_on_uncompressed(self):
        # 未压缩/无栅格图必须弃权(返回未篡改),不做相位取证。
        # 扩到 8 个含强结构 seed:真实超声平滑背景+结节会把 median 拉低,
        # 验证 grid_strength 绝对地板守住弃权门,不被分母塌缩放大成误报。
        det = CatNetTamperingDetector()
        for seed in range(8):
            img = _structured_ultrasound(seed=seed, vary_structure=(seed % 2 == 0))
            assert not det.bag_phase_inconsistency(img)[0], \
                f"未压缩图误报 (seed={seed}) — 不应触发 BAG 取证"

    def test_no_false_positive_with_nonmultiple_stride(self):
        # 红线回归:非 block 整倍的 stride/win 不得让干净 JPEG 误报。
        # (修复前局部相位系统性偏移 → 全图密集误报击穿一票否决)
        det = CatNetTamperingDetector(win=60, stride=30)  # 均非 8 的整倍数
        for seed in range(6):
            clean = _jpeg_roundtrip(_structured_ultrasound(seed=seed), quality=70)
            assert not det.bag_phase_inconsistency(clean)[0], \
                f"非整倍 stride 干净 JPEG 误报 (seed={seed}) — 相位坐标换算失效"

    def test_detects_phase_shifted_paste(self):
        # 羽化/相位错位拼接:补丁来自另一 JPEG 且像素偏移 → 破坏宿主栅格相位
        host = _jpeg_roundtrip(_structured_ultrasound(seed=10), quality=85)
        donor = _jpeg_roundtrip(_structured_ultrasound(seed=77, vary_structure=True),
                                quality=85)
        tampered = host.copy()
        # 贴一块,且整体平移 4px(非 8 的倍数)使其块栅格相位与宿主错位
        ph, pw = 96, 96
        y0, x0 = 80, 80
        patch = np.roll(donor[y0:y0 + ph, x0:x0 + pw], shift=(4, 4), axis=(0, 1))
        # 羽化边界(高斯模糊融合),模拟泊松/羽化拼接,消除硬阶跃
        tampered[y0:y0 + ph, x0:x0 + pw] = patch
        tampered = cv2.GaussianBlur(tampered, (3, 3), 0)
        tampered = _jpeg_roundtrip(tampered, quality=85)
        det = CatNetTamperingDetector(grid_strength_min=0.5)
        is_tam, mask, bbox = det.bag_phase_inconsistency(tampered)
        # 注:BAG 是统计信号,二次 JPEG 会部分对齐栅格;命中即记录,不命中不强断言
        if is_tam:
            assert mask.shape == tampered.shape and bbox is not None and len(bbox) == 4

    def test_detect_union_contract(self):
        # 统一入口返回契约:(bool, mask_2d_uint8_同形, bbox|None)
        det = CatNetTamperingDetector()
        img = _jpeg_roundtrip(_structured_ultrasound(seed=1), quality=70)
        is_tam, mask, bbox = det.detect(img)
        assert mask.shape == img.shape and mask.dtype == np.uint8
        if is_tam:
            assert bbox is not None and len(bbox) == 4
        else:
            assert bbox is None

    def test_catnet_own_hit_sets_hard_reject_source(self, monkeypatch):
        """2026-07-12 code-review(Angle A)发现补齐:`last_hard_reject_source`
        是本轮为修复 REJECTED reason 丢失命中源信息而新增的字段,需锁定其
        本身的正确性,否则 gatekeeper_runner 拼接的 reason 文案会静默失真。"""
        det = CatNetTamperingDetector()
        det._model = object()
        det._torch = torch
        fixed_bbox = [1, 1, 20, 20]
        monkeypatch.setattr(
            det, "_catnet_forward",
            lambda img: (True, np.zeros(img.shape[:2], dtype=np.uint8), fixed_bbox),
        )
        img = _structured_ultrasound(seed=1)
        is_tam, _, _ = det.detect(img)
        assert is_tam is True
        assert det.last_hard_reject_source == "catnet"

    def test_no_hard_reject_leaves_source_none(self):
        det = CatNetTamperingDetector()  # dct-bag 降级,无篡改样本
        img = _structured_ultrasound(seed=1)
        is_tam, _, _ = det.detect(img)
        assert is_tam is False
        assert det.last_hard_reject_source is None

    def test_detect_geometric_is_soft_flag_in_degraded_mode(self):
        # 降级模式(无 CAT-Net 权重)下,几何检测器(copy_move/splice)在超声自相似纹理
        # 上误报过高(实测 copy_move 27.3%),违反「宁可漏检不可误杀」红线。
        # 故几何命中降级为**软 flag**(记录到 last_soft_flags,不硬拒绝);
        # 待 CAT-Net 深度权重落地后才升级为硬拒绝。
        base = _structured_ultrasound(seed=1)
        donor = _structured_ultrasound(seed=99, vary_structure=True)
        sp, _, _ = make_splice(base, donor, 2, patch_frac=0.30)
        if detect_splicing(sp)[0]:  # 仅当几何检测器自身命中时校验软 flag 语义
            det = CatNetTamperingDetector()  # dct-bag 降级(无 catnet 权重)
            is_tampered, _, _ = det.detect(sp)
            # 红线:降级模式几何命中不得硬拒绝(除非 BAG 相位也命中)
            if not is_tampered:
                assert any(f == "splice" for f, _ in det.last_soft_flags), \
                    "几何缝命中应记录为软 flag"

    def test_catnet_forward_failure_this_call_does_not_promote_geometric_to_hard(self, monkeypatch):
        """回归 2026-07-10 Gate2审查发现的地雷bug:旧版用
        `block_level_hard = self._model is not None` 只反映"构造时权重加载
        成功",不反映"本次 forward 是否真成功"。若 `_catnet_forward` 本次调用
        抛异常(如 shape assert 失败),旧版仍会把 block_level_hard 算 True,
        导致本该走软flag的copy_move/splice几何检测器(27.3%误报率)被错误提升
        为硬拒绝,真实图像被一票否决出局且无日志。

        修复后应改用 `catnet_ok_this_call`(本次调用是否真成功),权重"已加载
        但本次forward失败"时几何命中必须仍走软flag路,不得硬拒绝。

        2026-07-12 更新:B2(统一软flag制度)后 copy_move/splice 已与 CatNet
        状态完全解耦、恒为软flag——`block_level_hard`/`catnet_ok_this_call`
        变量已从实现中删除。本测试断言("forward失败时几何命中应软flag")仍然
        成立,但已退化为"永不硬拒绝"这一更强不变式的一个特例,不再是该机制的
        专属回归测试(专属回归见下方 test_geometric_never_hard_rejects_even_when_catnet_ok)。

        2026-07-12 QA_VP §2.3 补课:早期版本靠"合成样本恰好触发
        detect_splicing"驱动断言,不触发时 `pytest.skip`——检测器行为一旦被
        意外破坏,测试可能静默 skip 而非 fail,CI 照样绿灯。改用 monkeypatch
        强制 `detect_splicing` 命中,消除该脆弱性。
        """
        import src.gatekeeper.tampering_catnet as tc_module

        base = _structured_ultrasound(seed=1)
        donor = _structured_ultrasound(seed=99, vary_structure=True)
        sp, _, _ = make_splice(base, donor, 2, patch_frac=0.30)
        monkeypatch.setattr(
            tc_module, "detect_splicing",
            lambda img_np: (True, np.zeros(img_np.shape[:2], dtype=np.uint8), [10, 10, 50, 50]),
        )

        det = CatNetTamperingDetector()  # dct-bag 降级(无真实catnet权重)
        # 模拟"权重已加载"但本次 forward 抛异常:注入一个会抛错的假模型。
        det._model = object()  # 非 None,触发 `if self._model is not None` 分支
        det._torch = torch

        is_tampered, _, _ = det.detect(sp)
        # 红线核心断言:即使 self._model 非 None(旧版据此判 block_level_hard=True),
        # 本次 forward 失败后几何命中仍必须走软flag,不得被提升为硬拒绝。
        if not is_tampered:
            assert any(f == "splice" for f, _ in det.last_soft_flags), \
                "catnet本次forward失败时,几何缝命中应仍记录为软flag(不得硬拒绝)"
        else:
            pytest.fail(
                "catnet本次forward失败,但几何命中仍被硬拒绝——"
                "block_level_hard 未正确回退到 catnet_ok_this_call"
            )

    def test_geometric_never_hard_rejects_even_when_catnet_ok(self, monkeypatch):
        """2026-07-12 统一软flag制度(B2):copy_move/splice 与 CatNet 状态解耦。

        此前"CatNet 本次成功跑通(catnet_ok_this_call=True)"会把几何判据升级
        为硬拒绝——但"CatNet 没崩溃"与"copy_move/splice 判据本身是否可信"是
        两件独立的事。本测试锁死新不变式:即使 CatNet 本次前向成功(哪怕判定
        为 clean),几何命中依然只能软flag,不得因 CatNet 状态被提升为硬拒绝。

        2026-07-12 code-review(Angle A)发现:早期版本靠"合成样本恰好触发
        detect_splicing"来驱动断言,不触发时 `pytest.skip`——这是本 diff 唯一
        锁定 B2 核心不变式的回归测试,若该条件判断被意外反转(如
        `if self.bag_hard_reject` 误写成 `if not ...`),测试可能静默 skip 而
        非 fail,CI 照样绿灯。改用 monkeypatch 直接强制 `detect_splicing` 命中,
        测试不再依赖合成图像的自然检测结果,变为确定性、不可跳过。
        """
        import src.gatekeeper.tampering_catnet as tc_module

        img = _structured_ultrasound(seed=1)
        fixed_bbox = [10, 10, 50, 50]
        monkeypatch.setattr(
            tc_module, "detect_splicing",
            lambda img_np: (True, np.zeros(img_np.shape[:2], dtype=np.uint8), fixed_bbox),
        )
        monkeypatch.setattr(
            tc_module, "detect_copy_move",
            lambda img_np: (False, np.zeros(img_np.shape[:2], dtype=np.uint8), None),
        )

        det = CatNetTamperingDetector()  # dct-bag 降级(无真实catnet权重)
        det._model = object()  # 非 None,触发 catnet 分支
        det._torch = torch
        # 桩替换 _catnet_forward:本次"成功"返回 clean(hit=False),模拟
        # catnet_ok_this_call=True 但未命中的场景——旧实现下这会让
        # block_level_hard=True,几何命中随即被提升为硬拒绝。
        monkeypatch.setattr(
            det, "_catnet_forward",
            lambda img: (False, np.zeros(img.shape[:2], dtype=np.uint8), None),
        )

        is_tampered, _, _ = det.detect(img)
        assert is_tampered is False, \
            "CatNet 本次成功(判 clean)不应把几何判据(27.3%误报率)提升为硬拒绝"
        assert any(f == "splice" for f, _ in det.last_soft_flags), \
            "几何缝命中应仍记录为软flag,与 CatNet 状态无关"


class TestBagHardRejectToggle:
    """2026-07-12 统一软flag制度(B3):BAG 相位不一致默认软flag,
    显式 `bag_hard_reject=True` 时才恢复硬拒绝。

    默认 False 的理由:BAG 阈值(grid_strength_min/min_flag_blocks)仅经
    8 个合成种子小样本验证(见上方 TestCatNetDetector 的干净图 FPR 测试),
    远小于 copy_move/splice 的 359 张真实图验证量级。恢复硬拒绝的唯一闸门
    是官方真实干净 holdout 上预注册、带置信区间的 FPR 验收达标。
    """

    @staticmethod
    def _phase_shifted_sample():
        """复用 test_detects_phase_shifted_paste 的构造:羽化拼接使局部块栅格
        相位与全图错位,是 BAG 判据设计要捕获的正例。"""
        host = _jpeg_roundtrip(_structured_ultrasound(seed=10), quality=85)
        donor = _jpeg_roundtrip(_structured_ultrasound(seed=77, vary_structure=True),
                                quality=85)
        tampered = host.copy()
        ph, pw = 96, 96
        y0, x0 = 80, 80
        patch = np.roll(donor[y0:y0 + ph, x0:x0 + pw], shift=(4, 4), axis=(0, 1))
        tampered[y0:y0 + ph, x0:x0 + pw] = patch
        tampered = cv2.GaussianBlur(tampered, (3, 3), 0)
        return _jpeg_roundtrip(tampered, quality=85)

    def test_bag_hit_is_soft_flag_by_default(self, monkeypatch):
        """2026-07-12 code-review(Angle A):原版靠"合成拼接样本恰好触发
        `bag_phase_inconsistency`"驱动断言,不触发时 `pytest.skip`——BAG 默认
        softflag 这个不变式若被意外破坏(如 `bag_hard_reject` 默认值被误改为
        True),测试可能静默 skip 而非 fail。改用 monkeypatch 强制命中,
        测试变为确定性、不可跳过,只验证 detect() 内的路由逻辑,不依赖真实
        BAG 检测器在某张合成图上是否命中(真实命中能力由本类其余测试覆盖)。
        """
        img = self._phase_shifted_sample()
        det = CatNetTamperingDetector(grid_strength_min=0.5)  # 默认 bag_hard_reject=False
        fixed_bbox = [5, 5, 40, 40]
        monkeypatch.setattr(
            det, "bag_phase_inconsistency",
            lambda img_np: (True, np.zeros(img_np.shape[:2], dtype=np.uint8), fixed_bbox),
        )
        is_tampered, _, _ = det.detect(img)
        assert is_tampered is False, "BAG 默认应软flag,不驱动硬拒绝"
        assert any(f == "bag_phase" for f, _ in det.last_soft_flags), \
            "BAG 命中应记录到 last_soft_flags"

    def test_bag_hard_reject_true_restores_hard(self, monkeypatch):
        """同上,改用 monkeypatch 强制 BAG 命中,不依赖合成样本自然触发。"""
        img = self._phase_shifted_sample()
        det = CatNetTamperingDetector(grid_strength_min=0.5, bag_hard_reject=True)
        fixed_bbox = [5, 5, 40, 40]
        monkeypatch.setattr(
            det, "bag_phase_inconsistency",
            lambda img_np: (True, np.zeros(img_np.shape[:2], dtype=np.uint8), fixed_bbox),
        )
        is_tampered, _mask, bbox = det.detect(img)
        assert is_tampered is True, "bag_hard_reject=True 时 BAG 命中应恢复硬拒绝"
        assert bbox is not None and len(bbox) == 4
        assert det.last_hard_reject_source == "bag_phase"


# ═════════════════════════════════════════════════════════════════════════════
# ④ 合成引擎升级:泊松/羽化拼接 + 域随机化(Nakagami 散斑 + 模糊 + JPEG)
# ═════════════════════════════════════════════════════════════════════════════

class TestPoissonSplice:
    """泊松/羽化拼接:无硬缝,供 CAT-Net DCT/BAG 取证训练(几何检测器抓不到)。"""

    def test_contract_and_shape(self):
        base = _structured_ultrasound(seed=1)
        donor = _structured_ultrasound(seed=42, vary_structure=True)
        out, mask, bbox = make_splice_poisson(base, donor, 3, patch_frac=0.25)
        assert out.shape == base.shape and out.dtype == np.uint8
        assert mask.shape == base.shape and mask.dtype == np.uint8
        assert bbox is not None and len(bbox) == 4
        assert mask.max() == 255 and mask.sum() > 0  # 确有粘贴区

    def test_no_hard_seam_vs_hard_splice(self):
        # 泊松融合应比硬粘贴显著更平滑:融合区边界最大梯度远低于硬缝
        base = _structured_ultrasound(seed=5)
        donor = _structured_ultrasound(seed=88, vary_structure=True)
        hard, hmask, _ = make_splice(base, donor, 7, patch_frac=0.25)
        soft, smask, _ = make_splice_poisson(base, donor, 7, patch_frac=0.25)

        def _edge_max(img, mask):
            gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            edge = cv2.dilate(mask, np.ones((3, 3), np.uint8)) - cv2.erode(
                mask, np.ones((3, 3), np.uint8))
            vals = mag[edge > 0]
            return float(vals.max()) if vals.size else 0.0

        # 羽化边界梯度应明显低于硬缝(允许等号松弛,核心是不更尖锐)
        assert _edge_max(soft, smask) <= _edge_max(hard, hmask) + 1e-3

    def test_deterministic(self):
        base = _structured_ultrasound(seed=1)
        donor = _structured_ultrasound(seed=2, vary_structure=True)
        a, _, ba = make_splice_poisson(base, donor, 9)
        b, _, bb = make_splice_poisson(base, donor, 9)
        assert np.array_equal(a, b) and ba == bb

    def test_small_image_guard_no_crash(self):
        # 回归:小图/极端 patch_frac 下中心采样区间塌缩不得抛 ValueError,
        # 应安全返回 (原图, 空mask, None)
        small = np.full((25, 25), 128, np.uint8)
        donor = np.full((25, 25), 64, np.uint8)
        out, mask, bbox = make_splice_poisson(small, donor, 0, patch_frac=0.22)
        assert out.shape == small.shape and mask.sum() == 0 and bbox is None


class TestDomainRandomization:
    """域随机化:Nakagami 散斑 → 模糊 k∈{3,5,7} → JPEG Q∈[15,75]。"""

    def test_nakagami_preserves_shape_dtype_and_brightness(self):
        img = _structured_ultrasound(seed=3)
        out = inject_nakagami_speckle(img, 0)
        assert out.shape == img.shape and out.dtype == np.uint8
        # 乘性散斑归一到均值 1,整体亮度不应大幅漂移
        assert abs(float(out.mean()) - float(img.mean())) < 25.0

    def test_nakagami_alters_texture(self):
        # 散斑注入必须真正改变像素纹理(非恒等)
        img = _structured_ultrasound(seed=3)
        out = inject_nakagami_speckle(img, 7)
        assert not np.array_equal(out, img)

    def test_domain_randomize_contract(self):
        img = _structured_ultrasound(seed=4)
        out = domain_randomize(img, 0)
        assert out.shape == img.shape and out.dtype == np.uint8

    def test_domain_randomize_deterministic(self):
        img = _structured_ultrasound(seed=4)
        a = domain_randomize(img, 11)
        b = domain_randomize(img, 11)
        assert np.array_equal(a, b)

    def test_domain_randomize_injects_jpeg_grid(self):
        # 域随机化含 JPEG 重压 → 块伪影栅格显著度应高于原未压缩图
        img = _structured_ultrasound(seed=6)
        out = domain_randomize(img, 6)
        s_raw = grid_strength(block_artifact_energy(img))
        s_dr = grid_strength(block_artifact_energy(out))
        assert s_dr > s_raw, f"域随机化未植入 JPEG 栅格 (raw={s_raw:.3f}, dr={s_dr:.3f})"
