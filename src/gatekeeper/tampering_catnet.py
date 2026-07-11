"""CAT-Net v2 风格的拼接/篡改取证(可插拔 + 降级)。

Phase 3 升级目标:几何长直缝检测器(`tampering.detect_splicing`)只能抓"硬粘贴、
不羽化"的轴对齐矩形缝;对**羽化 / 泊松融合**拼接(边界被平滑掉、无长直阶跃)无能为力。
CAT-Net v2(IJCV 2022, Apache-2.0)用 DCT 系数流追踪 JPEG 8x8 量化栅格的不连续——
拼接块若来自另一张 JPEG(或与宿主栅格相位错位),会破坏宿主图全局一致的 8x8 块伪影
栅格(Block Artifact Grid, BAG),即使 RGB 边界被融合得看不见。

本模块两层能力,沿用 Phase 2「可插拔 + 降级」范式:
  1) **核心可验证组件(现在就做、纯 NumPy/cv2、单测覆盖)**:BAG 相位不一致取证。
     全图估计主导块栅格相位,滑窗找相位错位的区域 → 命中羽化/泊松拼接。
     竞赛交付为 JPG(见 CLAUDE.md 约束),BAG 信号普遍存在;但对**未压缩/无栅格**
     的图必须**弃权**(返回未篡改),守住"真实影像零误报"的一票否决红线。
  2) **深度 CAT-Net(可选,需下载 HRNet 权重)**:权重存在则加载 TorchScript,
     缺失则降级——只跑 BAG + 既有几何检测器,零破坏现有 gatekeeper 测试。

最终 `CatNetTamperingDetector.detect` 的篡改判定 = 深度CAT-Net ∪ BAG相位 ∪ 几何缝。
"""

import os

import cv2
import numpy as np

from src.gatekeeper.tampering import detect_copy_move, detect_splicing

# ── 深度 CAT-Net 权重探测路径(缺失则降级)────────────────────────────────
_CATNET_WEIGHT_CANDIDATES = (
    "checkpoints/catnet_v2.torchscript.pt",
    os.path.expanduser("~/.cache/torch/hub/checkpoints/catnet_v2.torchscript.pt"),
)


def _find_catnet_weights(explicit_path=None):
    """返回第一个存在的 CAT-Net 权重路径;均不存在返回 None。"""
    candidates = [explicit_path] if explicit_path else []
    candidates += list(_CATNET_WEIGHT_CANDIDATES)
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


# ── 核心可验证组件:块级 DCT 系数提取(纯 cv2,供取证/单测)──────────────
def extract_dct_coeffs(img_np, block=8):
    """对非重叠 block×block 网格逐块做 2D-DCT,返回系数张量。

    这是 CAT-Net "DCT 系数流"的可验证基元:JPEG 在 8x8 块上量化,块级 DCT 系数
    是栅格取证的原料。纯 `cv2.dct`,可逆(idct 重建),单测覆盖。

    Args:
        img_np: 单通道灰度图 (H, W),uint8 或 float。
        block: 块边长,默认 8(对齐 JPEG 量化块)。

    Returns:
        coeffs: (n_by, n_bx, block, block) float32 DCT 系数;
                n_by = H // block, n_bx = W // block(右/下不足一块的边缘被丢弃)。
    """
    img_f = img_np.astype(np.float32)
    h, w = img_f.shape
    n_by, n_bx = h // block, w // block
    coeffs = np.zeros((n_by, n_bx, block, block), dtype=np.float32)
    for by in range(n_by):
        for bx in range(n_bx):
            patch = img_f[by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            coeffs[by, bx] = cv2.dct(patch)
    return coeffs


def _cross_diff(img_f):
    """2x2 交叉二阶差分:d(i,j)=|I[i,j]+I[i+1,j+1]-I[i,j+1]-I[i+1,j]|。

    对线性渐变恒为 0(各项抵消),故对真实超声平滑区鲁棒;JPEG 块边界处该量
    统计性偏大——块伪影。返回 (H-1, W-1)。注:差分图索引相对原图偏移 1,故 JPEG
    块边界(原图行 7↔8)落在差分图相位 ≡(block-1) mod block,非 ≡0;但全模块只做
    全图相位 vs 局部相位的**相对比较**,该常数偏移不影响判定。
    """
    a = img_f[:-1, :-1]
    b = img_f[:-1, 1:]
    c = img_f[1:, :-1]
    d = img_f[1:, 1:]
    return np.abs(a + d - b - c)


def block_artifact_energy(img_np, block=8):
    """块伪影栅格能量图:对 8x8 相位 (p,q) 聚合交叉差分均值,返回 (block, block)。

    全图主导块栅格相位 = argmax(E)。JPEG 图该图有尖锐峰(块边界相位),
    未压缩图近似平坦。供 `grid_strength` 与相位不一致定位使用。
    """
    img_f = img_np.astype(np.float32)
    d = _cross_diff(img_f)
    hh, ww = d.shape
    energy = np.zeros((block, block), dtype=np.float64)
    count = np.zeros((block, block), dtype=np.float64)
    # 向量化按相位聚合
    ys = np.arange(hh) % block
    xs = np.arange(ww) % block
    for p in range(block):
        rows = d[ys == p]                       # (?, ww)
        if rows.size == 0:
            continue
        for q in range(block):
            sel = rows[:, xs == q]
            if sel.size:
                energy[p, q] += sel.sum()
                count[p, q] += sel.size
    count[count == 0] = 1.0
    return energy / count


def grid_strength(energy):
    """块栅格显著度 =(峰值 - 中位)/(中位 + 地板)。高 → 图确实有 JPEG 块栅格。

    分母加**绝对能量地板**(median 与 peak·0.05 取大):真实超声大片平滑背景会把
    中位拉到近 0,若仅用 `med+eps` 作分母,局部结节高频会被塌缩的分母放大成虚高显著度,
    可能让**未压缩真实图**越过 BAG 弃权门 → 一票否决红线误报。地板确保无 JPEG 栅格的
    图显著度保持低位、稳稳弃权。
    """
    med = float(np.median(energy))
    peak = float(np.max(energy))
    floor = max(med, peak * 0.05)
    return (peak - med) / (floor + 1e-6)


class CatNetTamperingDetector:
    """拼接/篡改取证检测器:深度 CAT-Net(可选)∪ BAG 相位 ∪ 既有几何检测。

    Args:
        catnet_weights: 显式 TorchScript 权重路径;None 时按候选路径探测。
        grid_strength_min: 全图块栅格显著度门槛,低于此判图无 JPEG 栅格 → BAG 弃权
            (守红线:未压缩/无栅格图不做相位取证,避免真实影像误报)。
        win: 相位不一致滑窗边长(像素)。
        stride: 滑窗步长。
        min_flag_blocks: 命中窗聚类的最小窗数,低于此不判篡改(压随机噪声误报)。

    Attributes:
        backend_name: "catnet"(深度权重在)或 "dct-bag"(纯 NumPy 降级)。
    """

    def __init__(self, catnet_weights=None, grid_strength_min=0.8,
                 win=64, stride=32, min_flag_blocks=3):
        self.grid_strength_min = float(grid_strength_min)
        self.win = int(win)
        self.stride = int(stride)
        self.min_flag_blocks = int(min_flag_blocks)
        self._model = None
        self.backend_name = "dct-bag"
        self.last_soft_flags = []  # 最近一次 detect 的软 flag 命中(降级模式块级疑似)

        weights = _find_catnet_weights(catnet_weights)
        if weights is not None:
            try:
                import torch
                self._model = torch.jit.load(weights, map_location="cpu").eval()
                self._torch = torch
                self.backend_name = "catnet"
            except Exception as e:
                # 权重损坏/torch 缺失 → 安静降级到纯 NumPy BAG,绝不让审计崩溃；
                # 但须打印留痕区分"权重损坏"与"权重原本不存在"——2026-07-10
                # Gate2审查发现此前二者共享同一静默出口，权重损坏这种更严重的
                # 部署事故完全无迹可查（与 deduplication.py SSCD 分支的
                # fail-loud 哲学不一致）。
                print(
                    f"--> [Gatekeeper] CAT-Net 权重存在但加载失败"
                    f"({type(e).__name__}: {e})，降级为纯 NumPy BAG 取证: {weights}",
                    flush=True,
                )
                self._model = None
                self.backend_name = "dct-bag"

    # ── BAG 相位不一致定位(核心可验证组件)──────────────────────────────
    def bag_phase_inconsistency(self, img_np, block=8):
        """滑窗找局部块栅格相位 ≠ 全图主导相位的区域 → 羽化/泊松拼接。

        Returns:
            (is_tampered, mask uint8 (H,W), bbox [x1,y1,x2,y2] | None)
        """
        h, w = img_np.shape[:2]
        if img_np.ndim == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        mask = np.zeros((h, w), dtype=np.uint8)

        g_energy = block_artifact_energy(img_np, block)
        # 红线:全图无显著块栅格 → 弃权(返回未篡改),不对未压缩真实图做取证
        if grid_strength(g_energy) < self.grid_strength_min:
            return False, mask, None
        g_phase = np.unravel_index(int(np.argmax(g_energy)), g_energy.shape)

        flagged = []
        win, stride = self.win, self.stride
        for y in range(0, max(1, h - win + 1), stride):
            for x in range(0, max(1, w - win + 1), stride):
                patch = img_np[y:y + win, x:x + win]
                if patch.shape[0] < block * 2 or patch.shape[1] < block * 2:
                    continue
                e = block_artifact_energy(patch, block)
                # 局部需自身有栅格才比较相位,否则该窗弃权(融合区往往栅格被削弱)
                if grid_strength(e) < self.grid_strength_min:
                    continue
                l_phase = np.unravel_index(int(np.argmax(e)), e.shape)
                # 把局部相位换算回全图坐标系:patch 像素 0 = 全图像素 y,
                # 故局部相位 p 对应全图相位 (p+y)%block。消除窗口原点偏移——
                # 否则非 block 整倍的 stride/win 会让干净 JPEG 局部相位系统性
                # ≠ 全图相位 → 全图误报击穿一票否决红线(默认 win/stride 为 8 倍
                # 数时本式恒等,零行为变更)。
                l_phase = ((l_phase[0] + y) % block, (l_phase[1] + x) % block)
                if l_phase != g_phase:
                    mask[y:y + win, x:x + win] = 255
                    flagged.append((x, y))

        if len(flagged) < self.min_flag_blocks:
            return False, np.zeros_like(mask), None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, np.zeros_like(mask), None
        biggest = max(contours, key=cv2.contourArea)
        tx, ty, tw, th = cv2.boundingRect(biggest)
        return True, mask, [tx, ty, tx + tw, ty + th]

    # ── 深度 CAT-Net 前向(权重在时启用)──────────────────────────────────
    def _catnet_forward(self, img_np):
        """TorchScript CAT-Net 前向 → 篡改概率图二值化。权重缺失永不走到这里。"""
        torch = self._torch
        rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        ten = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            out = self._model(ten)
        prob = out.squeeze().cpu().numpy()
        if prob.ndim == 3:                       # (C,H,W) → 取篡改通道
            assert prob.shape[0] == 2, f"CAT-Net output expected 2 channels, got {prob.shape[0]}"
            prob = prob[1]                        # channel 0=clean, channel 1=tampered
        prob = cv2.resize(prob, (img_np.shape[1], img_np.shape[0]))
        mask = (prob > 0.5).astype(np.uint8) * 255
        if mask.sum() == 0:
            return False, mask, None
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest = max(contours, key=cv2.contourArea)
        tx, ty, tw, th = cv2.boundingRect(biggest)
        return True, mask, [tx, ty, tx + tw, ty + th]

    # ── 统一入口:硬拒绝路 ∪ 软 flag 路 ──────────────────────────────────
    def detect(self, img_np):
        """篡改判定。返回签名向后兼容:(is_tampered, mask, bbox)。

        **硬拒绝路（is_tampered=True，触发 REJECT）**:
            深度CAT-Net(若权重在) ∪ BAG相位不一致。
            二者均有红线保护:CAT-Net 是学习式判据;BAG 有"无 JPEG 栅格则弃权"门。

        **软 flag 路（记录但不拒绝，is_tampered 不受影响）**:
            copy_move / splice 几何块级判据。经全量 holdout 实测(359张真实图):
            copy_move 误报 27.3%、splice 亦在超声自相似纹理上不可靠——经典块匹配
            在超声模态无法兼顾零误报与有效召回(详见 duipai 实验 + AGENTS.md §3)。
            **红线「宁可漏检不可误杀」要求:降级模式下此二路不得硬拒绝真实图**,
            故仅作软 flag 审计输出(stdout);待 CAT-Net 深度权重(catnet_v2)落地后升级为硬拒绝。
            `self.last_soft_flags` 暴露最近一次 detect 的软命中详情(供调用方按需读取
            接入报告;当前 gatekeeper_runner 仅取 is_tampered 放行,软 flag 走 stdout)。

        Returns:
            (is_tampered, mask uint8 (H,W), bbox [x1,y1,x2,y2] | None)
        """
        self.last_soft_flags = []  # 每次 detect 重置:软 flag 命中清单(可审计)

        catnet_ok_this_call = False
        if self._model is not None:
            try:
                hit, mask, bbox = self._catnet_forward(img_np)
                catnet_ok_this_call = True
                if hit:
                    return True, mask, bbox
            except Exception as e:
                # 深度路本次调用出错(如223行shape assert触发)→降级到下方纯
                # NumPy取证,不中断审计。但绝不能让"权重已加载"继续冒充"本次
                # 判定可信"——2026-07-10 Gate2审查发现：旧版用
                # `self._model is not None` 直接判 block_level_hard,会让本该
                # 走软flag的copy_move/splice几何检测器(全量holdout实测硬拒绝
                # 误报27.3%)在catnet本次forward崩溃时被错误提升为硬拒绝,真实
                # 图像被一票否决出局且无任何日志区分"catnet判clean"与"catnet
                # 本次崩溃"。此处显式打印留痕,并用 catnet_ok_this_call(而非
                # self._model是否非None)驱动下方 block_level_hard。
                print(
                    f"--> [Gatekeeper] CAT-Net 本次前向失败"
                    f"({type(e).__name__}: {e})，降级为纯几何取证(不影响BAG判据)",
                    flush=True,
                )

        # 硬拒绝路:BAG 相位(有弃权红线保护)
        hit, mask, bbox = self.bag_phase_inconsistency(img_np)
        if hit:
            return True, mask, bbox

        # 深度 CAT-Net 本次调用真正成功时,块级几何判据可信 → 参与硬拒绝;
        # 否则(未加载 或 本次forward失败)→ 块级判据在超声上误报过高 → 降级为
        # 软 flag,不拒绝。用 catnet_ok_this_call 而非 self._model is not None，
        # 见上方注释。
        block_level_hard = catnet_ok_this_call

        is_cm, cm_mask, cm_bbox = detect_copy_move(img_np)
        if is_cm:
            if block_level_hard:
                return True, cm_mask, cm_bbox
            self.last_soft_flags.append(("copy_move", cm_bbox))

        is_sp, sp_mask, sp_bbox = detect_splicing(img_np)
        if is_sp:
            if block_level_hard:
                return True, sp_mask, sp_bbox
            self.last_soft_flags.append(("splice", sp_bbox))

        if self.last_soft_flags:
            print(f"--> [Gatekeeper 软 flag] 降级模式块级篡改疑似(不拒绝,仅记录): "
                  f"{[f for f, _ in self.last_soft_flags]}", flush=True)

        return False, np.zeros(img_np.shape[:2], dtype=np.uint8), None
