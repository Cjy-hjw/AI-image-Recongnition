"""ROI 版四头(两阶段 ROI 范式 · 波3 模型面,2026-09-25)。

## 与整图四头的关系(scoping §C.1 逐头判定,实读代码定案)

| 头 | 搬运性 | 本模块做法 |
|---|---|---|
| loc | 🟢 **可零训练搬运**(CoordConv 坐标通道是张量 H/W 的函数) | `RoiLocationHead`:结构与 `LocationHead` **逐层同构**(同一 in_channels/hidden/层序),state_dict 键同形 → `load_state_dict` 直接吃现有权重;坐标通道按 **ROI 在原图中的全局位置**生成(见该类 docstring) |
| ete | 🔴 不可搬运(整图 band 语义在 ROI 内消失) | `RoiExtensionHead`:ROI 内 band 加权池化新结构(ROI 域 mask 出带,带语义 = "结节 vs ROI 内腺体") |
| clin_t | 🟡 部分可(global_proj 可搬,scale 语义变) | `RoiClinicalTHead`:scale 通道 `[d_mm, log d_mm]`(物理 mm,`mask_features.physical_diameter_from_mask` 产出),结构同 `ClinicalTHead` |
| tirads | 🔴 不可搬运(概念来自整图级 CBM) | `RoiTiradsHead`:ROI 概念头(第二 CBM,新训)+ 概念 logits 拼接 |

**设计铁律**:每个 ROI 头与整图头**结构同构的部分必须逐层同名同参数**(loc 全部、
T/tirads 的 global_proj/classifier/scale_proj)——这是 warm-start 的载体;
语义不同的部分(ETE 的 band 池化、坐标通道生成方式)才允许新结构。
两处各建一份必然漂移(scoping §A.1.1 #2),故本模块是唯一构建点,训练/推理共用。

## 全局坐标通道(RoiLocationHead 的核心)

`CoordConv2d.forward` 里坐标通道 = `linspace(-1, 1, w/h)`——是**张量 H/W 的
函数**。ROI 特征图 [K,C,h,w] 的 (i,j) cell 对应 ROI 内相对位置;但 loc 头学的是
**绝对位置**(左叶/右叶),坐标必须映射回**原图归一化坐标系**:

    x_orig(j) = (x0 + j_patch_in_roi * scale + roi_w/2) / orig_w  → 再线性到 [-1,1]

其中 ROI 带上下文边距(bbox 外扩 pad_ratio 后的实际裁剪框)。本模块直接吃
`roi_meta` 的 **bbox_xyxy(实际生效裁剪框,含 pad)**与 orig_w/orig_h,生成
原图域 linspace——"ROI = 全图"退化(bbox=(0,0,orig_w,orig_h))时坐标通道与
整图头逐位一致,这是 `test_loc_head_portable` 对拍的数学基础。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import CoordConv2d
from src.models.heads.location import (
    NUM_LEVELS,
    NUM_SIDES,
    compose_location_probs,
)

__all__ = [
    "RoiLocationHead",
    "RoiLocationOutputs",
    "RoiExtensionHead",
    "RoiExtensionOutputs",
    "RoiClinicalTHead",
    "RoiClinicalTOutputs",
    "RoiTiradsHead",
    "RoiTiradsOutputs",
    "global_coord_channels",
    "RoiConceptHead",
    "NUM_ETE_CLASSES",
    "NUM_TIRADS_CLASSES",
    "NUM_CLIN_T_CLASSES",
]

from src.models.label_schema import ETE_CODES, TIRADS_CODES
from src.models.mask_features import boundary_band_from_mask

NUM_ETE_CLASSES = len(ETE_CODES)       # 4
NUM_TIRADS_CLASSES = len(TIRADS_CODES)  # 5
NUM_CLIN_T_CLASSES = 2                  # T ∈ {0,1}

# 与整图头共享的常量(顺序锁见 heads/extension.py / heads/tirads.py 的导入期断言,
# 这里不重复断言,只消费)。
_EPS = 1e-8
_SCALE_LOG_FLOOR = 1e-4


def global_coord_channels(
    bbox_xyxy, orig_w: int, orig_h: int, feat_h: int, feat_w: int,
    device=None, dtype=torch.float32,
) -> tuple:
    """ROI 特征图 cell → **原图归一化坐标系 [-1,1]** 的 (x_ch, y_ch)。

    取值约定 = **端点闭区间 linspace**(与整图头 `CoordConv2d` 的
    `torch.linspace(-1,1,N)` 同一约定:cell 0 落 ROI 左/上边界、cell N-1 落
    右/下边界)。⚠️ 早期版本用"cell 中心采样"((j+0.5)/N),与整图头差半 cell
    ——退化对拍实测 side/level logits 差 6.8e-5/4.5e-5(不是数值噪声,是
    采样约定错位),已改端点式。

    🔴 退化逐位一致的实现细节:直接调 `torch.linspace(u0, u1, N)`(u0/u1 =
    ROI 边界的原图归一化坐标 ×2−1),**不用** `-1 + j*step` 手算——linspace
    内部算术顺序不同会在末位 ulp 分叉,退化对拍(ROI=全图时与
    `CoordConv2d` 的 `linspace(-1,1,N)` 逐位相等)必须走同一库函数。

    Args:
        bbox_xyxy: (x0, y0, x1, y1) 像素半开区间 = **实际生效的裁剪框**
            (含 pad;roi_meta['bbox_xyxy'] 在 ROI 裁剪后是 pad 后的框——数据面
            `RoiDataset.crop_roi` 的 pad 记录在 pad_hw,坐标还原由消费端做)。
        orig_w / orig_h: 原图尺寸(归一化分母)。
        feat_h / feat_w: ROI 特征图空间尺寸。
    Returns:
        (x_ch, y_ch) 各 [1, 1, feat_h, feat_w](broadcast-ready,与 CoordConv2d
        的 cat 语义同形)。
    """
    x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
    # ROI 边界 → 原图归一化 [-1,1](乘 2 减 1)。
    u0, u1 = 2.0 * x0 / float(orig_w) - 1.0, 2.0 * x1 / float(orig_w) - 1.0
    v0, v1 = 2.0 * y0 / float(orig_h) - 1.0, 2.0 * y1 / float(orig_h) - 1.0
    # 与 CoordConv2d 同一函数同一形态(退化对拍逐位一致的载体,勿手算)。
    cx = torch.linspace(u0, u1, feat_w, device=device, dtype=dtype)
    cy = torch.linspace(v0, v1, feat_h, device=device, dtype=dtype)
    x_ch = cx.view(1, 1, 1, feat_w).expand(1, 1, feat_h, feat_w)
    y_ch = cy.view(1, 1, feat_h, 1).expand(1, 1, feat_h, feat_w)
    return x_ch, y_ch


# ── loc ─────────────────────────────────────────────────────────────────────


class RoiLocationOutputs(dict):
    """键与 `LocationOutputs` 同名同义(side/level logits + 7 维分布)。"""


class RoiLocationHead(nn.Module):
    """ROI 版位置头——结构逐层同构 `LocationHead`,state_dict 直接互载。

    前向差异**只有一处**:`CoordConv2d` 的坐标通道不用 `linspace(-1,1,h/w)`
    (那是 ROI 内相对坐标),改用 `global_coord_channels` 按 ROI 在原图中的
    全局位置生成。卷积/BN/pool/FC 的参数名与形状全部与 `LocationHead` 一致:
    `load_state_dict(LocationHead.state_dict())` 零键缺失零多余。

    Args:
        in_channels: 352(CLS_POOLED_DIM,同整图头约束)。
        hidden_channels: 与 LocationHead 默认一致(128),保证搬运可行。
    """

    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.spatial = nn.Sequential(
            CoordConv2d(in_channels, hidden_channels, kernel_size=3),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.side_fc = nn.Linear(hidden_channels, NUM_SIDES)
        self.level_fc = nn.Linear(hidden_channels, NUM_LEVELS)

    def forward(
        self,
        roi_spatial: torch.Tensor,
        bbox_xyxy=None,
        orig_w: int = 0,
        orig_h: int = 0,
    ) -> RoiLocationOutputs:
        """roi_spatial [K,C,h,w] + ROI 几何 → 7 类位置分布。

        bbox/orig 缺席(None/0)时退化为 **ROI 内相对坐标**(linspace 语义)——
        仅供结构冒烟,生产/训练路径必须给几何(缺几何的位置预测=编造监督,
        调用方契约)。
        """
        assert roi_spatial.dim() == 4, (
            f"RoiLocationHead 需要 [K,C,h,w],got {tuple(roi_spatial.shape)}"
        )
        assert roi_spatial.shape[1] == self.in_channels, (
            f"通道失配: 头按 {self.in_channels} 构建,实际 {roi_spatial.shape[1]}"
        )
        h, w = roi_spatial.shape[-2:]
        if bbox_xyxy is None or not orig_w or not orig_h:
            x_ch, y_ch = None, None
        else:
            x_ch, y_ch = global_coord_channels(
                bbox_xyxy, orig_w, orig_h, h, w,
                device=roi_spatial.device, dtype=roi_spatial.dtype,
            )
        feat_out = self._spatial_with_coords(roi_spatial, x_ch, y_ch)
        feat = self.pool(feat_out).flatten(1)
        side_logits = self.side_fc(feat)
        level_logits = self.level_fc(feat)
        # fp32 softmax:同整图头(合成概率是乘积,fp16 下溢被平方放大)。
        side_probs = torch.softmax(side_logits.float(), dim=-1)
        level_probs = torch.softmax(level_logits.float(), dim=-1)
        return RoiLocationOutputs(
            side_logits=side_logits,
            level_logits=level_logits,
            location_probs=compose_location_probs(side_probs, level_probs),
        )

    def _spatial_with_coords(self, x, x_ch, y_ch):
        """跑 self.spatial(CoordConv→BN→ReLU),坐标通道按需替换。

        `CoordConv2d.forward` 自带 linspace 坐标(ROI 相对);全局坐标路要"换芯":
        把 CoordConv 的 conv 输入从 [x, linspace_x, linspace_y] 换成
        [x, global_x, global_y]——卷积核与 BN/ReLU 复用同一批参数(结构同构
        的实现载体:同一 nn.Sequential,只替换坐标两通道的取值)。
        """
        coord_conv = self.spatial[0]
        if x_ch is None:
            return self.spatial(x)
        # 手工重放 CoordConv2d.forward,仅坐标通道换源(conv 权重共用);
        # 复用 self.spatial 的 BN/ReLU(结构同构的关键:参数与层序不变)。
        coords = torch.cat(
            [x_ch.to(x.dtype), y_ch.to(x.dtype)], dim=1
        ).expand(x.shape[0], 2, x.shape[2], x.shape[3])
        fused = coord_conv.conv(torch.cat([x, coords], dim=1))
        out = fused
        for layer in list(self.spatial)[1:]:
            out = layer(out)
        return out


# ── ETE ─────────────────────────────────────────────────────────────────────


class RoiExtensionOutputs(dict):
    """键与 `ExtensionOutputs` 同义(ete_logits/ete_probs/band_ratio)。"""


class RoiExtensionHead(nn.Module):
    """ROI 版 ETE 头——ROI 内 band 加权池化,新结构(不可搬运)。

    与整图 `ExtensionHead` 的差异(scoping §C.1:整图 band = 结节边界 vs
    周围腺体,ROI 裁掉后判据只剩"结节 vs ROI 内腺体"——band 在 **ROI 域 mask**
    上提取,语义重新成立):
      - band 从 ROI 域 soft mask(裁剪+resize 后与特征图同分辨率)提取
        (`boundary_band_from_mask`,同一形态学,核语义在 ROI 域重定义);
      - 无 CBM margin 旁路(整图头吃 `cbm.logits_margin`,ROI 域无整图 CBM
        前向,新概念头归 `RoiTiradsHead` 的概念通路,ETE 不掺合——三头
        "下游不反灌上游"原则在 ROI 域同样适用)。

    Args:
        in_channels: 352。
        hidden_dim: 局部投影宽度(默认与整图头一致 128)。
        band_kernel: 形态学核(奇数;ROI 域重定义,默认仍 5)。
        dropout: MLP dropout(稀有类过拟合防护,同整图 0.3)。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        band_kernel: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.band_kernel = band_kernel
        self.local_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # 整图头 classifier 输入 hidden+1+margin_dim(4);ROI 版无 margin 旁路
        # → hidden+1。维度不同,故 state_dict 不互载(新结构,勿试)。
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_ETE_CLASSES),
        )

    def forward(
        self, roi_spatial: torch.Tensor, roi_soft_mask: torch.Tensor
    ) -> RoiExtensionOutputs:
        """roi_spatial [K,C,h,w] + ROI 域 soft mask [K,1,H',W'] → ETE 4 类。

        mask 分辨率不必等于特征分辨率(内部 adaptive_avg_pool 到 feat 尺寸,
        与整图头同式;H'/W' 是 mask 原分辨率,band 在其上提取后池化)。
        """
        assert roi_spatial.dim() == 4, (
            f"RoiExtensionHead 需要 [K,C,h,w],got {tuple(roi_spatial.shape)}"
        )
        assert roi_spatial.shape[1] == self.in_channels, (
            f"通道失配: 头按 {self.in_channels} 构建,实际 {roi_spatial.shape[1]}"
        )
        assert roi_soft_mask.dim() == 4 and roi_soft_mask.shape[1] == 1, (
            f"roi_soft_mask 期望 [K,1,H,W],got {tuple(roi_soft_mask.shape)}"
        )
        assert roi_soft_mask.shape[0] == roi_spatial.shape[0], "K 不一致"

        feat = self.local_proj(roi_spatial)                 # [K, hidden, h, w]
        band = boundary_band_from_mask(roi_soft_mask, kernel_size=self.band_kernel)
        w = F.adaptive_avg_pool2d(band, feat.shape[-2:]).to(feat.dtype)

        # fp32 累加(同整图头:特征分辨率不限定,按 QA_VP §1.3 安全写法)。
        denom = w.sum(dim=(2, 3), dtype=torch.float32)      # [K,1]
        local = (feat * w).sum(dim=(2, 3), dtype=torch.float32) / (denom + _EPS)
        band_ratio = w.mean(dim=(2, 3), dtype=torch.float32)

        z = torch.cat([local.to(feat.dtype), band_ratio.to(feat.dtype)], dim=1)
        ete_logits = self.classifier(z)
        return RoiExtensionOutputs(
            ete_logits=ete_logits,
            ete_probs=torch.softmax(ete_logits.float(), dim=-1),
            band_ratio=band_ratio,
        )


# ── clinical T ──────────────────────────────────────────────────────────────


class RoiClinicalTOutputs(dict):
    """键与 `ClinicalTOutputs` 同义(t_logits/t_probs/t_conf)。"""


class RoiClinicalTHead(nn.Module):
    """ROI 版临床 T 头——scale 通道 [d_mm, log d_mm](物理 mm)。

    与整图 `ClinicalTHead` **结构同构**(global_proj/scale_proj/classifier 逐层
    同名同形,scale_proj 输入仍是 2 维 → **state_dict 可互载**,warm-start 载体);
    语义差异只有输入源:
      - `stage4_spatial` → ROI spatial(该病灶的特征);
      - `nodule_scale`(归一化面积根)→ `d_mm`(物理直径,`physical_diameter_from_mask`);
      - `ete_logits` → ROI ETE 头的 logits(内部 detach+softmax,同整图头)。

    scale 的物理语义:d_mm 未登录即 ~0(官方无 spacing,恒 None → d=0)。
    `log(clamp(d,min=1e-4))` 兜底后 = -9.21(同整图头空 mask 阴性语义:
    "物理尺度不可判"与"无结节"同处强离群点,头可一眼区分)。真拿到 spacing
    后同一路径自然恢复物理量纲——这就是"部分可搬"的准确含义:**结构可搬,
    scale 数值域要重标定**(fine-tune 阶段头自己适应)。
    """

    def __init__(
        self,
        in_channels: int,
        ete_dim: int = NUM_ETE_CLASSES,
        hidden_dim: int = 128,
        scale_dim: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.ete_dim = ete_dim
        # 逐层同构 ClinicalTHead(禁 BN1d 同款约束:末批可能只剩 1 病灶)。
        self.global_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.scale_proj = nn.Sequential(
            nn.Linear(2, scale_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + scale_dim + ete_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_CLIN_T_CLASSES),
        )

    def forward(
        self,
        roi_spatial: torch.Tensor,
        d_mm: torch.Tensor,
        ete_logits: torch.Tensor,
    ) -> RoiClinicalTOutputs:
        """roi_spatial [K,C,h,w] + d_mm [K,1] + ROI ETE logits [K,4] → T 2 类。"""
        assert roi_spatial.dim() == 4, (
            f"RoiClinicalTHead 需要 [K,C,h,w],got {tuple(roi_spatial.shape)}"
        )
        assert roi_spatial.shape[1] == self.in_channels, (
            f"通道失配: 头按 {self.in_channels} 构建,实际 {roi_spatial.shape[1]}"
        )
        assert d_mm.dim() == 2 and d_mm.shape[1] == 1, (
            f"d_mm 期望 [K,1](physical_diameter_from_mask 返回契约),"
            f"got {tuple(d_mm.shape)}"
        )
        assert ete_logits.dim() == 2 and ete_logits.shape[1] == self.ete_dim, (
            f"ete_logits 期望 [K,{self.ete_dim}],got {tuple(ete_logits.shape)}"
        )

        pooled = F.adaptive_avg_pool2d(roi_spatial, 1).flatten(1)
        g = self.global_proj(pooled)
        # 🔴 [d_mm, log d_mm] 双通道——log 域等距(T 分期倍数判据),
        # clamp 兜 d=0(无 spacing/空 mask),同整图头 _SCALE_LOG_FLOOR 语义。
        s = d_mm.detach().to(pooled.dtype)
        s_log = torch.log(torch.clamp(s, min=_SCALE_LOG_FLOOR))
        scale_feat = self.scale_proj(torch.cat([s, s_log], dim=1))
        # detach+softmax:消 ETE logits 平移自由度,不反灌 ETE 头(独立计分)。
        ete_feat = torch.softmax(ete_logits.detach().float(), dim=-1).to(pooled.dtype)
        t_logits = self.classifier(torch.cat([g, scale_feat, ete_feat], dim=1))
        t_probs = torch.softmax(t_logits.float(), dim=-1)
        return RoiClinicalTOutputs(
            t_logits=t_logits,
            t_probs=t_probs,
            t_conf=t_probs[:, 1],     # P(T=1),同整图头口径
        )


# ── TI-RADS ─────────────────────────────────────────────────────────────────


class RoiTiradsOutputs(dict):
    """键与 `TiradsOutputs` 同义(tirads_logits/tirads_probs)。"""


class RoiConceptHead(nn.Module):
    """ROI 域第二 CBM 概念头(轻量版,新训)。

    整图 CBM(`ConceptEmbeddingCBMHead`)产五概念 logits(18 维)供 TI-RADS
    头旁路;ROI 域没有整图 CBM 前向,故另建轻量概念头:**每概念一个 Linear
    投影到该概念的类别数**,结构对齐 `cbm_head.py` 的切片宽度(comp 4/echo 4/
    shape 2/margin 4/foci 4)。不做 CEM 嵌入/熵门控/物理先验融合——那些是
    整图可解释性交付件,ROI 概念头只服务 TI-RADS 分级的特征旁路,轻量即可
    (scoping §C.1:"新概念头必然新训",不是交付的可解释性模块)。

    输出与整图 CBM 的概念 logits **同键切片语义**(`logits_comp`[K,4] 等),
    供 `RoiTiradsHead` 的 18 维拼接消费。
    """

    _CONCEPT_DIMS = (4, 4, 2, 4, 4)   # comp/echo/shape/margin/foci

    def __init__(self, in_channels: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head_comp = nn.Linear(hidden_dim, 4)
        self.head_echo = nn.Linear(hidden_dim, 4)
        self.head_shape = nn.Linear(hidden_dim, 2)
        self.head_margin = nn.Linear(hidden_dim, 4)
        self.head_foci = nn.Linear(hidden_dim, 4)

    def forward(self, pooled: torch.Tensor) -> dict:
        h = self.proj(pooled)
        return {
            "logits_comp": self.head_comp(h),
            "logits_echo": self.head_echo(h),
            "logits_shape": self.head_shape(h),
            "logits_margin": self.head_margin(h),
            "logits_foci": self.head_foci(h),
        }


class RoiTiradsHead(nn.Module):
    """ROI 版 TI-RADS 头——ROI 概念头(新训)+ 概念 logits 拼接。

    结构与整图 `TiradsHead` **同构**(global_proj + classifier,classifier 输入
    hidden+18:概念维度一致),故 state_dict 键形同——但**概念 logits 来自
    新训的 `RoiConceptHead`**,数值域不同,加载整图 tirads_head 权重属
    warm-start(可载但需 fine-tune),非零训练搬运。
    """

    def __init__(
        self,
        in_channels: int,
        concept_dim: int = 18,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.concept_dim = concept_dim
        self.global_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + concept_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_TIRADS_CLASSES),
        )

    def forward(
        self,
        roi_spatial: torch.Tensor,
        concept_logits: dict,
    ) -> RoiTiradsOutputs:
        """roi_spatial [K,C,h,w] + 概念 logits dict(五键)→ TI-RADS 5 类。

        🔴 概念 logits **必须由调用方 detach**——与整图头同一原则(TI-RADS
        梯度不得反向重塑概念;ROI 概念头新训时,这里的 detach 防的是"TI-RADS
        的梯度绕过概念头自己的监督、直接拽概念表征")。
        """
        assert roi_spatial.dim() == 4, (
            f"RoiTiradsHead 需要 [K,C,h,w],got {tuple(roi_spatial.shape)}"
        )
        assert roi_spatial.shape[1] == self.in_channels, (
            f"通道失配: 头按 {self.in_channels} 构建,实际 {roi_spatial.shape[1]}"
        )
        dims = {"logits_comp": 4, "logits_echo": 4, "logits_shape": 2,
                "logits_margin": 4, "logits_foci": 4}
        batch = roi_spatial.shape[0]
        parts = []
        for k, dim in dims.items():
            t = concept_logits[k]
            assert t.dim() == 2 and t.shape[1] == dim, (
                f"{k} 期望 [K,{dim}],got {tuple(t.shape)}"
            )
            assert t.shape[0] == batch, f"{k} batch 不一致: {t.shape[0]} vs {batch}"
            parts.append(t.detach())
        pooled = F.adaptive_avg_pool2d(roi_spatial, 1).flatten(1)
        g = self.global_proj(pooled)
        concept_feat = torch.cat(parts, dim=1)  # [K, 18]
        tirads_logits = self.classifier(torch.cat([g, concept_feat], dim=1))
        return RoiTiradsOutputs(
            tirads_logits=tirads_logits,
            tirads_probs=torch.softmax(tirads_logits.float(), dim=-1),
        )
