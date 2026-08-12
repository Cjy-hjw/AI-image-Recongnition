"""侵犯头 —— 甲状腺外侵犯（ETE）4 类分级（Phase2 阶段 2.2）。

## 类别与不平衡（官方全量实测，`docs/训练数据定稿.md`）

| 0-based | 输出码 | 语义 | 例数 | 占比 |
|---|---|---|---|---|
| 0 | 1 | 无 | 3906 | 92.02% |
| 1 | 2 | 甲状腺包膜 | 267 | 6.29% |
| 2 | 3 | 毗邻软组织 | 67 | 1.58% |
| 3 | 4 | 血管 | **5** | **0.12%** |

🔴 **保 4 类，不并类**。血管仅 5 例，看着该并进"软组织"，但若官方指标是 Macro-AUC，
并类会让该类 AUC **无从计算**（等于直接丢该项分）。这不是配置偏好，是评分结构决定的
——`docs/训练架构方案.md` 据此定稿。要翻此决定，必须先拿到官方评测口径。

## 结构：交界带局部特征 + CBM 边缘旁路

ETE 的判据是**肿瘤与包膜/毗邻组织的界面是否被突破**，证据全在轮廓一圈，结节实心
内部与之无关。故不用全局池化，而是：

    stage4_spatial ──1×1 Conv+BN+ReLU──┐
                                        ├── 交界带加权池化 ──┐
    soft_mask_detached ──膨胀−腐蚀──────┘                    ├── MLP → 4 类
                                          band_ratio ────────┤
    margin(CBM 边缘 logits, **detach**) ──────────────────────┘

三路输入各自的理由：
- **局部特征**：把注意力钉在解剖学相关的窄环上（带的提取与实测行为见
  `mask_features.boundary_band_from_mask`）。
- **band_ratio**（带权重在特征图上的**均值**）：消解一个真实歧义 —— 阴性病例（无结节）
  的加权池化分母为 0、局部特征恒等于全 0 向量，而"带存在但特征恰好均值为 0"
  同样是全 0 向量。没有这个标量，头分不清"没有界面"和"界面无异常"。
  它同时是结节尺度的代理（实测 20px 结节 0.00122、200px 0.01221）。
  ⚠️ **它是权重均值，不是"非零 cell 占比"，两套数字不可互推**：200px 结节的
  band_ratio 是 0.01221（= 原图带像素占比，二者恒等，因 `adaptive_avg_pool2d`
  保均值），而它在 16×16 上占了 28/256 = 10.9% 的非零 cell —— 差了近 10 倍。
  拿后者去对前者会得出"数字矛盾"的错误结论（审查时真发生过）。
- 🔴 **CBM `margin` logits 必须 detach**。ETE 与 TI-RADS「边缘」概念高度相关，
  拿来当旁路先验是白捡的信息；但**不能让 ETE 的梯度反向重塑这个概念** ——
  CBM 是我方交付的可解释性模块（打包 CBM 报告生成器），概念一旦被 ETE 的监督
  拽着漂移，它就不再是"边缘"，报告随之失去可信度。这与阶段 2.3 里
  「T 头消费 ETE logits 也要 detach」是同一条原则。
  注意 `stage4_spatial` **不 detach** —— 那条通路回流 `cls_backbone` 是设计意图。

## 与分割主干的梯度隔离

`soft_mask_detached` 与本头内部的 `boundary_band_from_mask` 都已 detach，两道保险。
ETE 的梯度**到不了 `seg_backbone`**，`gate8_adaptor_isolation` 的精确零断言前提不破，
也不复刻 2026-08-11 Stage2 崩塌四因之一（物理先验通路梯度由 0 到满强度）。
**代价要说清**：ETE 的监督信号永远不会去改善分割边界 —— 这是主动放弃的一条增益，
换取旧任务不退化。若真实数据上 ETE 明显打不过影像学上限，这是第一个该复议的决策。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.label_schema import ETE_CODES
from src.models.mask_features import boundary_band_from_mask

__all__ = [
    "ExtensionHead",
    "ExtensionOutputs",
    "NUM_ETE_CLASSES",
    "ETE_IDX_TO_CODE",
]

# 顺序锁（同 `heads/location.py` 的做法）：本头输出的第 i 个 logit 代表 `ETE_CODES[i]`。
# 若哪天官方编码变了而这里没跟着改，概率分布会**静默错位** —— JSON 合法、和为 1、
# 测试照过，只是把"包膜侵犯"填进了"无侵犯"的槽。故在导入期就断死。
_EXPECTED_ETE_CODES = (1, 2, 3, 4)
assert tuple(ETE_CODES) == _EXPECTED_ETE_CODES, (
    f"ETE_CODES 已变更: {tuple(ETE_CODES)} != {_EXPECTED_ETE_CODES}。"
    " 侵犯头的 logit 次序、official_loader 的 `extrathy_ext = code - 1`、"
    " 以及交付侧 `ExtrathyroidalExtension` 字段三处都按旧序写死，须同步修正。"
)

NUM_ETE_CLASSES = len(ETE_CODES)
ETE_IDX_TO_CODE = ETE_CODES  # 0-based 头输出下标 → 官方输出码（= idx + 1）

_EPS = 1e-8


class ExtensionOutputs(dict):
    """侵犯头输出。dict 子类，与 `ModelOutputs` / `LocationOutputs` 风格一致。

    键：
      - `ete_logits` [B,4] —— **训练走这个**。损失内部还要再叠 Logit Adjustment，
        故这里必须是**未经任何不平衡校正的原始 logits**。
      - `ete_probs` [B,4] —— 推理/交付用，fp32 softmax，和为 1。
        🔴 **不要在这里减 `tau·log(prior)`**：Logit Adjustment 只作用于训练损失，
        训练完的原始 logits 本身已逼近类先验无关的平衡后验（Menon et al. 2021），
        推理侧再调一次等于反向过校正。
      - `band_ratio` [B,1] —— 交界带面积比，调试/可解释性用（也是头的输入之一）。
    """


class ExtensionHead(nn.Module):
    """Stage-4 空间特征 × 结节交界带 + CBM 边缘旁路 → ETE 4 类。

    Args:
        in_channels: 必须是 `MultiTaskThyroidModel.CLS_POOLED_DIM`（**352**），
            不是 `CLS_STAGE4_CHANNELS`（208）—— 后者是 adaptor 的输入通道。
        margin_dim: CBM `logits_margin` 的宽度（4）。
        hidden_dim: 局部投影与 MLP 的宽度。
        band_kernel: 交界带形态学核边长（奇数）；实测带宽 = `band_kernel − 1` px。
        dropout: MLP dropout。极端不平衡下（血管 5 例）过拟合风险高，默认给到 0.3，
            与 `config.train.cbm_dropout` 同量级。
    """

    def __init__(
        self,
        in_channels: int,
        margin_dim: int = 4,
        hidden_dim: int = 128,
        band_kernel: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.margin_dim = margin_dim
        self.band_kernel = band_kernel
        # 1×1 投影放在加权池化**之前**：带只覆盖少数 cell（16×16 下小结节仅 4 个），
        # 先投影再池化能让 BN 在整张特征图上统计、ReLU 逐 cell 生效；
        # 反过来先池化再 Linear，等价于对一个已被平均掉的向量做线性变换，表达力更弱。
        # （与位置头 CoordConv 的「先非线性再池化」是同一类考量，但成因不同：
        #   那里是坐标通道在线性算子下退化成常数，这里是 BN/ReLU 的作用域问题。）
        self.local_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 1 + margin_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_ETE_CLASSES),
        )

    def forward(
        self,
        stage4_spatial: torch.Tensor,
        soft_mask: torch.Tensor,
        margin_logits: torch.Tensor,
    ) -> ExtensionOutputs:
        assert stage4_spatial.dim() == 4, (
            f"侵犯头需要空间特征图 [B,C,H,W]，got {tuple(stage4_spatial.shape)}"
            " —— 交界带加权池化要按位置对齐，池化后的 [B,C] 用不了"
        )
        assert stage4_spatial.shape[1] == self.in_channels, (
            f"通道失配: 头按 in_channels={self.in_channels} 构建，"
            f"实际输入 {stage4_spatial.shape[1]}"
        )
        assert soft_mask.dim() == 4 and soft_mask.shape[1] == 1, (
            f"soft_mask 期望 [B,1,H,W]，got {tuple(soft_mask.shape)}"
        )
        assert soft_mask.shape[0] == stage4_spatial.shape[0], (
            f"batch 不一致: mask {soft_mask.shape[0]} vs feat {stage4_spatial.shape[0]}"
        )
        assert margin_logits.dim() == 2 and margin_logits.shape[1] == self.margin_dim, (
            f"margin_logits 期望 [B,{self.margin_dim}]，got {tuple(margin_logits.shape)}"
        )

        feat = self.local_proj(stage4_spatial)  # [B, hidden, h, w]
        band = boundary_band_from_mask(soft_mask, kernel_size=self.band_kernel)
        # 带在**原分辨率**上提取（保住 4px 带宽的物理语义），再降到特征分辨率。
        # 反过来先降采样再做形态学，16×16 上一个 cell 的核就等于原图 32px，带宽失真。
        w = F.adaptive_avg_pool2d(band, feat.shape[-2:]).to(feat.dtype)  # [B,1,h,w]

        # 求和一律 fp32 累加：虽然此处只有 h*w=256 项、fp16 不会溢出，但本函数不限定
        # 特征分辨率（换 backbone/img_size 即变），按 QA_VP §1.3 取安全写法。
        denom = w.sum(dim=(2, 3), dtype=torch.float32)  # [B,1]
        local = (feat * w).sum(dim=(2, 3), dtype=torch.float32) / (denom + _EPS)
        band_ratio = w.mean(dim=(2, 3), dtype=torch.float32)  # [B,1]

        # 转回 feat.dtype 再进 Linear：autocast 下无所谓，但 `.half()` 直接推理时
        # fp32 输入撞 fp16 权重会 RuntimeError。
        z = torch.cat(
            [
                local.to(feat.dtype),
                band_ratio.to(feat.dtype),
                # 🔴 detach：见模块 docstring —— 不让 ETE 梯度反向重塑 CBM「边缘」概念。
                margin_logits.detach().to(feat.dtype),
            ],
            dim=1,
        )
        ete_logits = self.classifier(z)
        return ExtensionOutputs(
            ete_logits=ete_logits,
            # fp32 softmax：fp16 autocast 下指数运算易下溢，而 ETE 的稀有类概率本就
            # 在 1e-3 量级，下溢会把它直接抹成 0。
            ete_probs=torch.softmax(ete_logits.float(), dim=-1),
            band_ratio=band_ratio,
        )
