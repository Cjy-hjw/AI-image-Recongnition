"""TI-RADS 头 —— ACR TI-RADS 总分类 5 级（backlog #124）。

## 为什么需要这个头

目标三(60% 权重)四项子任务里,分割/检测/良恶性都有训练信号,唯独 TI-RADS 加权
Kappa 这项从设计起就没有专属训练路径 —— 此前交付的 `TI_RADS_Class` 是拿**从未受过
监督**的 CBM 概念反推 ACR 点数得出的一个数,结构合法但 Kappa 期望 ≈ 0(乱猜)。
本头补上这条缺失的监督链路,详见 `docs/项目现状总纲.md` §1 与 `docs/训练与架构/
训练数据定稿.md` §1.7/§2.2。

## 类别(官方全量实测,`docs/训练与架构/训练数据定稿.md` §1.7)

xlsx `ROI-TIRADS` 列真实字面是**罗马数字+"类"**(`V类`,无数字前缀);总分类列
M 行独立缺失仅 2.8%(≈4740 条可监督),与"详细评分子项"(缺失 75.8%,本头不做)
是两个不同粒度的列。官方指标 = TI-RADS 加权 Kappa,适用范围「仅匹配成功的主
病灶」——**不带**"且为恶性"限定(与 ETE/T 不同,评审规则原文无此限定)。

## 结构:全局特征 + CBM 五概念旁路(不新造几何特征)

TI-RADS 分级是**结节整体特征**的综合判断(成分/回声/形状/边缘/钙化灶),不像 ETE
那样依赖轮廓局部交界带,故用 GAP 而非交界带池化(与 `ClinicalTHead` 同一选择)。

    stage4_spatial ──GAP──────────────────┐
                                            ├── MLP → 5 类
    CBM 五概念 logits(**detach**,18维) ────┘

🔴 **CBM 五概念 logits 必须 detach**(与 `ExtensionHead` 对 `margin` 的处理同一
原则):ACR TI-RADS 的判据本就是这五个概念的加权计数,拿来当旁路先验是最直接的
信息;但不能让 TI-RADS 的梯度反向重塑这些概念 —— CBM 是交付的可解释性模块。

⚠️ **明确记录风险**:这 5 个概念在官方数据上**没有监督**(`_FIELD_ALIASES` 没有
对应候选列,概念分支梯度恒 0,见 `docs/训练与架构/三头实现记录.md` §5.1),此刻
喂给 TI-RADS 头的是噪声。detach 保证对概念头无害,概念一旦有监督这条路径现成。

考虑过、放弃的方案:曾计划新加 `nodule_aspect_ratio_from_mask`(soft mask 二阶矩
推纵横比)喂 ACR"taller-than-wide"判据。放弃理由:① `logits_shape`[B,2] 这个
现成 CBM 概念已承载同一信息,二阶矩是冗余的第二路 shape 信号;② 若真要引入,
`clamp(min=1e-4)` 这个下界是从 `clinical_t.py` 的**面积占比**用例照搬的,量纲对
"纵横比"不成立(1e-4 对应面积占比 1e-8,不是长度比),照抄反而引入新错误。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.label_schema import TIRADS_CODES

__all__ = [
    "TiradsHead",
    "TiradsOutputs",
    "NUM_TIRADS_CLASSES",
    "TIRADS_IDX_TO_CODE",
]

# 顺序锁(同 `heads/extension.py`/`heads/location.py` 的做法):本头输出的第 i 个
# logit 代表 `TIRADS_CODES[i]`。若官方编码变了而这里没跟着改,概率分布会**静默
# 错位** —— JSON 合法、和为 1、测试照过,只是把"V 级"填进了"I 级"的槽。
_EXPECTED_TIRADS_CODES = (1, 2, 3, 4, 5)
assert tuple(TIRADS_CODES) == _EXPECTED_TIRADS_CODES, (
    f"TIRADS_CODES 已变更: {tuple(TIRADS_CODES)} != {_EXPECTED_TIRADS_CODES}。"
    " TI-RADS 头的 logit 次序、official_loader 的 `tirads = code - 1`、"
    " 以及交付侧 `TI_RADS_Class` 字段三处都按旧序写死,须同步修正。"
)

NUM_TIRADS_CLASSES = len(TIRADS_CODES)
TIRADS_IDX_TO_CODE = TIRADS_CODES  # 0-based 头输出下标 → 官方输出码(= idx + 1)

# CBM 五概念 logits 的宽度(`cbm_head.py` 固定切片:comp/echo 各4、shape 2、
# margin 4、foci 4,concat = 18),与 `vclub` 的 y_dim=18 是同一份契约。
_COMP_DIM = 4
_ECHO_DIM = 4
_SHAPE_DIM = 2
_MARGIN_DIM = 4
_FOCI_DIM = 4
CONCEPT_DIM = _COMP_DIM + _ECHO_DIM + _SHAPE_DIM + _MARGIN_DIM + _FOCI_DIM  # 18


class TiradsOutputs(dict):
    """TI-RADS 头输出。dict 子类,与 `ExtensionOutputs`/`LocationOutputs` 风格一致。

    键:
      - `tirads_logits` [B,5] —— 训练走这个,未经任何不平衡校正的原始 logits。
      - `tirads_probs` [B,5] —— 推理/交付用,fp32 softmax,和为 1。
    """


class TiradsHead(nn.Module):
    """Stage-4 全局特征 + CBM 五概念旁路(detach) → TI-RADS 5 类。

    Args:
        in_channels: 必须是 `MultiTaskThyroidModel.CLS_POOLED_DIM`(**352**),
            不是 `CLS_STAGE4_CHANNELS`(208)—— 后者是 adaptor 的输入通道。
        concept_dim: CBM 五概念 logits 拼接后的宽度(18,见模块常量 `CONCEPT_DIM`)。
        hidden_dim: 全局投影与 MLP 的宽度。
        dropout: MLP dropout,与 CBM 头同量级默认 0.3。
    """

    def __init__(
        self,
        in_channels: int,
        concept_dim: int = CONCEPT_DIM,
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
        stage4_spatial: torch.Tensor,
        logits_comp: torch.Tensor,
        logits_echo: torch.Tensor,
        logits_shape: torch.Tensor,
        logits_margin: torch.Tensor,
        logits_foci: torch.Tensor,
    ) -> TiradsOutputs:
        assert stage4_spatial.dim() == 4, (
            f"TI-RADS 头需要空间特征图 [B,C,H,W],got {tuple(stage4_spatial.shape)}"
            " —— 本头自己做 GAP,传池化后的 [B,C] 会在这里就被拦下"
        )
        assert stage4_spatial.shape[1] == self.in_channels, (
            f"通道失配: 头按 in_channels={self.in_channels} 构建,"
            f"实际输入 {stage4_spatial.shape[1]}"
        )
        batch = stage4_spatial.shape[0]
        concepts = (
            (logits_comp, _COMP_DIM),
            (logits_echo, _ECHO_DIM),
            (logits_shape, _SHAPE_DIM),
            (logits_margin, _MARGIN_DIM),
            (logits_foci, _FOCI_DIM),
        )
        for name, (t, dim) in zip(
            ("comp", "echo", "shape", "margin", "foci"), concepts
        ):
            assert t.dim() == 2 and t.shape[1] == dim, (
                f"logits_{name} 期望 [B,{dim}],got {tuple(t.shape)}"
            )
            assert t.shape[0] == batch, (
                f"batch 不一致: stage4_spatial {batch} vs logits_{name} {t.shape[0]}"
            )

        pooled = F.adaptive_avg_pool2d(stage4_spatial, 1).flatten(1)  # [B, C]
        g = self.global_proj(pooled)  # [B, hidden]

        # 🔴 detach:不让 TI-RADS 的梯度反向重塑 CBM 五概念(见模块 docstring)。
        concept_feat = torch.cat(
            [t.detach().to(pooled.dtype) for t, _ in concepts], dim=1
        )  # [B, 18]

        tirads_logits = self.classifier(torch.cat([g, concept_feat], dim=1))
        return TiradsOutputs(
            tirads_logits=tirads_logits,
            # fp32 softmax:与 ETE/T 头同一考量,稀有类概率在 softmax 边界易被 fp16 抹零。
            tirads_probs=torch.softmax(tirads_logits.float(), dim=-1),
        )
