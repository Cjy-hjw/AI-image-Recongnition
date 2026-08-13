"""临床 T 头 —— T 临床显著性二分类（Phase2 阶段 2.3）。

## 任务与上限

官方 `T_ClinSig: {"T": 0|1, "T_conf": float}`。标签分布 `0无 2410 / 1有 1831`
（阳性率 **35.7%**，`docs/训练与架构/训练数据定稿.md:87`）—— 三头里唯一**不存在极端不平衡**的
任务，故不用 CB-Focal，普通 masked CE 即可（`losses/components/clinical_t.py`）。

🔴 **真正的难点是影像学上限而非不平衡**。T 分期的临床判据包含淋巴结与远处转移，
而这些**常不在超声视野内**，`docs/训练与架构/训练架构方案.md:22` 因此把预期 AUC 定在 **0.72**（⚠️ 蓝图里的 0.72 是**有影像学上限的论证口径、非任何实测/文献值**，是期望参考而非验收线）。
这意味着：训练到 0.72 附近就该停手去看别的头，继续在 T 上加容量是在拟合噪声。
标签本身带不可约的不确定性，这也是 `label_smoothing` 默认关闭的背景（见下）。

## 结构：三路轻量融合

    stage4_spatial ──GAP──> [B,352] ──Linear+ReLU──┐
                                                    │
    nodule_scale_detached [B,1] ──[s, log s]──────> ├── MLP → 2 类
                          ──Linear+ReLU──> [B,16]  │
                                                    │
    ete_logits ──detach──softmax──> [B,4] ─────────┘

### 为什么 GAP 而不是交界带（与侵犯头相反的选择）

ETE 看的是**界面**（证据在轮廓一圈），T 看的是**整体分期**——结节大小、腺体受累
范围、周围结构，是全局属性。故这里用全局平均池化，不做局部加权。

⚠️ **`GAP(stage4_spatial)` 与模型内部的 `x_cls` 逐位相同**（实测最大差 **0.0**）：
`MultiTaskThyroidModel.cls_pool` 就是 `AdaptiveAvgPool2d(1)`，其后只有一次 `view`。
所以本头**不需要模型额外暴露一个 pooled 键** —— 三头统一吃 `stage4_spatial`，
接口面更小。若哪天 `cls_pool` 改成 max/attention pooling，这条等价性就断了，
届时本头拿到的是 GAP 而非 CBM 用的那份特征（不会报错，只是两者不再是同一向量）。

### 为什么尺度标量要带 log 通道

`nodule_scale_from_mask` 给的是归一化等效直径，实测量级：面积占比 0.1% → 0.0313、
1% → 0.0996、5% → 0.2227、20% → 0.4453。典型结节落在 **0.03 ~ 0.1**，而它要和
352 维全局特征拼在一起 —— 单看线性值，小结节之间的差异被压在 0.07 的窄区间里。

T 分期判据（T1 ≤2cm / T2 2-4cm / T3 >4cm）关心的是直径的**倍数**关系，而倍数在
线性尺度上不是等距的、在 log 尺度上才是。实测：`0.01→0.02`、`0.1→0.2`、
`0.4→0.8` 三对都是 2 倍，线性差分别是 0.01 / 0.1 / 0.4（差 40 倍），
**log 差恒为 0.6931**。故送 `[s, log s]` 两个通道，让 MLP 两种尺度都能用。

⚠️ **别把这条护栏说得比实际强**：去掉 log 通道（两路都送线性 `s`）后，头**照样
分得开** 0.001 和 0.01 —— `scale_proj` 的 Linear 能把 0.009 的差放大成任意可测的
输出差（变异实验实证，`tmp_main_falsify_clinical_t.py`）。log 通道解决的不是
"能否表示"，而是**分辨率是否均匀**：没有它，同一组权重要同时分辨 0.03~0.1 段和
0.2~0.45 段，两段的梯度尺度差几十倍。对应的测试因此只能是白盒的
（`test_scale_input_has_log_channel_with_equal_spacing_per_ratio`），黑盒行为判据
在这里零区分力。

`log` 前 `clamp(min=1e-4)`：空 mask（阴性病例）时 s 恰为 0，`log 0 = -inf` 会
直接毒化整个 MLP。下界 1e-4 对应面积占比 1e-8 —— 512² 图上不足 0.03 个像素，
物理上不可能是真结节，故 clamp 不会截断任何有效样本。截断后 `log = -9.21`，
比最小的真结节（0.0313 → -3.47）低约 3 倍量级，**阴性样本因此在 log 通道上是
强离群点**，这是有意的：让头能一眼分出"根本没有结节"。
（交付推理侧阴性病例不产出结节条目，这条路径只在训练时走到。）

### 为什么 ETE 送 probs 而不是原始 logits

`docs/训练与架构/训练架构方案.md` 的措辞是「ETE logits」，本实现接收 logits 但**内部转 softmax**。
理由**不是**"随机初始化下 logits 就很大"——那是错的，Gate2 实测真模型 5 个 seed 下
`|ete_logits|max` 只有 **0.12 ~ 0.27**。真正的两条理由是：

1. **平移不变性**。`logits` 与 `logits + c` 表示同一个分布，但直接拼进 MLP 会让 T 头
   对这个**无意义的自由度**敏感 —— ETE 头训练中整体漂移一个常数（完全不改变它的
   预测），T 头的输出却会跟着变。softmax 消掉这个自由度，且**信息无损**（丢掉的正是
   这个自由度本身）。已由 `test_head_is_invariant_to_ete_logit_shift` 锁死。
2. **量级随训练无上界增长**。交叉熵会持续推大 logit gap，随机初始化时的 0.27 说明不了
   收敛后的量级。而 scale ∈ [0,1]、全局特征经 ReLU 后是 O(1) —— 让一个无界量与两个
   有界量共享 MLP 首层，等于把量级平衡交给运气。softmax 后 ∈ [0,1] 且和为 1，与训练
   进度无关。

🔴 **ETE 输入必须 detach**。理由与侵犯头 detach CBM `margin` 同源但更硬：ETE 是
官方**独立计分**的维度，若 T 头的梯度能反灌侵犯头，ETE 表征会被 T 任务拽着走 ——
换来 T 涨一点、丢掉 ETE 的分，净亏。三头的原则统一为「**下游头消费上游头的输出
一律 detach**」：位置头独立、侵犯头 detach CBM margin、T 头 detach ETE。

### 与分割主干的梯度隔离

`nodule_scale_detached` 已 detach（`mask_features` 模块级契约），`ete_logits` 在本头
内 detach。唯一有梯度的入口是 `stage4_spatial` → 回流 `cls_backbone`，**这是设计
意图**（供调度阶段 B 的 LLRD 解冻），且 `adaptor` 对 `seg_features` 早已 detach，
故 `seg_backbone` 梯度仍精确 0，`gate8_adaptor_isolation` 的前提不破。

## 🔴 T_conf 的语义（交付层口径，别填反）

`t_conf` 取 **P(T=1)**，不是"对所报 T 值的置信度"。二者在 T=0 时给出的数完全不同
（前者 0.1、后者 0.9），官方样例 `{"T":1,"T_conf":0.92}` 单看无法区分。

取前者的依据不是猜测，是**与交付层既有口径自洽**：`prediction_json.build_nodule_entry`
里 `Malignancy.predicted = bool(confidence >= threshold)` —— 即 `confidence` 是
"恶性的概率"而非"预测的可信度"。同一份 schema 里两个置信度字段没有理由用相反语义。
⚠️ 仍属**官方未明确项**，已列入待问组委会清单；若官方澄清为后者，改动点只有
`ClinicalTOutputs.t_conf` 这一行 + 交付侧读取处。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ClinicalTHead",
    "ClinicalTOutputs",
    "NUM_CLIN_T_CLASSES",
    "T_POSITIVE_INDEX",
]

# 官方 `T_ClinSig.T` ∈ {0, 1}，与 0-based 类别下标恰好同值（不像 ETE/Location 要 ±1）。
NUM_CLIN_T_CLASSES = 2
T_POSITIVE_INDEX = 1  # `t_probs[:, 1]` = P(T=1)，即 T_conf 的取值来源

# log 的下界。见模块 docstring：对应面积占比 1e-8，物理上不可能是真结节。
_SCALE_LOG_FLOOR = 1e-4


class ClinicalTOutputs(dict):
    """临床 T 头输出。dict 子类，与 `LocationOutputs` / `ExtensionOutputs` 风格一致。

    键：
      - `t_logits` [B,2] —— **训练走这个**（原始 logits，损失内部自己做 log_softmax）。
      - `t_probs` [B,2] —— fp32 softmax，和为 1。交付需要完整分布（官方要求所有
        分类任务给概率）。
      - `t_conf` [B] —— `t_probs[:, 1]` = **P(T=1)**，直接填官方 `T_ClinSig.T_conf`。
        🔴 是"阳性概率"不是"预测可信度"，见模块 docstring 的口径论证。
        形状是 **[B] 一维**（不是 [B,1]）—— 交付侧逐样本取标量，
        且 [B,1] 与 [B] 相乘会隐式广播成 [B,B]（QA_VP §2.2）。
    """


class ClinicalTHead(nn.Module):
    """全局特征 + 结节尺度 + ETE 概率 → T 临床显著性 2 类。

    Args:
        in_channels: 必须是 `MultiTaskThyroidModel.CLS_POOLED_DIM`（**352**），
            与位置头/侵犯头同一约束（208 是 adaptor 的**输入**通道，不是这里的）。
        ete_dim: ETE 类别数（4）。
        hidden_dim: 全局投影与 MLP 的宽度。
        scale_dim: 尺度标量的投影宽度。给 2 维输入一个独立的 16 维通路，是为了
            让它不被 352 维全局特征在首层淹没 —— 直接把 2 维拼进 352 维再过一个
            Linear，尺度信号只占 2/354 的输入权重。
        dropout: MLP dropout。

    🔴 **本头刻意不使用 `BatchNorm1d`**。实测 `BatchNorm1d` 在 train 模式下
    batch=1 直接 `ValueError: Expected more than 1 value per channel`，而训练最后
    一个 batch 完全可能只剩 1 个样本（`drop_last` 并非到处都开）。侵犯头能用
    `BatchNorm2d` 是因为它作用在 16×16 空间上、batch=1 仍有 256 个元素可统计；
    这里 GAP 之后只剩 [B,C]，没有那个余量。需要归一化时应改用 `LayerNorm`
    （与 batch 无关），不要把 BN 加回来。
    """

    def __init__(
        self,
        in_channels: int,
        ete_dim: int = 4,
        hidden_dim: int = 128,
        scale_dim: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.ete_dim = ete_dim
        self.global_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # 输入 2 维 = [s, log s]，见模块 docstring 的 log 通道论证。
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
        stage4_spatial: torch.Tensor,
        nodule_scale: torch.Tensor,
        ete_logits: torch.Tensor,
    ) -> ClinicalTOutputs:
        assert stage4_spatial.dim() == 4, (
            f"临床 T 头需要空间特征图 [B,C,H,W]，got {tuple(stage4_spatial.shape)}"
            " —— 本头自己做 GAP，传池化后的 [B,C] 会在这里就被拦下"
        )
        assert stage4_spatial.shape[1] == self.in_channels, (
            f"通道失配: 头按 in_channels={self.in_channels} 构建，"
            f"实际输入 {stage4_spatial.shape[1]}"
        )
        # [B,1] 而非 [B]：`mask_features.nodule_scale_from_mask` 的返回契约。
        # 若有人 squeeze 成 [B] 再传进来，下面的 cat(dim=1) 会 IndexError（能崩，
        # 但报错指向 torch 内部）；这里提前拦下并说清期望。
        assert nodule_scale.dim() == 2 and nodule_scale.shape[1] == 1, (
            f"nodule_scale 期望 [B,1]，got {tuple(nodule_scale.shape)}"
            " —— 见 mask_features.nodule_scale_from_mask 的返回契约"
        )
        assert ete_logits.dim() == 2 and ete_logits.shape[1] == self.ete_dim, (
            f"ete_logits 期望 [B,{self.ete_dim}]，got {tuple(ete_logits.shape)}"
        )
        batch = stage4_spatial.shape[0]
        assert nodule_scale.shape[0] == batch and ete_logits.shape[0] == batch, (
            f"batch 不一致: feat {batch} / scale {nodule_scale.shape[0]}"
            f" / ete {ete_logits.shape[0]}"
        )

        # GAP 与 `MultiTaskThyroidModel.cls_pool` 逐位等价（实测差 0.0），见 docstring。
        pooled = F.adaptive_avg_pool2d(stage4_spatial, 1).flatten(1)  # [B, C]
        g = self.global_proj(pooled)  # [B, hidden]

        # 尺度：线性 + log 双通道。detach 是二道保险 —— 上游 `nodule_scale_detached`
        # 已 detach，但本头的契约不依赖调用方传对（公开 forward 可能被别处复用）。
        s = nodule_scale.detach().to(pooled.dtype)
        s_log = torch.log(torch.clamp(s, min=_SCALE_LOG_FLOOR))
        scale_feat = self.scale_proj(torch.cat([s, s_log], dim=1))  # [B, scale_dim]

        # 🔴 detach：不让 T 的梯度反灌侵犯头（ETE 是官方独立计分维度）。
        # softmax：消掉 logits 的平移自由度 + 给出与训练进度无关的有界表示（见 docstring；
        # 注意随机初始化下 logits 实测只有 0.12~0.27，理由不是"当前量级太大"）。
        # 先 `.float()` 再 softmax 是**精度**而非溢出考虑：4 类 softmax 在 fp16 下
        # 概率和恒为 1.0、永远有限（实测 logit gap=20/50/100 均 sum=1.0、finite），
        # 但 gap≳20 时小项会被抹成 0.0 —— ETE 稀有类（血管 0.12%）的概率正落在会被
        # 抹掉的量级。fp32 softmax 保住这个信息，回转 fp16 时它已是有限精度内的值。
        ete_feat = torch.softmax(ete_logits.detach().float(), dim=-1).to(pooled.dtype)

        t_logits = self.classifier(torch.cat([g, scale_feat, ete_feat], dim=1))
        t_probs = torch.softmax(t_logits.float(), dim=-1)
        return ClinicalTOutputs(
            t_logits=t_logits,
            t_probs=t_probs,
            # P(T=1)。squeeze 到 [B] —— 见 ClinicalTOutputs docstring 的维度说明。
            t_conf=t_probs[:, T_POSITIVE_INDEX],
        )
