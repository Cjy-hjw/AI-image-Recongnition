"""门控注意力多示例学习(Gated-Attention MIL)——患者级良恶性聚合。

动机(见 memory/patient-leakage-and-probe-trap.md + reference/…MIL深度方案.md):
    ThyUS2Path 是**患者级弱标签**:一个患者一个病理 histo_label,广播到其所有
    超声帧。但一个患者几十帧里只有少数帧拍到病灶靶区,大量帧是正常腺体/伪影。
    单帧强监督 → 梯度被背景帧污染 → 无泄漏患者级 AUC 卡在 ~0.50(已对拍证实)。

    MIL 把一个患者的全部帧当作一个「包(bag)」,门控注意力自动学习每帧对最终
    诊断的贡献权重(压制背景帧、放大病灶帧),加权聚合成**患者级**特征向量再分类。

阶段1(本文件当前用法):冻结 ImageNet backbone,只训 mil_pooling + classifier。
    目标:无泄漏患者级 AUC 从 0.50 突破到 0.65-0.70,验证 MIL 路线成立。

架构:Ilse et al. (2018) "Attention-based Deep MIL" 的门控注意力变体。
    独立 timm EfficientNet-B2(in_chans=1, num_classes=0 → 1408 维标准 pooled 特征),
    不复用 MultiTaskThyroidModel(去掉无关 seg/CBM 分支,省显存)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from src.models.pretrained_loader import load_local_safetensors

# EfficientNet-B2 标准 forward_features 后 pooled 特征维度(含 conv_head,非项目 352 截断)。
EFFNET_B2_FEATURE_DIM = 1408
# ImageNet 权重(A800 已上传;缺失时 load_local_safetensors 静默返回 False 走随机初始化)。
_DEFAULT_BACKBONE_WEIGHTS = "checkpoints/efficientnet_b2_pretrained.safetensors"


class GatedAttentionPooling(nn.Module):
    """门控注意力池化:把一个 bag 的 N 个实例特征聚合成 1 个 bag 级向量。

    a_j = softmax_j( w^T ( tanh(V x_j) ⊙ sigmoid(U x_j) ) )
    z   = Σ_j a_j x_j

    tanh 分支捕获非线性关联,sigmoid 门控自适应放大/抑制,⊙ 逐元素乘比单 tanh
    梯度更稳(Ilse et al. 2018)。attn_dim 为注意力投影维度 L。
    """

    def __init__(self, feature_dim: int = EFFNET_B2_FEATURE_DIM, attn_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.attn_dim = attn_dim
        self.attention_V = nn.Sequential(nn.Linear(feature_dim, attn_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(feature_dim, attn_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor):
        """x: [N, C] 单患者 N 帧的实例特征。返回 (bag_repr[1, C], weights[N, 1])。"""
        v = self.attention_V(x)              # [N, L]
        u = self.attention_U(x)              # [N, L]
        scores = self.attention_w(v * u)     # [N, 1]  门控融合后打分
        weights = F.softmax(scores, dim=0)   # [N, 1]  在帧维度归一化
        bag_repr = torch.sum(weights * x, dim=0, keepdim=True)  # [1, C]
        return bag_repr, weights


class ThyroidMILNet(nn.Module):
    """端到端 MIL 网络:逐帧 backbone 提特征 → 门控注意力聚合 → 患者级分类。

    输入一个患者的 bag [N, 1, H, W],输出 (logits[1, 2], weights[N, 1])。
    weights 可视化每帧对诊断的贡献,供临床解释。
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b2",
        pretrained: bool = True,
        feature_dim: int = EFFNET_B2_FEATURE_DIM,
        attn_dim: int = 128,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        backbone_weights: str = _DEFAULT_BACKBONE_WEIGHTS,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        # 独立单通道 backbone,num_classes=0 → 输出 pooled 特征(EfficientNet-B2=1408)。
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1, num_classes=0
        )
        if pretrained:
            loaded = load_local_safetensors(
                self.backbone, backbone_weights, in_chans=1, stem_key="conv_stem.weight"
            )
            if not loaded:
                print(
                    f"[ThyroidMILNet] 本地 backbone 权重未加载({backbone_weights}),"
                    "走随机/timm 默认初始化。",
                    flush=True,
                )
        # 阶段1:冻结 backbone,只训聚合头 + 分类头。
        self.backbone_frozen = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.mil_pooling = GatedAttentionPooling(feature_dim, attn_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim, num_classes),
        )

    def _extract_features(self, bag: torch.Tensor) -> torch.Tensor:
        """bag [N,1,H,W] → 实例特征 [N, feature_dim]。冻结时 backbone 不建图省显存。"""
        if self.backbone_frozen:
            self.backbone.eval()  # 冻结:BN 用 running stats,前向不更新统计量
            with torch.no_grad():
                return self.backbone(bag)
        return self.backbone(bag)

    def forward(self, bag: torch.Tensor):
        """bag: [N, 1, H, W] 单患者所有帧。返回 (logits[1, num_classes], weights[N, 1])。"""
        assert bag.dim() == 4, f"bag 须为 [N,1,H,W],得到 {tuple(bag.shape)}"
        feats = self._extract_features(bag)            # [N, C]
        bag_repr, weights = self.mil_pooling(feats)    # [1, C], [N, 1]
        logits = self.classifier(bag_repr)             # [1, num_classes]
        return logits, weights

    def trainable_parameters(self):
        """阶段1 供 optimizer 用:只含 mil_pooling + classifier(backbone 已冻)。"""
        return [p for p in self.parameters() if p.requires_grad]

    # ── Phase2 接口 ───────────────────────────────────────────────────────────

    def unfreeze_backbone(self) -> None:
        """Phase2: 解冻 backbone，使其参与梯度更新（配合 LLRD optimizer 使用）。

        调用后 backbone_frozen=False；_extract_features 将正常建计算图。
        """
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.backbone_frozen = False
        self.backbone.train()

    def get_llrd_param_groups(
        self,
        head_lr: float,
        backbone_lr_scale: float = 0.1,
        eta: float = 0.75,
    ) -> list:
        """Phase2 LLRD 参数组:深层 lr 高、浅层 lr 低，head 用 head_lr。

        backbone_base_lr = head_lr * backbone_lr_scale（conv_head 等最深层）。
        每往输入方向一层，lr 乘以 eta（标准 LLRD，Sun et al. 2019）。

        EfficientNet-B2 层序（深→浅，depth_idx 0 最深）:
            conv_head/bn2/global_pool → blocks[6] → ... → blocks[0] → conv_stem/bn1/act1

        Returns:
            list of dicts，可直接传给 torch.optim.AdamW。
        """
        # 深→浅排列；depth_idx=0 最深，lr = backbone_base_lr * eta^0 = backbone_base_lr
        _backbone_layer_prefixes = [
            ["conv_head", "bn2", "act2", "global_pool"],  # 最深输出端
            ["blocks.6"],
            ["blocks.5"],
            ["blocks.4"],
            ["blocks.3"],
            ["blocks.2"],
            ["blocks.1"],
            ["blocks.0"],
            ["conv_stem", "bn1", "act1"],                  # 最浅输入端
        ]
        backbone_base_lr = head_lr * backbone_lr_scale

        groups: list = []
        # head 参数(mil_pooling + classifier)使用最高 lr
        groups.append({
            "params": (
                list(self.mil_pooling.parameters())
                + list(self.classifier.parameters())
            ),
            "lr": head_lr,
            "name": "head",
        })

        # backbone 分层参数，按深度分组
        assigned: set = set()
        for depth_idx, prefixes in enumerate(_backbone_layer_prefixes):
            lr = backbone_base_lr * (eta ** depth_idx)
            params = []
            for name, param in self.backbone.named_parameters():
                if id(param) not in assigned and any(
                    name.startswith(pfx) for pfx in prefixes
                ):
                    params.append(param)
                    assigned.add(id(param))
            if params:
                groups.append({
                    "params": params,
                    "lr": lr,
                    "name": f"backbone_d{depth_idx}",
                })

        # 兜底：未被前面分组覆盖的 backbone 参数（防漏）
        remaining = [
            p for _, p in self.backbone.named_parameters()
            if id(p) not in assigned
        ]
        if remaining:
            deepest_lr = backbone_base_lr * (eta ** (len(_backbone_layer_prefixes) - 1))
            groups.append({
                "params": remaining,
                "lr": deepest_lr,
                "name": "backbone_remaining",
            })

        return groups
