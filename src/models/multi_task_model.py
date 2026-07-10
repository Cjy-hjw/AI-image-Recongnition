from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.models.physical_prior import DifferentiablePhysicalPriorExtractor
from src.models.cbm_head import ConceptEmbeddingCBMHead, VariationalCLUBMean
from src.models.pretrained_loader import load_local_safetensors


class ModelOutputs(TypedDict):
    """Single source of truth for MultiTaskThyroidModel.forward() output keys.

    Downstream consumers (train_step, validate_step, vclub, losses, evaluate,
    predict, postprocess, export) all index these keys; renaming any key here
    flags every consumer site.
    """

    mask: torch.Tensor
    composition: torch.Tensor
    echogenicity: torch.Tensor
    shape: torch.Tensor
    margin: torch.Tensor
    foci: torch.Tensor
    malignancy: torch.Tensor
    concept_vector: torch.Tensor
    latent_res: torch.Tensor
    z_latent: torch.Tensor
    z_contrastive: torch.Tensor
    physical_vector: torch.Tensor
    spin_l1: torch.Tensor


class ConvBlock(nn.Module):
    """Basic double convolution block for U-Net style decoder."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class TaskAlignmentAdaptor(nn.Module):
    def __init__(self, seg_in_channels, cls_in_channels):
        super().__init__()
        self.channel_projector = nn.Sequential(
            nn.Conv2d(seg_in_channels, cls_in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cls_in_channels),
            nn.ReLU(inplace=True),
        )
        self.attention_generator = nn.Sequential(
            nn.Conv2d(
                cls_in_channels,
                cls_in_channels,
                kernel_size=3,
                padding=1,
                groups=cls_in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(cls_in_channels),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f_seg_detached, f_cls):
        if f_seg_detached.shape[-2:] != f_cls.shape[-2:]:
            f_seg_detached = F.interpolate(
                f_seg_detached,
                size=f_cls.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        f_seg_detached = f_seg_detached.float()
        f_cls_float = f_cls.float()
        projected = self.channel_projector(f_seg_detached)
        attention_mask = self.attention_generator(projected)
        out = f_cls_float + self.gamma * (f_cls_float * attention_mask)
        return out.to(dtype=f_cls.dtype)


class MultiTaskThyroidModel(nn.Module):
    """
    基于双分支特征解耦与可微物理先验注入的2D甲状腺超声多任务一体化诊断系统
    """

    CLS_STAGE3_CHANNELS = 88
    CLS_STAGE4_CHANNELS = 208

    STAGE1_FROZEN_CBM_SUBMODULES = ("classifier_direct", "latent_mlp")
    STAGE3_FROZEN_PREFIXES = (
        "seg_backbone",
        "cls_backbone",
        "up_stages",
        "dec_stages",
        "seg_head",
        "prior_extractor",
        "prior_mlp",
        "supcon_projector",
        "adaptor_stage3",
        "adaptor_stage4",
    )

    def __init__(
        self,
        backbone_name="efficientnet_b2",
        pretrained=True,
        config=None,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()

        self.img_size = 512
        if config is not None and "data" in config:
            self.img_size = config["data"].get("img_size", 512)

        self.current_epoch = 120

        self.seg_backbone = timm.create_model(
            "convnext_nano",
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=1,
        )
        if pretrained:
            seg_pretrained_path = "checkpoints/convnext_nano_pretrained.safetensors"
            loaded = load_local_safetensors(
                self.seg_backbone,
                seg_pretrained_path,
                in_chans=1,
                stem_key="stem.0.weight",
            )
            if not loaded:
                print(
                    "--> Local ConvNeXt-Nano weights not found; falling back to timm online pretrained."
                )
                online_seg = timm.create_model(
                    "convnext_nano",
                    pretrained=True,
                    features_only=True,
                    out_indices=(0, 1, 2, 3),
                    in_chans=1,
                )
                self.seg_backbone.load_state_dict(online_seg.state_dict(), strict=False)

        self.cls_backbone = timm.create_model(
            backbone_name, pretrained=False, features_only=False, in_chans=1
        )
        if backbone_name == "efficientnet_b2" and pretrained:
            load_local_safetensors(
                self.cls_backbone,
                "checkpoints/efficientnet_b2_pretrained.safetensors",
                stem_key="conv_stem.weight",
            )

        # LoRA 注入：在预训练权重加载完成后，替换目标 1x1 Conv2d 层
        if use_lora:
            from src.models.lora_utils import apply_lora
            n_cls = apply_lora(self.cls_backbone, r=lora_r, alpha=lora_alpha)
            n_seg = apply_lora(self.seg_backbone, r=lora_r, alpha=lora_alpha)
            print(f"--> LoRA 注入: cls_backbone {n_cls} 层, seg_backbone {n_seg} 层 (r={lora_r}, alpha={lora_alpha})")
            # 冻结所有参数（含 LoRA A/B）：模型以"LoRA 未激活"状态构建，
            # 训练脚本在 FT 阶段显式解冻 lora_A/lora_B（见 train_thyroidxl.py）。
            # 若不冻结 LoRA 参数，LP 阶段它们不在 optimizer 内、梯度永不清零，
            # 会跨 batch 累积直至数值溢出（§1.1 教训：LP/FT 参数集合必须与 requires_grad 严格对齐）。
            for p in self.cls_backbone.parameters():
                p.requires_grad = False
            for p in self.seg_backbone.parameters():
                p.requires_grad = False

        seg_channels = list(self.seg_backbone.feature_info.channels())
        self.seg_stage_channels = seg_channels
        self.cls_stage3_channels = self.CLS_STAGE3_CHANNELS
        self.cls_stage4_channels = self.CLS_STAGE4_CHANNELS
        self.adaptor_stage3 = TaskAlignmentAdaptor(
            seg_channels[1], self.cls_stage3_channels
        )
        self.adaptor_stage4 = TaskAlignmentAdaptor(
            seg_channels[2], self.cls_stage4_channels
        )

        self.up_stages = nn.ModuleList(
            [
                nn.ConvTranspose2d(seg_channels[3], 320, kernel_size=2, stride=2),
                nn.ConvTranspose2d(320, 160, kernel_size=2, stride=2),
                nn.ConvTranspose2d(160, 80, kernel_size=2, stride=2),
                nn.ConvTranspose2d(80, 40, kernel_size=2, stride=2),
                nn.ConvTranspose2d(40, 20, kernel_size=2, stride=2),
            ]
        )
        self.dec_stages = nn.ModuleList(
            [
                ConvBlock(320 + seg_channels[2], 320),
                ConvBlock(160 + seg_channels[1], 160),
                ConvBlock(80 + seg_channels[0], 80),
                ConvBlock(40, 40),
                ConvBlock(20, 20),
            ]
        )
        self.seg_head = nn.Conv2d(20, 1, kernel_size=1)

        self.prior_extractor = DifferentiablePhysicalPriorExtractor(
            img_size=self.img_size
        )
        self.prior_mlp = nn.Sequential(
            nn.Linear(3, 16), nn.LayerNorm(16), nn.ReLU(inplace=True), nn.Linear(16, 16)
        )

        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        # 分类分支 pooled 特征维度(EfficientNet-B2 最后 block 输出通道)与概念总维,
        # cbm_head / vclub / supcon_projector 三者必须共用同一 in_features,否则静默错配。
        self.CLS_POOLED_DIM = 352
        self.NUM_CONCEPTS = 18
        self.cbm_head = ConceptEmbeddingCBMHead(
            in_features=self.CLS_POOLED_DIM, num_concepts=self.NUM_CONCEPTS
        )
        self.vclub = VariationalCLUBMean(
            x_dim=self.CLS_POOLED_DIM, y_dim=self.NUM_CONCEPTS, hidden_size=64
        )
        self.supcon_projector = nn.Sequential(
            nn.Linear(self.CLS_POOLED_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

    def is_param_trainable(self, name: str, stage: int) -> bool:
        if "vclub" in name:
            return True
        if stage == 1:
            return not any(sub in name for sub in self.STAGE1_FROZEN_CBM_SUBMODULES)
        if stage == 2:
            return True
        return not any(name.startswith(pfx) for pfx in self.STAGE3_FROZEN_PREFIXES)

    def set_training_stage(self, stage: int) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = self.is_param_trainable(name, stage)

    def _forward_cls_backbone(self, x, seg_features):
        """EfficientNet-B2 classification branch with task-alignment injection."""
        B = x.shape[0]
        x_cls = self.cls_backbone.conv_stem(x)
        x_cls = self.cls_backbone.bn1(x_cls)
        x_cls = self.cls_backbone.blocks[0](x_cls)
        x_cls = self.cls_backbone.blocks[1](x_cls)
        x_cls = self.cls_backbone.blocks[2](x_cls)
        x_cls = self.cls_backbone.blocks[3](x_cls)
        x_cls = self.adaptor_stage3(seg_features[1].detach(), x_cls)
        x_cls = self.cls_backbone.blocks[4](x_cls)
        x_cls = self.cls_backbone.blocks[5](x_cls)
        x_cls = self.adaptor_stage4(seg_features[2].detach(), x_cls)
        x_cls = self.cls_backbone.blocks[6](x_cls)
        return self.cls_pool(x_cls).view(B, -1)

    def forward(self, x, gamma_prior=1.0) -> ModelOutputs:
        B, C, H, W = x.shape
        assert (
            H == self.img_size and W == self.img_size
        ), f"Input shape mismatch: expected {self.img_size}x{self.img_size}, got {H}x{W}"

        seg_features = self.seg_backbone(x)
        d = self.up_stages[0](seg_features[-1])
        d = self.dec_stages[0](torch.cat([d, seg_features[2]], dim=1))
        d = self.up_stages[1](d)
        d = self.dec_stages[1](torch.cat([d, seg_features[1]], dim=1))
        d = self.up_stages[2](d)
        d = self.dec_stages[2](torch.cat([d, seg_features[0]], dim=1))
        d = self.up_stages[3](d)
        d = self.dec_stages[3](d)
        d = self.up_stages[4](d)
        d = self.dec_stages[4](d)
        logits_mask = self.seg_head(d)
        soft_mask = torch.sigmoid(logits_mask)

        v_phys = self.prior_extractor(soft_mask, x)
        e_prior = self.prior_mlp(v_phys) * gamma_prior

        x_cls = self._forward_cls_backbone(x, seg_features)
        assert x_cls.shape[1] == self.CLS_POOLED_DIM, (
            f"cls pooled dim mismatch: expected {self.CLS_POOLED_DIM}, "
            f"got {x_cls.shape[1]} — cbm_head/vclub/supcon in_features will misalign"
        )

        cbm = self.cbm_head(x_cls, e_prior)
        z_contrastive = self.supcon_projector(x_cls)

        outputs = {
            "mask": logits_mask,
            "composition": cbm.logits_comp,
            "echogenicity": cbm.logits_echo,
            "shape": cbm.logits_shape,
            "margin": cbm.logits_margin,
            "foci": cbm.logits_foci,
            "malignancy": cbm.logits_malignancy,
            "concept_vector": cbm.concept_vector,
            "latent_res": cbm.latent_res,
            "z_latent": x_cls,
            "z_contrastive": z_contrastive,
            "physical_vector": v_phys,
            "spin_l1": self.cbm_head.get_l1_penalty(),
        }

        return outputs
