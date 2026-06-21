import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from src.models.physical_prior import DifferentiablePhysicalPriorExtractor
from src.models.cbm_head import ConceptEmbeddingCBMHead, VariationalCLUBMean


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


class MultiTaskThyroidModel(nn.Module):
    """
    基于双分支特征解耦与可微物理先验注入的2D甲状腺超声多任务一体化诊断系统
    """

    def __init__(self, backbone_name="efficientnet_b2", pretrained=True, config=None):
        super().__init__()

        # 提取图像尺寸 (默认 512)
        self.img_size = 512
        if config is not None and "data" in config:
            self.img_size = config["data"].get("img_size", 512)

        self.current_epoch = 120  # 默认设置为较大值，使评估或默认前向时物理特征全激活

        # 1. 创建单 full_backbone 并加载离线预训练权重 (避免 aliasing 破坏优化器/梯度)
        local_pretrained_path = "checkpoints/efficientnet_b2_pretrained.safetensors"
        full_backbone = None

        if (
            backbone_name == "efficientnet_b2"
            and pretrained
            and os.path.exists(local_pretrained_path)
        ):
            print(
                f"--> Found offline local backbone checkpoint: {local_pretrained_path}. Loading weights locally."
            )
            full_backbone = timm.create_model(
                backbone_name, pretrained=False, features_only=False, in_chans=1
            )
            try:
                from safetensors.torch import load_file

                state_dict = load_file(local_pretrained_path)
                if "conv_stem.weight" in state_dict:
                    stem_weight = state_dict["conv_stem.weight"]
                    if stem_weight.shape[1] == 3:
                        state_dict["conv_stem.weight"] = stem_weight.sum(
                            dim=1, keepdim=True
                        )
                full_backbone.load_state_dict(state_dict, strict=False)
                print(
                    "--> Successfully loaded offline local backbone pretrained weights."
                )
            except Exception as e:
                print(
                    f"[WARNING] Failed to load offline backbone weights: {e}. Falling back to default timm initialization."
                )
                full_backbone = None

        if full_backbone is None:
            full_backbone = timm.create_model(
                backbone_name, pretrained=pretrained, features_only=False, in_chans=1
            )

        # 2. 提取低级特征共享底座 (Stage 0 - Stage 3)
        stem_modules = [full_backbone.conv_stem, full_backbone.bn1]
        if hasattr(full_backbone, "act1"):
            stem_modules.append(full_backbone.act1)
        elif hasattr(full_backbone, "act"):
            stem_modules.append(full_backbone.act)
        self.shared_stem = nn.Sequential(*stem_modules)
        self.shared_blocks = nn.Sequential(
            *[full_backbone.blocks[i] for i in range(5)]
        )  # Stage 0-4

        # 3. 独立构建分割专属分支 (Stage 5)
        self.seg_stage5 = nn.Sequential(
            full_backbone.blocks[5], full_backbone.blocks[6]
        )

        # 4. 通过 copy.deepcopy 独立构建分类专属分支 (Stage 5)
        self.cls_stage5 = nn.Sequential(
            copy.deepcopy(full_backbone.blocks[5]),
            copy.deepcopy(full_backbone.blocks[6]),
        )

        # 5. U-Net 堆叠上采样对齐解码器
        self.up5 = nn.ConvTranspose2d(352, 120, kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            nn.Conv2d(240, 120, 3, padding=1),
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.ConvTranspose2d(120, 48, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(96, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(48, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        )

        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)

        # 6. 可微物理先验流注入模块
        self.prior_extractor = DifferentiablePhysicalPriorExtractor(
            img_size=self.img_size
        )
        self.prior_mlp = nn.Sequential(
            nn.Linear(3, 16), nn.LayerNorm(16), nn.ReLU(inplace=True), nn.Linear(16, 16)
        )

        # 7. 概念瓶颈架构 (CBM) 分类头层设计与 vCLUB 估计器
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cbm_head = ConceptEmbeddingCBMHead(in_features=352, num_concepts=18)
        self.vclub = VariationalCLUBMean(x_dim=352, y_dim=18, hidden_size=64)

        # 8. 有监督对比学习投影头 (SupCon Head) - 输入映射变更为 352 维全局平均池化特征
        self.supcon_projector = nn.Sequential(
            nn.Linear(352, 128), nn.ReLU(inplace=True), nn.Linear(128, 128)
        )

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size and W == self.img_size
        ), f"Input shape mismatch: expected {self.img_size}x{self.img_size}, got {H}x{W}"

        # 1. 共享基础编码特征提取 (Stage 0 - Stage 3)
        # 1. 共享基础编码特征提取 (Stage 0 - Stage 4)
        feat = self.shared_stem(x)
        f_s0 = self.shared_blocks[0](feat)
        f_s1 = self.shared_blocks[1](f_s0)
        f_s2 = self.shared_blocks[2](f_s1)
        f_temp3 = self.shared_blocks[3](f_s2)
        f_s3 = self.shared_blocks[4](f_temp3)

        # 2. 分割编码专用流演进 (Seg-Encoder Stage 5)
        f_seg5 = self.seg_stage5(f_s3)

        # 3. 分割解码重组 (U-Net Dense Reconstruction)
        d5 = self.dec5(torch.cat([self.up5(f_seg5), f_s3], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), f_s2], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), f_s1], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), f_s0], dim=1))
        d1 = self.dec1(self.up1(d2))
        logits_mask = self.seg_head(d1)
        soft_mask = torch.sigmoid(logits_mask)

        # 4. 可微物理先验主动估计与提取
        v_phys = self.prior_extractor(soft_mask, x)  # [B, 3]
        e_prior_raw = self.prior_mlp(v_phys)  # [B, 16]

        # 动态时间退火门控：前 20 轮全关闭，之后 20 轮渐进式解冻，Epoch 40 全激活以防噪声污染（训练与评估须保持一致以防分布偏移）
        gamma = max(0.0, min(1.0, (self.current_epoch - 20) / 20.0))
        e_prior = e_prior_raw * gamma

        # 5. 分类编码专用流演进 (Cls-Encoder Stage 5)
        f_cls5 = self.cls_stage5(f_s3)
        f_cls_pooled = self.cls_pool(f_cls5).view(B, -1)  # [B, 352]

        # 6. 计算双极 CBM 概率及其概念分布与先验特征拼接
        (
            logits_comp,
            logits_echo,
            logits_shape,
            logits_margin,
            logits_foci,
            concept_vector,
            logits_malignancy,
            W_norm,
            latent_res,
        ) = self.cbm_head(f_cls_pooled, e_prior)

        z_contrastive = self.supcon_projector(f_cls_pooled)  # [B, 128]

        outputs = {
            "mask": logits_mask,  # 注意这里必须放 logits_mask，否则 criterion 内计算 BCE/Dice 会崩
            "soft_mask": soft_mask,  # 同时也把 soft_mask 放进去，方便其他地方读取
            "composition": logits_comp,
            "echogenicity": logits_echo,
            "shape": logits_shape,
            "margin": logits_margin,
            "foci": logits_foci,
            "concept_vector": concept_vector,
            "malignancy": logits_malignancy,
            "W_norm": W_norm,
            "latent_res": latent_res,
            "z_latent": f_cls_pooled,
            "z_contrastive": z_contrastive,
            "physical_vector": v_phys,
            "spin_l1": self.cbm_head.get_l1_penalty(),
        }

        return outputs
