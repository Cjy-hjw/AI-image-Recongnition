import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ConvBlock(nn.Module):
    """Basic double convolution block for U-Net style decoder."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class GatedTaskAttention(nn.Module):
    """
    Gated Task Attention (GTA): Decouples spatial/channel details of features
    before feeding into the classification CBM head.
    """
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = max(1, in_channels // 4)
        self.conv_reduce = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(mid_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.conv_project = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        feat_reduced = self.conv_reduce(x)
        c_weight = self.channel_gate(feat_reduced)
        s_weight = self.spatial_gate(feat_reduced)
        
        decoupled = x * c_weight * s_weight
        return self.conv_project(decoupled)

class LatentResidualCBMHead(nn.Module):
    """
    Latent-Residual Concept Bottleneck Model (CBM) Head.
    Combines direct linear clinical pathways and latent non-linear MLP pathways.
    """
    def __init__(self, in_features, num_concepts=18):
        super().__init__()
        # 1. Concept prediction heads (logits)
        self.head_composition = nn.Linear(in_features, 4)
        self.head_echogenicity = nn.Linear(in_features, 4)
        self.head_shape = nn.Linear(in_features, 2)
        self.head_margin = nn.Linear(in_features, 4)
        self.head_foci = nn.Linear(in_features, 4)
        
        # 2. Concept attention加权网络 (CoAt)
        self.concept_attention = nn.Sequential(
            nn.Linear(num_concepts, num_concepts),
            nn.ReLU(),
            nn.Linear(num_concepts, num_concepts),
            nn.Sigmoid()
        )
        
        # 3. 辅助非线性关系提取器 (Latent MLP)
        self.latent_mlp = nn.Sequential(
            nn.Linear(num_concepts, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # 4. 混合分类器 (2 classes for compatibility with cross-entropy evaluation)
        self.classifier_direct = nn.Linear(num_concepts, 2, bias=False)
        self.classifier_latent = nn.Linear(16, 2)

    def forward(self, x):
        # Predict individual concept logits
        logits_comp = self.head_composition(x)
        logits_echo = self.head_echogenicity(x)
        logits_shape = self.head_shape(x)
        logits_margin = self.head_margin(x)
        logits_foci = self.head_foci(x)
        
        # Compute probabilities for the bottleneck representation
        prob_comp = F.softmax(logits_comp, dim=1)
        prob_echo = F.softmax(logits_echo, dim=1)
        prob_shape = F.softmax(logits_shape, dim=1)
        prob_margin = F.softmax(logits_margin, dim=1)
        prob_foci = torch.sigmoid(logits_foci)
        
        # Concat to get 18-dim concept probability vector
        concept_vector = torch.cat([
            prob_comp,
            prob_echo,
            prob_shape,
            prob_margin,
            prob_foci
        ], dim=1)
        
        # Concept-wise Attention (CoAt)
        attn_weights = self.concept_attention(concept_vector)
        weighted_concepts = concept_vector * attn_weights
        
        # Extract latent complex interactions
        latent_features = self.latent_mlp(weighted_concepts)
        
        # Residual fusion to predict malignancy logits (2 classes)
        logits_malignancy = self.classifier_direct(weighted_concepts) + self.classifier_latent(latent_features)
        
        return logits_comp, logits_echo, logits_shape, logits_margin, logits_foci, concept_vector, logits_malignancy

class MultiTaskThyroidModel(nn.Module):
    """
    Multi-task Thyroid AI Model.
    - Backbone: 2D timm encoder.
    - Decoder: U-Net style decoder for pixel-level segmentation.
    - Classification: Concept Bottleneck Model (CBM) for TI-RADS and malignancy classification.
    """
    def __init__(self, backbone_name="resnet18", pretrained=True, config=None):
        super().__init__()
        
        # Load feature extraction backbone (in_chans=1 for B-mode ultrasound)
        self.encoder = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            features_only=True, 
            in_chans=1
        )
        encoder_channels = self.encoder.feature_info.channels()
        
        # U-Net style decoder for segmentation
        self.up_blocks = nn.ModuleList()
        in_ch = encoder_channels[-1]
        
        for i in reversed(range(len(encoder_channels) - 1)):
            skip_ch = encoder_channels[i]
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2),
                    nn.Conv2d(in_ch // 2 + skip_ch, in_ch // 2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_ch // 2),
                    nn.ReLU(inplace=True)
                )
            )
            in_ch = in_ch // 2
            
        # Final segmentation output head (upsample from 1/2 back to full resolution)
        self.final_up = nn.ConvTranspose2d(in_ch, 32, kernel_size=2, stride=2)
        self.seg_head = nn.Conv2d(32, 1, kernel_size=1)
        
        # Classification & CBM decoupling gateway
        self.decouple_gate = GatedTaskAttention(encoder_channels[-1])
        self.cbm_head = LatentResidualCBMHead(in_features=encoder_channels[-1], num_concepts=18)
        
    def forward(self, x):
        # 1. Encoder (forward through backbone layers)
        features = self.encoder(x)
        deepest_feat = features[-1]
        
        # 2. Decoder (segmentation mask)
        seg_feat = deepest_feat
        idx = 0
        for i in reversed(range(len(features) - 1)):
            skip = features[i]
            seg_feat = self.up_blocks[idx][0](seg_feat)
            if seg_feat.shape[2:] != skip.shape[2:]:
                seg_feat = F.interpolate(seg_feat, size=skip.shape[2:], mode='bilinear', align_corners=True)
            seg_feat = torch.cat([seg_feat, skip], dim=1)
            seg_feat = self.up_blocks[idx][1:](seg_feat)
            idx += 1
            
        seg_feat = self.final_up(seg_feat)
        if seg_feat.shape[2:] != x.shape[2:]:
            seg_feat = F.interpolate(seg_feat, size=x.shape[2:], mode='bilinear', align_corners=True)
        mask_out = self.seg_head(seg_feat) # Raw logits
        
        # 3. Classification / CBM决策路径
        decoupled_feat = self.decouple_gate(deepest_feat)
        gap = F.adaptive_avg_pool2d(decoupled_feat, (1, 1)).view(decoupled_feat.size(0), -1)
        
        logits_comp, logits_echo, logits_shape, logits_margin, logits_foci, concept_vector, logits_malignancy = self.cbm_head(gap)
        
        outputs = {
            "mask": mask_out,
            "composition": logits_comp,
            "echogenicity": logits_echo,
            "shape": logits_shape,
            "margin": logits_margin,
            "foci": logits_foci,
            "concept_vector": concept_vector,
            "malignancy": logits_malignancy
        }
        
        return outputs
