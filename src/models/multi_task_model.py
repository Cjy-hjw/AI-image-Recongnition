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

class MultiTaskThyroidModel(nn.Module):
    """
    Multi-task Thyroid AI Model.
    - Backbone: 2D timm encoder.
    - Decoder: U-Net style decoder for pixel-level segmentation.
    - Classification: Concept Bottleneck Model (CBM) for TI-RADS and malignancy classification.
    """
    def __init__(self, backbone_name="resnet18", pretrained=True, config=None):
        super().__init__()
        
        # Load feature extraction backbone
        # e.g., resnet18 has feature channels: [64, 64, 128, 256, 512] at reductions [2, 4, 8, 16, 32]
        self.encoder = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            features_only=True, 
            in_chans=1
        )
        encoder_channels = self.encoder.feature_info.channels()
        
        # U-Net style decoder for segmentation
        # Input resolutions: 1/32, 1/16, 1/8, 1/4, 1/2 of original size
        self.up4 = nn.ConvTranspose2d(encoder_channels[4], encoder_channels[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(encoder_channels[3] * 2, encoder_channels[3])
        
        self.up3 = nn.ConvTranspose2d(encoder_channels[3], encoder_channels[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(encoder_channels[2] * 2, encoder_channels[2])
        
        self.up2 = nn.ConvTranspose2d(encoder_channels[2], encoder_channels[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(encoder_channels[1] * 2, encoder_channels[1])
        
        self.up1 = nn.ConvTranspose2d(encoder_channels[1], encoder_channels[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(encoder_channels[0] * 2, encoder_channels[0])
        
        # Final segmentation output head (upsample from 1/2 back to full resolution)
        self.final_up = nn.ConvTranspose2d(encoder_channels[0], 32, kernel_size=2, stride=2)
        self.segmentation_head = nn.Conv2d(32, 1, kernel_size=1)
        
        # Classification Heads (TI-RADS Concepts)
        latent_dim = encoder_channels[4]
        
        # 1. Composition head: 4 classes
        self.head_composition = nn.Linear(latent_dim, 4)
        # 2. Echogenicity head: 4 classes
        self.head_echogenicity = nn.Linear(latent_dim, 4)
        # 3. Shape head: 2 classes
        self.head_shape = nn.Linear(latent_dim, 2)
        # 4. Margin head: 4 classes
        self.head_margin = nn.Linear(latent_dim, 4)
        # 5. Foci head: 4 independent binary labels (multi-label)
        self.head_foci = nn.Linear(latent_dim, 4)
        
        # 6. Concept Bottleneck layer -> Malignancy
        # Total concept dimensions = 4 (comp) + 4 (echo) + 2 (shape) + 4 (margin) + 4 (foci) = 18
        self.concept_dim = 18
        self.head_malignancy = nn.Linear(self.concept_dim, 2)
        
    def forward(self, x):
        # 1. Encoder (forward through backbone layers)
        features = self.encoder(x)
        c1, c2, c3, c4, c5 = features # features at different scales
        
        # 2. Decoder (segmentation mask)
        d4 = self.up4(c5)
        # Pad or crop if spatial dimensions differ slightly
        if d4.shape[2:] != c4.shape[2:]:
            d4 = F.interpolate(d4, size=c4.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, c4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        if d3.shape[2:] != c3.shape[2:]:
            d3 = F.interpolate(d3, size=c3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, c3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape[2:] != c2.shape[2:]:
            d2 = F.interpolate(d2, size=c2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, c2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape[2:] != c1.shape[2:]:
            d1 = F.interpolate(d1, size=c1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, c1], dim=1)
        d1 = self.dec1(d1)
        
        # Upsample to input resolution
        mask_out = self.final_up(d1)
        if mask_out.shape[2:] != x.shape[2:]:
            mask_out = F.interpolate(mask_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        mask_out = self.segmentation_head(mask_out)
        
        # 3. Classification (Concept Bottleneck Model)
        # Global Average Pooling on the deepest feature map
        gap = F.adaptive_avg_pool2d(c5, (1, 1)).view(c5.size(0), -1)
        
        # Predict concepts (logits)
        logits_comp = self.head_composition(gap)
        logits_echo = self.head_echogenicity(gap)
        logits_shape = self.head_shape(gap)
        logits_margin = self.head_margin(gap)
        logits_foci = self.head_foci(gap)
        
        # Convert logits to probability distributions to pass through bottleneck
        prob_comp = F.softmax(logits_comp, dim=1)
        prob_echo = F.softmax(logits_echo, dim=1)
        prob_shape = F.softmax(logits_shape, dim=1)
        prob_margin = F.softmax(logits_margin, dim=1)
        prob_foci = torch.sigmoid(logits_foci) # Foci is multi-label
        
        # Concatenate concepts
        concept_vector = torch.cat([
            prob_comp, 
            prob_echo, 
            prob_shape, 
            prob_margin, 
            prob_foci
        ], dim=1)
        
        # Predict malignancy from concepts
        logits_malignancy = self.head_malignancy(concept_vector)
        
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
