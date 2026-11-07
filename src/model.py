"""
Multi-Task Volumetric PoseNet Architecture.
Direct Coordinate & Volumetric Regression with Pretrained MobileNetV3 / ResNet Backbone.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MultiTaskVolumetricPoseNet(nn.Module):
    """
    Ultra-compact, low-latency neural network that predicts:
      1. 17 Normalized Joint Coordinates [B, 17, 2] in range [0, 1] (Skeleton Mode)
      2. 6 Limb Volume Radii [B, 6] in range [0, 1] (Volume Mode)
      3. 17 Joint Visibility Logits [B, 17]
    """
    def __init__(self, backbone_name: str = "mobilenet_v3_large", pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.backbone = nn.Sequential(*list(base.children())[:-2]) # [B, 512, 8, 8]
            feat_dim = 512
        elif backbone_name == "mobilenet_v3_large":
            base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
            self.backbone = base.features # [B, 960, 8, 8]
            feat_dim = 960
        elif backbone_name == "mobilenet_v3_small":
            base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
            self.backbone = base.features # [B, 576, 8, 8]
            feat_dim = 576
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose 'mobilenet_v3_large', 'mobilenet_v3_small', or 'resnet18'.")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Shared Latent Representation
        self.latent_mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # Head 1: Kinematic Joint Coordinates (17 * 2 = 34 values)
        self.coord_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 17 * 2),
            nn.Sigmoid() # Strictly bounds predictions to [0.0, 1.0] image crop space
        )

        # Head 2: Volumetric Limb Parameters (6 radii: head, torso, uarm, farm, thigh, calf)
        self.volume_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 6),
            nn.Sigmoid() # Strictly bounds radii to [0.0, 1.0] crop fraction
        )

        # Head 3: Keypoint Visibility Logits (17 logits)
        self.vis_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 17)
        )

    def forward(self, x: torch.Tensor):
        # x: [B, 3, 256, 256]
        feats = self.backbone(x) # [B, feat_dim, 8, 8]
        pooled = self.pool(feats).flatten(1) # [B, feat_dim]
        latent = self.latent_mlp(pooled) # [B, 512]

        coords = self.coord_head(latent).view(-1, 17, 2) # [B, 17, 2] in [0, 1]
        volume = self.volume_head(latent)                # [B, 6] in [0, 1]
        vis_logits = self.vis_head(latent)              # [B, 17]

        return coords, volume, vis_logits
