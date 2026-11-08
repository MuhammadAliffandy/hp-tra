"""
Parametric Volumetric Human Mesh Recovery (HMR) Network
Iterasi 2: True 3D Volumetric Mannequin Architecture

Formulation:
Instead of predicting flat 2D lines, the network predicts:
1. Shape Parameters (beta in R^10): Physical body volume, muscle thickness, and proportions.
2. 3D Joint Rotations (theta in R^{24 x 3 x 3}): Continuous 6D representations projected to SO(3).
3. Weak-Perspective Camera (c in R^3): Scale and 2D translation [s, tx, ty].
"""

import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D continuous rotation representation to 3x3 orthogonal rotation matrices.
    Reference: Zhou et al., 'On the Continuity of Rotation Representations in Neural Networks' (CVPR 2019).
    Input:  [..., 6]
    Output: [..., 3, 3] in SO(3)
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    dot = torch.sum(b1 * a2, dim=-1, keepdim=True)
    b2 = F.normalize(a2 - dot * b1, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)

def rotmat_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """
    Extracts 6D representation (first two columns) from a 3x3 rotation matrix.
    Input:  [..., 3, 3]
    Output: [..., 6]
    """
    col1 = R[..., :, 0]
    col2 = R[..., :, 1]
    return torch.cat([col1, col2], dim=-1)

class VolumetricHMRNet(nn.Module):
    def __init__(self, backbone_name: str = "mobilenet_v3_large", pretrained_base: Optional[str] = None):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "mobilenet_v3_large":
            base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            self.features = base.features
            self.avgpool = base.avgpool
            feat_dim = 960
        elif backbone_name == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.features = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            feat_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Optional: Load pretrained weights from Iterasi 1 (trained on 114k COCO images)
        if pretrained_base and os.path.exists(pretrained_base):
            print(f"📦 Transfer Learning: Loading Iterasi 1 pretrained backbone from {pretrained_base}...")
            ckpt = torch.load(pretrained_base, map_location="cpu")
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            backbone_dict = {}
            for k, v in state_dict.items():
                if k.startswith("backbone.features."):
                    backbone_dict[k.replace("backbone.features.", "")] = v
                elif k.startswith("features."):
                    backbone_dict[k.replace("features.", "")] = v
            if backbone_dict:
                self.features.load_state_dict(backbone_dict, strict=False)
                print(f"✅ Loaded {len(backbone_dict)} convolutional layers from Iterasi 1!")

        # Parametric Head Regressors (SMPL: 24 joints x 6D rotation + 10 shape beta + 3 camera)
        self.fc_shared = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        # 1. Pose: 24 joints * 6 continuous parameters = 144 outputs
        self.fc_pose = nn.Linear(512, 24 * 6)
        
        # 2. Shape (Volume / Muscle beta): 10 PCA parameters
        self.fc_shape = nn.Linear(512, 10)
        
        # 3. Weak-Perspective Camera: scale (1) + translation (2) = 3 outputs
        self.fc_cam = nn.Linear(512, 3)

        # Initialize pose to identity rotations and camera scale to 0.9
        self._init_heads()

    def _init_heads(self):
        # Identity 6D rotation is [1, 0, 0, 0, 1, 0]
        init_rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32).repeat(24)
        nn.init.zeros_(self.fc_pose.weight)
        self.fc_pose.bias.data.copy_(init_rot6d)

        nn.init.zeros_(self.fc_shape.weight)
        nn.init.zeros_(self.fc_shape.bias)

        nn.init.zeros_(self.fc_cam.weight)
        # Standard camera: scale ~ 1.0, translation [0, 0]
        self.fc_cam.bias.data.copy_(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat_flat = torch.flatten(feat, 1)

        shared = self.fc_shared(feat_flat)

        pred_rot6d = self.fc_pose(shared).view(-1, 24, 6)
        pred_rotmat = rot6d_to_rotmat(pred_rot6d) # [B, 24, 3, 3]

        pred_shape = self.fc_shape(shared)       # [B, 10]
        pred_cam = self.fc_cam(shared)           # [B, 3]

        return pred_rotmat, pred_shape, pred_cam, pred_rot6d

if __name__ == "__main__":
    net = VolumetricHMRNet()
    dummy = torch.randn(2, 3, 256, 256)
    rotmat, shape, cam, rot6d = net(dummy)
    print("Test Forward Pass:")
    print("  rotmat:", rotmat.shape) # [2, 24, 3, 3]
    print("  shape :", shape.shape)  # [2, 10]
    print("  cam   :", cam.shape)    # [2, 3]
    print("✅ Model instantiation & forward pass successful!")
