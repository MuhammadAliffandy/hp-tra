"""
Production Parametric Human Mesh Recovery (HMR) Network with IEF (Iterative Error Feedback)
Architecture: MobileNetV3-Large Feature Extractor + Iterative Kinematic Regressor (3-Step Loop)
Output:
- 24 3D Rotation Matrices (SO(3)) via continuous 6D Gram-Schmidt projection
- 10 Shape parameters (beta)
- 3 Weak-perspective camera parameters [scale, tx, ty]
- 3D Joint positions via forward kinematics
- Projected 2D Keypoints for Reprojection Loss Supervision
"""

import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# SMPL 24 Kinematic Tree Parents
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# Base template offsets (in meters)
SMPL_TEMPLATE_OFFSETS = [
    [0.0, 0.0, 0.0],          # 0: Pelvis
    [0.09, -0.08, 0.0],       # 1: L_Hip
    [-0.09, -0.08, 0.0],      # 2: R_Hip
    [0.0, 0.12, 0.0],         # 3: Spine1
    [0.0, -0.38, 0.0],        # 4: L_Knee
    [0.0, -0.38, 0.0],        # 5: R_Knee
    [0.0, 0.14, 0.0],         # 6: Spine2
    [0.0, -0.38, 0.0],        # 7: L_Ankle
    [0.0, -0.38, 0.0],        # 8: R_Ankle
    [0.0, 0.14, 0.0],         # 9: Spine3
    [0.0, -0.08, 0.14],       # 10: L_Foot
    [0.0, -0.08, 0.14],       # 11: R_Foot
    [0.0, 0.12, 0.0],         # 12: Neck
    [0.08, 0.08, -0.02],      # 13: L_Collar
    [-0.08, 0.08, -0.02],     # 14: R_Collar
    [0.0, 0.18, 0.0],         # 15: Head
    [0.10, 0.0, 0.0],         # 16: L_Shoulder
    [-0.10, 0.0, 0.0],        # 17: R_Shoulder
    [0.0, -0.28, 0.0],        # 18: L_Elbow
    [0.0, -0.28, 0.0],        # 19: R_Elbow
    [0.0, -0.25, 0.0],        # 20: L_Wrist
    [0.0, -0.25, 0.0],        # 21: R_Wrist
    [0.0, -0.10, 0.0],        # 22: L_Hand
    [0.0, -0.10, 0.0],        # 23: R_Hand
]

def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D continuous rotation representation to 3x3 orthogonal rotation matrices.
    Reference: Zhou et al., 'On the Continuity of Rotation Representations in Neural Networks' (CVPR 2019).
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    dot = torch.sum(b1 * a2, dim=-1, keepdim=True)
    b2 = F.normalize(a2 - dot * b1, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)

def forward_kinematics_torch(rotmats: torch.Tensor, shape_beta: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """
    Differentiable forward kinematics in PyTorch without in-place operations.
    rotmats: [B, 24, 3, 3]
    shape_beta: [B, 10]
    offsets: [24, 3]
    Returns: joints_3d [B, 24, 3]
    """
    B = rotmats.shape[0]
    scale_y = 1.0 + 0.05 * shape_beta[:, 0:1] # [B, 1]
    scale_fat = 1.0 + 0.06 * shape_beta[:, 1:2] # [B, 1]

    # Non-inplace scaling of template offsets
    offsets_exp = offsets.unsqueeze(0).repeat(B, 1, 1) # [B, 24, 3]
    scaled_x = offsets_exp[:, :, 0] * scale_fat
    scaled_y = offsets_exp[:, :, 1] * scale_y
    scaled_z = offsets_exp[:, :, 2] * scale_fat
    scaled_offsets = torch.stack([scaled_x, scaled_y, scaled_z], dim=-1)

    global_rotmats = [None] * 24
    joints_3d = [None] * 24

    for i in range(24):
        p = SMPL_PARENTS[i]
        local_R = rotmats[:, i]
        if p == -1:
            global_rotmats[i] = local_R
            joints_3d[i] = scaled_offsets[:, i]
        else:
            p_rot = global_rotmats[p]
            curr_rot = torch.bmm(p_rot, local_R)
            global_rotmats[i] = curr_rot
            delta_pos = torch.bmm(p_rot, scaled_offsets[:, i].unsqueeze(-1)).squeeze(-1)
            joints_3d[i] = joints_3d[p] + delta_pos

    return torch.stack(joints_3d, dim=1)

def project_3d_to_2d(joints_3d: torch.Tensor, cam: torch.Tensor) -> torch.Tensor:
    """
    Weak-perspective projection: proj_2d = scale * J_3d[:, :2] + trans
    joints_3d: [B, 24, 3]
    cam: [B, 3] -> [scale, tx, ty]
    Returns: [B, 24, 2] in normalized image coordinates [-1, 1]
    """
    scale = cam[:, 0:1].unsqueeze(1) # [B, 1, 1]
    trans = cam[:, 1:3].unsqueeze(1) # [B, 1, 2]
    proj_2d = scale * joints_3d[:, :, :2] + trans
    return proj_2d

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

        # Transfer learning from Iterasi 1
        if pretrained_base and os.path.exists(pretrained_base):
            print(f"📦 Transfer Learning: Loading Iterasi 1 weights from {pretrained_base}...")
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
                print(f"✅ Loaded {len(backbone_dict)} layers from Iterasi 1!")

        # Iterative Error Feedback (IEF) Setup
        # Total parameters: 24*6 (144 pose) + 10 (shape) + 3 (camera) = 157
        self.n_param = 24 * 6 + 10 + 3
        self.fc_feat = nn.Sequential(
            nn.Linear(feat_dim + self.n_param, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.fc_delta = nn.Linear(1024, self.n_param)
        nn.init.xavier_uniform_(self.fc_delta.weight, gain=0.01)
        nn.init.zeros_(self.fc_delta.bias)

        # Neutral identity pose initialization [1, 0, 0, 0, 1, 0]
        init_pose_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32).repeat(24)
        init_shape = torch.zeros(10, dtype=torch.float32)
        init_cam = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

        init_params = torch.cat([init_pose_6d, init_shape, init_cam], dim=0) # [157]
        self.register_buffer("init_params", init_params)
        self.register_buffer("template_offsets", torch.tensor(SMPL_TEMPLATE_OFFSETS, dtype=torch.float32))

    def forward(self, x: torch.Tensor, n_iter: int = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, 3, 256, 256]
        n_iter: number of IEF iterative refinement loops (default 3)
        Returns:
          rotmats: [B, 24, 3, 3]
          shape:   [B, 10]
          cam:     [B, 3]
          joints_3d: [B, 24, 3]
          proj_2d: [B, 24, 2]
        """
        B = x.shape[0]
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat_flat = torch.flatten(feat, 1)

        curr_params = self.init_params.unsqueeze(0).repeat(B, 1)

        for _ in range(n_iter):
            inp = torch.cat([feat_flat, curr_params], dim=1)
            h = self.fc_feat(inp)
            delta = self.fc_delta(h)
            curr_params = curr_params + delta

        pose_6d = curr_params[:, :144].view(B, 24, 6)
        pred_shape = curr_params[:, 144:154]
        pred_cam = curr_params[:, 154:157]

        pred_rotmat = rot6d_to_rotmat(pose_6d)

        joints_3d = forward_kinematics_torch(pred_rotmat, pred_shape, self.template_offsets)
        proj_2d = project_3d_to_2d(joints_3d, pred_cam)

        return pred_rotmat, pred_shape, pred_cam, joints_3d, proj_2d

if __name__ == "__main__":
    net = VolumetricHMRNet()
    dummy = torch.randn(2, 3, 256, 256)
    rotmat, shape, cam, j3d, p2d = net(dummy)
    print("Test Forward Pass with IEF:")
    print("  rotmat   :", rotmat.shape)   # [2, 24, 3, 3]
    print("  shape    :", shape.shape)    # [2, 10]
    print("  cam      :", cam.shape)      # [2, 3]
    print("  joints_3d:", j3d.shape)      # [2, 24, 3]
    print("  proj_2d  :", p2d.shape)      # [2, 24, 2]
    print("✅ Model instantiation & IEF forward pass successful!")
