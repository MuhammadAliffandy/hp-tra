"""
Loss Functions & Geodesic Rotation Metrics for Parametric HMR
Iterasi 2: True 3D Volumetric Mesh & Pose Supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VolumetricHMRLoss(nn.Module):
    def __init__(self, lambda_pose: float = 10.0, lambda_shape: float = 5.0, lambda_cam: float = 1.0):
        super().__init__()
        self.lambda_pose = lambda_pose
        self.lambda_shape = lambda_shape
        self.lambda_cam = lambda_cam

    def forward(self, pred_rotmat: torch.Tensor, pred_shape: torch.Tensor, pred_cam: torch.Tensor,
                gt_rotmat: torch.Tensor, gt_shape: torch.Tensor, gt_cam: torch.Tensor):
        """
        pred_rotmat: [B, 24, 3, 3]
        pred_shape : [B, 10]
        pred_cam   : [B, 3]
        gt_rotmat  : [B, 24, 3, 3]
        gt_shape   : [B, 10]
        gt_cam     : [B, 3]
        """
        # 1. Pose Rotation Loss (Chordal Frobenius Distance in SO(3))
        loss_pose = F.mse_loss(pred_rotmat, gt_rotmat)

        # 2. Shape / Volume Beta Loss (Smooth L1)
        loss_shape = F.smooth_l1_loss(pred_shape, gt_shape, beta=0.1)

        # 3. Weak-Perspective Camera Loss
        loss_cam = F.smooth_l1_loss(pred_cam, gt_cam, beta=0.1)

        total_loss = (self.lambda_pose * loss_pose + 
                      self.lambda_shape * loss_shape + 
                      self.lambda_cam * loss_cam)

        return total_loss, {
            "loss_pose": loss_pose.item(),
            "loss_shape": loss_shape.item(),
            "loss_cam": loss_cam.item(),
            "loss_total": total_loss.item()
        }

def compute_geodesic_angle_error(pred_rotmat: torch.Tensor, gt_rotmat: torch.Tensor) -> float:
    """
    Computes mean geodesic angular error (in degrees) across all 24 joints.
    Formula: theta = arccos((Tr(R_pred * R_gt^T) - 1) / 2)
    Input: [B, 24, 3, 3]
    """
    with torch.no_grad():
        # R_diff: [B, 24, 3, 3]
        R_diff = torch.matmul(pred_rotmat, gt_rotmat.transpose(-1, -2))
        # Trace of R_diff along last two dimensions: [B, 24]
        trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
        cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
        angle_rad = torch.acos(cos_theta)
        angle_deg = torch.rad2deg(angle_rad)
        return float(torch.mean(angle_deg).item())

if __name__ == "__main__":
    criterion = VolumetricHMRLoss()
    B = 4
    pred_R = torch.eye(3).view(1, 1, 3, 3).repeat(B, 24, 1, 1)
    gt_R = torch.eye(3).view(1, 1, 3, 3).repeat(B, 24, 1, 1)
    pred_s = torch.randn(B, 10)
    gt_s = torch.randn(B, 10)
    pred_c = torch.ones(B, 3)
    gt_c = torch.ones(B, 3)

    loss, details = criterion(pred_R, pred_s, pred_c, gt_R, gt_s, gt_c)
    err_deg = compute_geodesic_angle_error(pred_R, gt_R)
    print("Loss Calculation:")
    print("  Details:", details)
    print(f"  Geodesic Error: {err_deg:.2f} degrees (expected ~0 for identical matrices)")
    assert err_deg < 1e-4
    print("✅ Loss & Metrics computation verified!")
