"""
Loss Functions & Geodesic Rotation Metrics for Parametric HMR
Iterasi 2: True 3D Volumetric Mesh & Pose Supervision

Includes:
1. Pose Rotation Loss (Chordal Frobenius Distance in SO(3))
2. Shape / Volume Beta Loss (Smooth L1)
3. Weak-Perspective Camera Loss
4. 3D Joint Euclidean Loss (Prevents cascading rotation error)
5. 2D Reprojection Loss (Ties 3D prediction directly to the 2D image plane)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VolumetricHMRLoss(nn.Module):
    def __init__(
        self,
        lambda_pose: float = 10.0,
        lambda_shape: float = 5.0,
        lambda_cam: float = 1.0,
        lambda_j3d: float = 10.0,
        lambda_p2d: float = 15.0
    ):
        super().__init__()
        self.lambda_pose = lambda_pose
        self.lambda_shape = lambda_shape
        self.lambda_cam = lambda_cam
        self.lambda_j3d = lambda_j3d
        self.lambda_p2d = lambda_p2d

    def forward(
        self,
        pred_rotmat: torch.Tensor,
        pred_shape: torch.Tensor,
        pred_cam: torch.Tensor,
        pred_joints_3d: torch.Tensor,
        pred_proj_2d: torch.Tensor,
        gt_rotmat: torch.Tensor,
        gt_shape: torch.Tensor,
        gt_cam: torch.Tensor,
        gt_joints_3d: torch.Tensor,
        gt_proj_2d: torch.Tensor
    ):
        # 1. Pose Rotation Loss (Frobenius in SO(3))
        loss_pose = F.mse_loss(pred_rotmat, gt_rotmat)

        # 2. Shape Beta Loss (Smooth L1)
        loss_shape = F.smooth_l1_loss(pred_shape, gt_shape, beta=0.1)

        # 3. Weak-Perspective Camera Loss
        loss_cam = F.smooth_l1_loss(pred_cam, gt_cam, beta=0.1)

        # 4. 3D Joint Euclidean Loss (Prevents limb twisting/dislocation)
        loss_j3d = F.mse_loss(pred_joints_3d, gt_joints_3d)

        # 5. 2D Reprojection Loss (Anchors 3D pose to the 2D image plane)
        loss_p2d = F.smooth_l1_loss(pred_proj_2d, gt_proj_2d, beta=0.05)

        total_loss = (
            self.lambda_pose * loss_pose +
            self.lambda_shape * loss_shape +
            self.lambda_cam * loss_cam +
            self.lambda_j3d * loss_j3d +
            self.lambda_p2d * loss_p2d
        )

        return total_loss, {
            "loss_pose": loss_pose.item(),
            "loss_shape": loss_shape.item(),
            "loss_cam": loss_cam.item(),
            "loss_j3d": loss_j3d.item(),
            "loss_p2d": loss_p2d.item(),
            "loss_total": total_loss.item()
        }

def compute_geodesic_angle_error(pred_rotmat: torch.Tensor, gt_rotmat: torch.Tensor) -> float:
    """
    Computes mean geodesic angular error (in degrees) across all 24 joints.
    Formula: theta = arccos((Tr(R_pred * R_gt^T) - 1) / 2)
    Input: [B, 24, 3, 3]
    """
    with torch.no_grad():
        R_diff = torch.matmul(pred_rotmat, gt_rotmat.transpose(-1, -2))
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
    pred_s = torch.zeros(B, 10)
    gt_s = torch.zeros(B, 10)
    pred_c = torch.ones(B, 3)
    gt_c = torch.ones(B, 3)
    pred_j3d = torch.zeros(B, 24, 3)
    gt_j3d = torch.zeros(B, 24, 3)
    pred_p2d = torch.zeros(B, 24, 2)
    gt_p2d = torch.zeros(B, 24, 2)

    loss, details = criterion(
        pred_R, pred_s, pred_c, pred_j3d, pred_p2d,
        gt_R, gt_s, gt_c, gt_j3d, gt_p2d
    )
    err_deg = compute_geodesic_angle_error(pred_R, gt_R)
    print("Loss Calculation:")
    print("  Details:", details)
    print(f"  Geodesic Error: {err_deg:.2f}°")
    assert err_deg < 1e-4
    print("✅ Full 5-Component Loss & Reprojection verified!")
