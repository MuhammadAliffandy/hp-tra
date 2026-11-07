"""
Multi-Task Loss & Evaluation Metrics for Volumetric Pose Estimation.
Coordinates: Masked Smooth L1 Loss (Huber Loss).
Volume: Smooth L1 Loss.
Visibility: Binary Cross Entropy with Logits.
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskVolumetricLoss(nn.Module):
    def __init__(
        self,
        weight_coord: float = 20.0,
        weight_volume: float = 5.0,
        weight_vis: float = 1.0
    ):
        super().__init__()
        self.weight_coord = weight_coord
        self.weight_volume = weight_volume
        self.weight_vis = weight_vis

        self.smooth_l1 = nn.SmoothL1Loss(beta=0.02, reduction="none")
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred_coords: torch.Tensor,       # [B, 17, 2]
        pred_volume: torch.Tensor,       # [B, 6]
        pred_vis_logits: torch.Tensor,   # [B, 17]
        gt_coords: torch.Tensor,         # [B, 17, 2]
        gt_vis: torch.Tensor,            # [B, 17]
        gt_volume: torch.Tensor          # [B, 6]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Masked Coordinate Loss (Only punish visible joints)
        # diff: [B, 17, 2]
        diff_coord = self.smooth_l1(pred_coords, gt_coords).mean(dim=-1) # [B, 17]
        masked_coord_loss = (diff_coord * gt_vis).sum() / (gt_vis.sum() + 1e-6)

        # 2. Volume Parameter Loss
        vol_loss = F.smooth_l1_loss(pred_volume, gt_volume, beta=0.02)

        # 3. Visibility Classification Loss
        vis_loss = self.bce(pred_vis_logits, gt_vis)

        # Total Weighted Multi-Task Loss
        total_loss = (
            self.weight_coord * masked_coord_loss +
            self.weight_volume * vol_loss +
            self.weight_vis * vis_loss
        )

        # Calculate Mean Pixel Error on 256x256 image
        with torch.no_grad():
            pixel_dist = torch.linalg.vector_norm(
                (pred_coords - gt_coords) * 256.0, dim=-1
            ) # [B, 17]
            mean_px_error = (pixel_dist * gt_vis).sum() / (gt_vis.sum() + 1e-6)

            # PCK@0.1 (Tolerance: 25.6 pixels on 256x256 image)
            pck_01 = ((pixel_dist <= 25.6).float() * gt_vis).sum() / (gt_vis.sum() + 1e-6) * 100.0

        metrics = {
            "loss_total": total_loss.item(),
            "loss_coord": masked_coord_loss.item(),
            "loss_vol": vol_loss.item(),
            "loss_vis": vis_loss.item(),
            "mean_px_err": mean_px_error.item(),
            "pck_01": pck_01.item()
        }

        return total_loss, metrics
