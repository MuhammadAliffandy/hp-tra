"""
Production Training Engine for Parametric Volumetric HMR with IEF & 2D Reprojection (Iterasi 2)
Platform: NVIDIA DGX Cluster (H100/A100)
Architecture: MobileNetV3-Large + 3-Step Iterative Error Feedback (IEF)
Supervision:
- 3D Rotation Matrices (SO(3))
- Body Shape Beta (Volume)
- 3D Joint Euclidean Supervision
- 2D Weak-Perspective Reprojection Loss (Anchors 3D pose to 2D image plane)
"""

import os
import sys
import time
import argparse
from typing import Dict, Any

# Ensure repo root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model_hmr import VolumetricHMRNet, forward_kinematics_torch, project_3d_to_2d
from src.dataset_eft import COCOEFTDataset
from src.loss_hmr import VolumetricHMRLoss, compute_geodesic_angle_error

def parse_args():
    parser = argparse.ArgumentParser(description="Train Parametric Volumetric HMR with IEF & 2D Reprojection on COCO-EFT")
    parser.add_argument("--eft-json", type=str, default="./dataset_coco/eft/COCO2014-Part-ver01.json")
    parser.add_argument("--coco-train-dir", type=str, default="./dataset_coco/train2017")
    parser.add_argument("--coco-val-dir", type=str, default="./dataset_coco/val2017")
    parser.add_argument("--pretrained-base", type=str, default="./checkpoints_dgx/volumetric_posenet_best.pth")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_large", choices=["mobilenet_v3_large", "resnet18"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--output-dir", type=str, default="./checkpoints_dgx_hmr")
    parser.add_argument("--lambda-pose", type=float, default=10.0)
    parser.add_argument("--lambda-shape", type=float, default=5.0)
    parser.add_argument("--lambda-cam", type=float, default=1.0)
    parser.add_argument("--lambda-j3d", type=float, default=10.0)
    parser.add_argument("--lambda-p2d", type=float, default=15.0)
    return parser.parse_args()

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_pose_loss = 0.0
    total_shape_loss = 0.0
    total_p2d_loss = 0.0
    total_geo_err = 0.0
    num_batches = len(dataloader)

    offsets = getattr(model, "template_offsets")

    progress = tqdm(dataloader, desc="Training", leave=False, dynamic_ncols=True)
    for images, gt_rotmat, gt_shape, gt_cam in progress:
        images = images.to(device, non_blocking=True)
        gt_rotmat = gt_rotmat.to(device, non_blocking=True)
        gt_shape = gt_shape.to(device, non_blocking=True)
        gt_cam = gt_cam.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            # Forward through IEF model (3 refinement loops)
            pred_rotmat, pred_shape, pred_cam, pred_j3d, pred_p2d = model(images, n_iter=3)

            # Compute Ground-Truth 3D joints and 2D projections
            gt_j3d = forward_kinematics_torch(gt_rotmat, gt_shape, offsets)
            gt_p2d = project_3d_to_2d(gt_j3d, gt_cam)

            loss, details = criterion(
                pred_rotmat, pred_shape, pred_cam, pred_j3d, pred_p2d,
                gt_rotmat, gt_shape, gt_cam, gt_j3d, gt_p2d
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        geo_err = compute_geodesic_angle_error(pred_rotmat, gt_rotmat)

        total_loss += details["loss_total"]
        total_pose_loss += details["loss_pose"]
        total_shape_loss += details["loss_shape"]
        total_p2d_loss += details["loss_p2d"]
        total_geo_err += geo_err

        progress.set_postfix({
            "Loss": f"{details['loss_total']:.2f}",
            "Angle": f"{geo_err:.1f}°",
            "P2D": f"{details['loss_p2d']:.3f}",
            "Shape": f"{details['loss_shape']:.3f}"
        })

    return {
        "loss": total_loss / max(1, num_batches),
        "pose_loss": total_pose_loss / max(1, num_batches),
        "shape_loss": total_shape_loss / max(1, num_batches),
        "p2d_loss": total_p2d_loss / max(1, num_batches),
        "geo_err": total_geo_err / max(1, num_batches)
    }

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_geo_err = 0.0
    total_shape_loss = 0.0
    total_p2d_loss = 0.0
    num_batches = len(dataloader)

    offsets = getattr(model, "template_offsets")

    for images, gt_rotmat, gt_shape, gt_cam in dataloader:
        images = images.to(device, non_blocking=True)
        gt_rotmat = gt_rotmat.to(device, non_blocking=True)
        gt_shape = gt_shape.to(device, non_blocking=True)
        gt_cam = gt_cam.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred_rotmat, pred_shape, pred_cam, pred_j3d, pred_p2d = model(images, n_iter=3)

            gt_j3d = forward_kinematics_torch(gt_rotmat, gt_shape, offsets)
            gt_p2d = project_3d_to_2d(gt_j3d, gt_cam)

            loss, details = criterion(
                pred_rotmat, pred_shape, pred_cam, pred_j3d, pred_p2d,
                gt_rotmat, gt_shape, gt_cam, gt_j3d, gt_p2d
            )

        geo_err = compute_geodesic_angle_error(pred_rotmat, gt_rotmat)
        total_loss += details["loss_total"]
        total_shape_loss += details["loss_shape"]
        total_p2d_loss += details["loss_p2d"]
        total_geo_err += geo_err

    return {
        "val_loss": total_loss / max(1, num_batches),
        "val_geo_err": total_geo_err / max(1, num_batches),
        "val_shape_loss": total_shape_loss / max(1, num_batches),
        "val_p2d_loss": total_p2d_loss / max(1, num_batches)
    }

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Parametric Volumetric HMR with IEF & 2D Reprojection on {device}...")
    if device.type == "cuda":
        print(f"   Active GPU: {torch.cuda.get_device_name(0)}")

    # 1. Dataset & DataLoaders
    print("📦 Setting up COCO-EFT Datasets...")
    train_dataset = COCOEFTDataset(
        eft_json_path=args.eft_json,
        coco_train_dir=args.coco_train_dir,
        coco_val_dir=args.coco_val_dir,
        is_train=True,
        train_split_ratio=0.95
    )
    val_dataset = COCOEFTDataset(
        eft_json_path=args.eft_json,
        coco_train_dir=args.coco_train_dir,
        coco_val_dir=args.coco_val_dir,
        is_train=False,
        train_split_ratio=0.95
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # 2. Model & Transfer Learning
    model = VolumetricHMRNet(
        backbone_name=args.backbone,
        pretrained_base=args.pretrained_base
    ).to(device)

    criterion = VolumetricHMRLoss(
        lambda_pose=args.lambda_pose,
        lambda_shape=args.lambda_shape,
        lambda_cam=args.lambda_cam,
        lambda_j3d=args.lambda_j3d,
        lambda_p2d=args.lambda_p2d
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_geo_err = float("inf")
    best_ckpt_path = os.path.join(args.output_dir, "hmr_best.pth")

    print("\n" + "=" * 80)
    print(f"🔥 STARTING IEF TRAINING: {args.epochs} Epochs | Batch Size: {args.batch_size} | Samples: {len(train_dataset)}")
    print(f"   Supervision: SO(3) Pose + Beta Shape + 3D Joints + 2D Reprojection (Weight: {args.lambda_p2d})")
    print("=" * 80 + "\n")

    start_total_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        epoch_duration = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        is_best = val_metrics["val_geo_err"] < best_geo_err
        if is_best:
            best_geo_err = val_metrics["val_geo_err"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_geo_err": best_geo_err,
                "val_shape_loss": val_metrics["val_shape_loss"],
                "val_p2d_loss": val_metrics["val_p2d_loss"],
                "val_loss": val_metrics["val_loss"],
                "backbone": args.backbone
            }, best_ckpt_path)

        star_marker = " ⭐ [NEW BEST!]" if is_best else ""
        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.2f} (Angle: {train_metrics['geo_err']:.1f}°, Reproj2D: {train_metrics['p2d_loss']:.3f}) | "
            f"Val Loss: {val_metrics['val_loss']:.2f} (Angle: {val_metrics['val_geo_err']:.1f}°, Reproj2D: {val_metrics['val_p2d_loss']:.3f}) | "
            f"LR: {current_lr:.6f}{star_marker}"
        )

    total_time_min = (time.time() - start_total_time) / 60.0
    print("\n" + "=" * 80)
    print(f"🎉 Iterasi 2 HMR (IEF + 2D Reprojection) Complete in {total_time_min:.1f} minutes!")
    print(f"🏆 Best Validation Geodesic Error: {best_geo_err:.2f}° across 24 SMPL joints")
    print(f"📦 Best Weights Saved to: {best_ckpt_path}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
