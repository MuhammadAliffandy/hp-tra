"""
Headless Production Training Pipeline for Multi-Task Volumetric PoseNet.
Designed for execution on NVIDIA DGX Clusters (A100 / H100) and local workstations.
"""

import os
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = str(PROJECT_ROOT / "dataset_coco")
DEFAULT_SAVE_DIR = str(PROJECT_ROOT / "checkpoints_dgx")

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import COCOVolumetricPoseDataset
from src.model import MultiTaskVolumetricPoseNet
from src.losses import MultiTaskVolumetricLoss
from src.visualize import create_dual_mode_canvas


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Task Volumetric PoseNet on COCO")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Root path to COCO dataset (strictly contained in project directory)")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR,
                        help="Directory to save model checkpoints (contained in project directory)")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_large",
                        choices=["mobilenet_v3_large", "mobilenet_v3_small", "resnet18"],
                        help="Backbone architecture for feature extraction")
    parser.add_argument("--epochs", type=int, default=80, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (use 0 for debugging)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on (e.g. 'cuda:0', 'cuda', 'cpu', 'mps')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = save_dir / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)

    data_dir = Path(args.data_dir).resolve()
    train_json = data_dir / "annotations" / "person_keypoints_train2017.json"
    val_json = data_dir / "annotations" / "person_keypoints_val2017.json"
    train_imgs = data_dir / "train2017"
    val_imgs = data_dir / "val2017"

    # Fallback to val for training if train split has not been downloaded
    if not train_json.exists() or not train_imgs.exists():
        print("⚠️ train2017 images or annotations not found! Using val2017 for both train and validation smoke test.")
        train_json = val_json
        train_imgs = val_imgs

    print("======================================================================")
    print("🚀 LAUNCHING MULTI-TASK VOLUMETRIC POSENET TRAINING")
    print("======================================================================")
    print(f"Hardware Device : {args.device}")
    print(f"Backbone        : {args.backbone}")
    print(f"Batch Size      : {args.batch_size}")
    print(f"Total Epochs    : {args.epochs}")
    print(f"Base LR         : {args.lr}")
    print(f"Checkpoints Dir : {save_dir}")
    print("======================================================================")

    # 1. Datasets & Loaders
    train_dataset = COCOVolumetricPoseDataset(
        annotations_json=str(train_json),
        images_dir=str(train_imgs),
        is_train=True
    )
    val_dataset = COCOVolumetricPoseDataset(
        annotations_json=str(val_json),
        images_dir=str(val_imgs),
        is_train=False
    )

    pin_mem = "cuda" in args.device
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem
    )

    # 2. Model & Optimization
    model = MultiTaskVolumetricPoseNet(backbone_name=args.backbone, pretrained=True).to(args.device)
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model Instantiated! Total Parameters: {params_count / 1e6:.2f} Million")

    criterion = MultiTaskVolumetricLoss(weight_coord=20.0, weight_volume=5.0, weight_vis=1.0).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_err = float("inf")
    start_time = time.time()

    # 3. Training Loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_total, train_px_err_total = 0.0, 0.0
        train_batches = 0

        for imgs, coords, vis, vols in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]", leave=False):
            imgs = imgs.to(args.device, dtype=torch.float32)
            coords = coords.to(args.device, dtype=torch.float32)
            vis = vis.to(args.device, dtype=torch.float32)
            vols = vols.to(args.device, dtype=torch.float32)

            optimizer.zero_grad()
            pred_coords, pred_vols, pred_vis = model(imgs)
            loss, metrics = criterion(pred_coords, pred_vols, pred_vis, coords, vis, vols)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += metrics["loss_total"]
            train_px_err_total += metrics["mean_px_err"]
            train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss_total / max(1, train_batches)
        avg_train_err = train_px_err_total / max(1, train_batches)

        # Validation Pass
        model.eval()
        val_loss_total, val_px_err_total, val_pck_total = 0.0, 0.0, 0.0
        val_batches = 0

        with torch.no_grad():
            for imgs, coords, vis, vols in val_loader:
                imgs = imgs.to(args.device, dtype=torch.float32)
                coords = coords.to(args.device, dtype=torch.float32)
                vis = vis.to(args.device, dtype=torch.float32)
                vols = vols.to(args.device, dtype=torch.float32)

                pred_coords, pred_vols, pred_vis = model(imgs)
                _, metrics = criterion(pred_coords, pred_vols, pred_vis, coords, vis, vols)

                val_loss_total += metrics["loss_total"]
                val_px_err_total += metrics["mean_px_err"]
                val_pck_total += metrics["pck_01"]
                val_batches += 1

        avg_val_loss = val_loss_total / max(1, val_batches)
        avg_val_err = val_px_err_total / max(1, val_batches)
        avg_val_pck = val_pck_total / max(1, val_batches)
        epoch_sec = time.time() - t0

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_sec:.1f}s) | "
              f"Train Loss: {avg_train_loss:.4f} (Err: {avg_train_err:.1f}px) | "
              f"Val Loss: {avg_val_loss:.4f} (Err: {avg_val_err:.1f}px, PCK: {avg_val_pck:.1f}%) | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save Best Checkpoint based on lowest pixel error
        is_best = avg_val_err < best_val_err
        if is_best:
            best_val_err = avg_val_err
            best_ckpt_path = save_dir / "volumetric_posenet_best.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_err": avg_val_err,
                "val_pck": avg_val_pck,
                "backbone": args.backbone
            }, best_ckpt_path)
            print(f"  ⭐ NEW BEST CHECKPOINT SAVED! (Val Pixel Error: {best_val_err:.2f} px)")

        # Periodic Visual Snapshot (every 10 epochs or at first epoch)
        if epoch % 10 == 0 or epoch == 1 or is_best:
            with torch.no_grad():
                sample_img, s_coords, s_vis, s_vols = val_dataset[0]
                pred_c, pred_v, pred_vis = model(sample_img.unsqueeze(0).to(args.device))
                p_coords_np = pred_c[0].cpu().numpy()
                p_vols_np = pred_v[0].cpu().numpy()
                p_vis_np = torch.sigmoid(pred_vis[0]).cpu().numpy()

                disp_img = sample_img.permute(1, 2, 0).numpy()
                disp_img = (disp_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
                bgr_img = (disp_img * 255.0).astype(np.uint8)[:, :, ::-1]

                canvas = create_dual_mode_canvas(bgr_img, p_coords_np, p_vols_np, p_vis_np)
                snapshot_path = snapshots_dir / f"snapshot_epoch_{epoch:03d}.png"
                cv2.imwrite(str(snapshot_path), canvas)

    total_min = (time.time() - start_time) / 60
    print(f"\n🎉 TRAINING COMPLETED IN {total_min:.1f} MINUTES!")
    print(f"Best Validation Pixel Error : {best_val_err:.2f} px")
    print(f"Best Weights Saved At       : {save_dir / 'volumetric_posenet_best.pth'}")


if __name__ == "__main__":
    main()
