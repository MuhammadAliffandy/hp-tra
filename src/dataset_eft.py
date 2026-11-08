"""
COCO-EFT Dataset Loader for Parametric 3D Human Mesh Recovery (HMR)
Iterasi 2: True 3D Volumetric Mannequin Dataset

Loads:
- Images from MS-COCO 2017 (train2017 / val2017)
- EFT 3D SMPL Annotations (COCO2014-Part-ver01.json):
  * parm_pose:  [24, 3, 3] rotation matrices
  * parm_shape: [10] volume/muscle beta vector
  * parm_cam:   [3] weak-perspective camera [scale, tx, ty]
  * bbox_center & bbox_scale
"""

import os
import json
from typing import Optional, List, Dict, Tuple
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class COCOEFTDataset(Dataset):
    def __init__(
        self,
        eft_json_path: str,
        coco_train_dir: str,
        coco_val_dir: Optional[str] = None,
        img_size: int = 256,
        is_train: bool = True,
        train_split_ratio: float = 0.95,
        margin: float = 1.15
    ):
        super().__init__()
        self.img_size = img_size
        self.margin = margin
        self.is_train = is_train

        if not os.path.exists(eft_json_path):
            raise FileNotFoundError(f"EFT annotations not found: {eft_json_path}. Please run download_eft.py first!")

        print(f"🔍 Loading EFT annotations from {eft_json_path}...")
        with open(eft_json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        entries = raw_data.get("data", [])
        print(f"   Total EFT raw entries: {len(entries)}")

        # Index available images in coco folders
        train_files = set(os.listdir(coco_train_dir)) if os.path.exists(coco_train_dir) else set()
        val_files = set(os.listdir(coco_val_dir)) if coco_val_dir and os.path.exists(coco_val_dir) else set()

        self.samples: List[Dict] = []
        for item in entries:
            img_raw_name = item.get("imageName", "")
            img_id = img_raw_name.split("_")[-1]

            img_path = None
            if img_id in train_files:
                img_path = os.path.join(coco_train_dir, img_id)
            elif coco_val_dir and img_id in val_files:
                img_path = os.path.join(coco_val_dir, img_id)
            elif os.path.exists(os.path.join(coco_train_dir, img_raw_name)):
                img_path = os.path.join(coco_train_dir, img_raw_name)

            if img_path:
                pose = item.get("parm_pose")
                shape = item.get("parm_shape")
                cam = item.get("parm_cam")
                center = item.get("bbox_center")
                scale = item.get("bbox_scale")

                if pose and shape and cam and center and scale:
                    self.samples.append({
                        "img_path": img_path,
                        "pose": np.array(pose, dtype=np.float32),      # [24, 3, 3]
                        "shape": np.array(shape, dtype=np.float32),    # [10]
                        "cam": np.array(cam, dtype=np.float32),        # [3]
                        "center": np.array(center, dtype=np.float32),  # [2]
                        "scale": float(scale)
                    })

        total_valid = len(self.samples)
        print(f"✅ Successfully matched {total_valid} samples with local COCO images!")

        # Train / Validation Split
        split_idx = int(total_valid * train_split_ratio)
        if is_train:
            self.samples = self.samples[:split_idx]
            print(f"   [Train Set]: {len(self.samples)} samples")
        else:
            self.samples = self.samples[split_idx:]
            print(f"   [Val Set]  : {len(self.samples)} samples")

        # Transforms
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.samples[idx]

        try:
            with Image.open(item["img_path"]) as img:
                img = img.convert("RGB")
                img_w, img_h = img.size

                # Square crop around bbox center
                cx, cy = item["center"]
                box_size = 200.0 * item["scale"] * self.margin

                x1 = int(round(cx - box_size / 2.0))
                y1 = int(round(cy - box_size / 2.0))
                x2 = int(round(cx + box_size / 2.0))
                y2 = int(round(cy + box_size / 2.0))

                # Handle padding if crop falls outside image borders
                pad_left = max(0, -x1)
                pad_top = max(0, -y1)
                pad_right = max(0, x2 - img_w)
                pad_bottom = max(0, y2 - img_h)

                if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                    new_w = img_w + pad_left + pad_right
                    new_h = img_h + pad_top + pad_bottom
                    padded_img = Image.new("RGB", (new_w, new_h), (0, 0, 0))
                    padded_img.paste(img, (pad_left, pad_top))
                    crop_x1 = x1 + pad_left
                    crop_y1 = y1 + pad_top
                    crop_x2 = x2 + pad_left
                    crop_y2 = y2 + pad_top
                    cropped = padded_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                else:
                    cropped = img.crop((x1, y1, x2, y2))

                resized = cropped.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
                tensor_img = self.normalize(resized)

        except Exception as e:
            # Fallback zero tensor on corrupted image
            tensor_img = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)

        gt_rotmat = torch.from_numpy(item["pose"])   # [24, 3, 3]
        gt_shape = torch.from_numpy(item["shape"])    # [10]
        gt_cam = torch.from_numpy(item["cam"])        # [3]

        return tensor_img, gt_rotmat, gt_shape, gt_cam
