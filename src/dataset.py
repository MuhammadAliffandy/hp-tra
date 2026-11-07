"""
PyTorch Dataset for MS-COCO Keypoints with Volumetric Geometry Supervision.
Streams image crops on-the-fly and extracts 17 continuous keypoint coordinates + 6 limb volume parameters.
"""

import json
from pathlib import Path
from typing import Tuple, List, Dict, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


# Official 17 MS-COCO Keypoint Names
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Standard Kinematic Skeleton Links (17 COCO Keypoints)
COCO_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Facial features
    (5, 6), (5, 11), (6, 12), (11, 12),       # Torso rectangle & pelvis
    (5, 7), (7, 9),                           # Left arm (shoulder -> elbow -> wrist)
    (6, 8), (8, 10),                          # Right arm (shoulder -> elbow -> wrist)
    (11, 13), (13, 15),                       # Left leg (hip -> knee -> ankle)
    (12, 14), (14, 16)                        # Right leg (hip -> knee -> ankle)
]

# Color Coding for Skeleton Visualization (BGR format for OpenCV)
SKELETON_COLORS = [
    (255, 153, 51), (255, 153, 51), (255, 153, 51), (255, 153, 51), # Face (Cyan)
    (0, 102, 255), (0, 102, 255), (0, 102, 255), (0, 102, 255),     # Torso (Blue)
    (0, 230, 115), (0, 230, 115),                                     # Left Arm (Green)
    (255, 128, 0), (255, 128, 0),                                     # Right Arm (Orange)
    (0, 204, 204), (0, 204, 204),                                     # Left Leg (Yellow-Cyan)
    (255, 179, 0), (255, 179, 0)                                      # Right Leg (Amber)
]


class COCOVolumetricPoseDataset(Dataset):
    """
    Lightweight, streaming dataset that indexes COCO person annotations with minimum
    keypoints filter and computes 17 normalized keypoint coordinates + 6 limb volume radii.
    """
    def __init__(
        self,
        annotations_json: str,
        images_dir: str,
        img_size: Tuple[int, int] = (256, 256),
        min_keypoints: int = 8,
        min_box_size: int = 25,
        is_train: bool = True
    ):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        self.is_train = is_train

        with open(annotations_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        id_to_img = {img["id"]: img for img in data.get("images", [])}
        self.samples = []

        print(f"🔍 Indexing COCO annotations from {Path(annotations_json).name}...")
        for ann in data.get("annotations", []):
            # 1. Filter out crowd instances and non-person categories
            if ann.get("iscrowd", 0) == 1 or ann.get("category_id") != 1:
                continue

            # 2. Minimum Keypoint Gate (Filters out cropped hands/feet/background noise)
            if ann.get("num_keypoints", 0) < min_keypoints:
                continue

            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            bx, by, bw, bh = bbox
            if bw < min_box_size or bh < min_box_size:
                continue

            img_id = ann.get("image_id")
            if img_id not in id_to_img:
                continue

            img_info = id_to_img[img_id]
            file_name = img_info["file_name"]
            img_path = self.images_dir / file_name
            if not img_path.is_file():
                continue

            raw_kpts = ann.get("keypoints", [])
            if len(raw_kpts) != 51: # 17 * 3
                continue

            self.samples.append({
                "path": img_path,
                "bbox": (float(bx), float(by), float(bw), float(bh)),
                "kpts": raw_kpts
            })

        print(f"✅ Successfully indexed {len(self.samples)} high-quality person instances!")

        # Transforms
        if self.is_train:
            self.transform = T.Compose([
                T.Resize(img_size),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def _compute_limb_volume_params(self, kpts_crop: np.ndarray, vis: np.ndarray) -> np.ndarray:
        """
        Calculates normalized anatomical volume parameters for 6 body segments:
        [head_r, torso_r, uarm_r, farm_r, thigh_r, calf_r] in range [0, 1].
        """
        vol = np.zeros(6, dtype=np.float32)

        def dist(i, j):
            if vis[i] > 0 and vis[j] > 0:
                return float(np.linalg.norm(kpts_crop[i] - kpts_crop[j]))
            return 0.0

        # 1. Head Radius (ears or eyes distance)
        d_ears = dist(3, 4)
        d_eyes = dist(1, 2)
        head_r = max(d_ears * 0.6, d_eyes * 0.9, 0.08)
        vol[0] = np.clip(head_r, 0.04, 0.25)

        # 2. Torso Half-Width
        d_shoulders = dist(5, 6)
        d_hips = dist(11, 12)
        torso_r = max(d_shoulders * 0.5, d_hips * 0.5, 0.12)
        vol[1] = np.clip(torso_r, 0.08, 0.35)

        # 3. Upper Arm Radius
        d_l_uarm = dist(5, 7)
        d_r_uarm = dist(6, 8)
        uarm_len = max(d_l_uarm, d_r_uarm, 0.15)
        vol[2] = np.clip(uarm_len * 0.20, 0.03, 0.12)

        # 4. Forearm Radius
        d_l_farm = dist(7, 9)
        d_r_farm = dist(8, 10)
        farm_len = max(d_l_farm, d_r_farm, 0.12)
        vol[3] = np.clip(farm_len * 0.16, 0.02, 0.10)

        # 5. Thigh Radius
        d_l_thigh = dist(11, 13)
        d_r_thigh = dist(12, 14)
        thigh_len = max(d_l_thigh, d_r_thigh, 0.20)
        vol[4] = np.clip(thigh_len * 0.24, 0.05, 0.16)

        # 6. Calf Radius
        d_l_calf = dist(13, 15)
        d_r_calf = dist(14, 16)
        calf_len = max(d_l_calf, d_r_calf, 0.18)
        vol[5] = np.clip(calf_len * 0.18, 0.04, 0.14)

        return vol

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        bx, by, bw, bh = sample["bbox"]

        with Image.open(sample["path"]) as img_pil:
            img_rgb = img_pil.convert("RGB")
        img_w, img_h = img_rgb.size

        # Crop with 10% breathing margin
        margin_x, margin_y = bw * 0.10, bh * 0.10
        x1 = max(0, int(bx - margin_x))
        y1 = max(0, int(by - margin_y))
        x2 = min(img_w, int(bx + bw + margin_x))
        y2 = min(img_h, int(by + bh + margin_y))

        # Safe clamp if crop is unusually small
        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w < 30:
            cx = (x1 + x2) / 2
            x1 = max(0, int(cx - 15))
            x2 = min(img_w, x1 + 30)
            crop_w = max(1, x2 - x1)
        if crop_h < 30:
            cy = (y1 + y2) / 2
            y1 = max(0, int(cy - 15))
            y2 = min(img_h, y1 + 30)
            crop_h = max(1, y2 - y1)

        cropped_img = img_rgb.crop((x1, y1, x2, y2))
        img_tensor = self.transform(cropped_img)

        # Transform 17 Keypoints to Normalized [0, 1] Range relative to crop
        raw_kpts = sample["kpts"]
        coords_norm = np.zeros((17, 2), dtype=np.float32)
        visibility = np.zeros(17, dtype=np.float32)

        for i in range(17):
            kx, ky, kv = raw_kpts[i * 3], raw_kpts[i * 3 + 1], raw_kpts[i * 3 + 2]
            if kv > 0:
                rel_x = np.clip((kx - x1) / crop_w, 0.0, 1.0)
                rel_y = np.clip((ky - y1) / crop_h, 0.0, 1.0)
                coords_norm[i] = [rel_x, rel_y]
                visibility[i] = 1.0
            else:
                coords_norm[i] = [0.0, 0.0]
                visibility[i] = 0.0

        # Calculate volumetric parameters
        volume_params = self._compute_limb_volume_params(coords_norm, visibility)

        return (
            img_tensor,
            torch.from_numpy(coords_norm),      # [17, 2] in [0, 1]
            torch.from_numpy(visibility),       # [17] in {0, 1}
            torch.from_numpy(volume_params)     # [6] in [0, 1]
        )
