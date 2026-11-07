#!/usr/bin/env python3
"""
Automated Downloader for Official MS-COCO 2017 Keypoints Dataset.
Downloads and extracts annotations and images directly from the official COCO mirror.

Usage:
    # 1. Download validation dataset (~1.2 GB total, ideal for smoke tests & fast setup):
    python download_coco.py --split val

    # 2. Download full training + validation dataset (~20 GB total, for DGX cluster):
    python download_coco.py --split all --clean-zip
"""

import os
import sys
import argparse
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm


COCO_URLS = {
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path):
    if output_path.exists():
        print(f"✅ File already exists: {output_path.name} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"⬇️ Downloading: {url} -> {output_path.name}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    print(f"✅ Download complete: {output_path.name}")


def extract_zip(zip_path: Path, extract_to: Path, clean_zip: bool = False):
    print(f"📦 Extracting {zip_path.name} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extracted: {zip_path.name}")
    if clean_zip:
        print(f"🧹 Removing temporary zip archive: {zip_path.name}")
        zip_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Automated COCO 2017 Keypoints Downloader")
    parser.add_argument("--data-dir", type=str, default="./dataset_coco", help="Root directory to store COCO dataset (contained in project for easy cleanup)")
    parser.add_argument("--split", type=str, choices=["val", "train", "all"], default="val",
                        help="Data split to download: 'val' (~1.2GB), 'train' (~19GB), or 'all' (~20GB)")
    parser.add_argument("--clean-zip", action="store_true", help="Delete zip archives after extraction to save disk space")
    args = parser.parse_args()

    root_dir = Path(args.data_dir).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)
    zips_dir = root_dir / "zips"
    zips_dir.mkdir(exist_ok=True)

    print("======================================================================")
    print("📥 MS-COCO 2017 KEYPOINTS DATASET AUTOMATED DOWNLOADER")
    print("======================================================================")
    print(f"Destination Root : {root_dir}")
    print(f"Split Requested  : {args.split.upper()}")
    print("======================================================================")

    # 1. Always download and extract annotations
    ann_json_val = root_dir / "annotations" / "person_keypoints_val2017.json"
    ann_json_train = root_dir / "annotations" / "person_keypoints_train2017.json"
    if not (ann_json_val.exists() and ann_json_train.exists()):
        ann_zip = zips_dir / "annotations_trainval2017.zip"
        download_file(COCO_URLS["annotations"], ann_zip)
        extract_zip(ann_zip, root_dir, clean_zip=args.clean_zip)
    else:
        print("✅ Annotations already present in annotations/ folder.")

    # 2. Download Images based on split
    if args.split in ["val", "all"]:
        val_img_dir = root_dir / "val2017"
        if not val_img_dir.exists() or len(list(val_img_dir.glob("*.jpg"))) < 1000:
            val_zip = zips_dir / "val2017.zip"
            download_file(COCO_URLS["val2017"], val_zip)
            extract_zip(val_zip, root_dir, clean_zip=args.clean_zip)
        else:
            print(f"✅ val2017 images already extracted ({len(list(val_img_dir.glob('*.jpg')))} images found).")

    if args.split in ["train", "all"]:
        train_img_dir = root_dir / "train2017"
        if not train_img_dir.exists() or len(list(train_img_dir.glob("*.jpg"))) < 10000:
            train_zip = zips_dir / "train2017.zip"
            download_file(COCO_URLS["train2017"], train_zip)
            extract_zip(train_zip, root_dir, clean_zip=args.clean_zip)
        else:
            print(f"✅ train2017 images already extracted ({len(list(train_img_dir.glob('*.jpg')))} images found).")

    print("\n🎉 COCO KEYPOINTS DATASET READY!")
    print(f"📂 Dataset Root: {root_dir}")


if __name__ == "__main__":
    main()
