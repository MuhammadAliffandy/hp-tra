"""
Download Script for EFT (Exemplar Fine-Tuning) 3D Body Annotations
Paper: "Exemplar Fine-Tuning for 3D Human Pose Fitting Towards In-the-Wild 3D Human Pose Estimation" (FAIR / Meta AI, 3DV 2021)

Downloads COCO2014-Part-ver01.json (~174 MB) containing SMPL 3D pose (24 joint rotations)
and shape (10-dim beta volume) annotations for MS-COCO images.
"""

import os
import sys
import urllib.request
import time

DEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_coco", "eft")
TARGET_FILE = os.path.join(DEST_DIR, "COCO2014-Part-ver01.json")
EFT_URL = "https://dl.fbaipublicfiles.com/eft/eft_fit_ver01/COCO2014-Part-ver01.json"

def download_eft():
    os.makedirs(DEST_DIR, exist_ok=True)
    
    if os.path.exists(TARGET_FILE) and os.path.getsize(TARGET_FILE) > 100 * 1024 * 1024:
        size_mb = os.path.getsize(TARGET_FILE) / (1024 * 1024)
        print(f"✅ EFT annotations already downloaded and verified: {TARGET_FILE} ({size_mb:.1f} MB)")
        return TARGET_FILE

    print(f"📥 Downloading COCO-EFT 3D Body Annotations from Facebook AI Research...")
    print(f"   URL : {EFT_URL}")
    print(f"   Dest: {TARGET_FILE}")

    start_time = time.time()
    
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            downloaded_mb = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r   Progress: [{percent:3d}%] {downloaded_mb:.1f} / {total_mb:.1f} MB")
            sys.stdout.flush()

    try:
        req = urllib.request.Request(EFT_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(TARGET_FILE, "wb") as out_file:
            total_size = int(resp.headers.get("Content-Length", 0))
            count = 0
            block_size = 1024 * 1024  # 1 MB blocks
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                count += 1
                reporthook(count, block_size, total_size)
                
        elapsed = time.time() - start_time
        print(f"\n🎉 Download finished in {elapsed:.1f}s ({os.path.getsize(TARGET_FILE) / (1024 * 1024):.1f} MB)!")
        return TARGET_FILE
    except Exception as e:
        print(f"\n❌ Error downloading EFT annotations: {e}")
        if os.path.exists(TARGET_FILE):
            os.remove(TARGET_FILE)
        sys.exit(1)

if __name__ == "__main__":
    download_eft()
