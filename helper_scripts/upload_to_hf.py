#!/usr/bin/env python3
"""
Upload LiDAR-only checkpoints to HuggingFace Hub
"""

from huggingface_hub import HfApi, CommitOperationAdd
from pathlib import Path
import os

# Configuration
REPO_ID = "sangramrout/topodiffuser"
REPO_TYPE = "model"

# Files to upload
checkpoints_dir = Path("/media/skr/storage/self_driving/TopoDiffuser/checkpoints")
files_to_upload = {
    "encoder_expanded_best.pth": "encoder_lidar_only_best.pth",
    "diffusion_unet_best.pth": "diffusion_lidar_only_best.pth",
}

# Initialize API
api = HfApi()

print(f"Uploading to {REPO_ID}...")
print("=" * 60)

for local_name, remote_name in files_to_upload.items():
    local_path = checkpoints_dir / local_name
    
    if not local_path.exists():
        print(f"❌ {local_name} not found at {local_path}")
        continue
    
    file_size = local_path.stat().st_size / (1024**2)  # MB
    print(f"\n📤 Uploading: {local_name} ({file_size:.1f} MB)")
    print(f"   → Remote path: {remote_name}")
    
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_name,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"Add LiDAR-only checkpoint: {remote_name}"
        )
        print(f"   ✅ Uploaded successfully!")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("Upload complete!")
