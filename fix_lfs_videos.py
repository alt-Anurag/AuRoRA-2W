import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "varunpaturkar/MOTOR"
REPO_TYPE = "dataset"
DESTINATION_DIR = Path("./data")

CLIP_IDS = [
    "01_019", "02_044", "03_119",
    "01_071", "03_033",
    "01_006", "01_025", "04_110", "05_027",
    "01_001", "01_037", "03_058",
    "08_107", "14_043", "15_008"
]

def download_real_files():
    video_dir = DESTINATION_DIR / "raw_video"
    telemetry_dir = DESTINATION_DIR / "raw_imu"
    
    for clip_id in tqdm(CLIP_IDS, desc="Downloading REAL videos"):
        ride_id = clip_id.split("_")[0]
        video_remote_path = f"clips/{ride_id}/front/{clip_id}.mp4"
        
        print(f"Fetching {clip_id}...")
        try:
            # Force download bypasses the cached 133-byte pointer
            cached_video = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=video_remote_path,
                force_download=True
            )
            shutil.copy(cached_video, video_dir / f"{clip_id}.mp4")
            
            # Print size to verify
            size = os.path.getsize(video_dir / f"{clip_id}.mp4")
            print(f"Successfully saved {clip_id}.mp4 (Size: {size / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"Failed to fetch {clip_id}: {e}")

if __name__ == "__main__":
    download_real_files()
