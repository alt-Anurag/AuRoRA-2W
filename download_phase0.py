import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# --- Configuration ---
REPO_ID = "varunpaturkar/MOTOR"
REPO_TYPE = "dataset"
# Points directly to the data folder in your aurora-2w root
DESTINATION_DIR = Path("./data")

# The 12 curated clips covering standard and edge-case maneuvers
CLIP_IDS = [
    "01_019", "02_044", "03_119",           # Left Turns
    "01_071", "03_033",                     # Right Turns
    "01_006", "01_025", "04_110", "05_027", # Lane Changes
    "01_001", "01_037", "03_058"            # Overtakes
]

def setup_directories(base_dir: Path):
    """Ensures the raw_video and raw_imu directories exist."""
    video_dir = base_dir / "raw_video"
    telemetry_dir = base_dir / "raw_imu"
    
    video_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    
    return video_dir, telemetry_dir

def download_clip_pair(clip_id: str, video_dir: Path, telemetry_dir: Path):
    """Fetches the MP4 and CSV for a clip and places them in the correct folders."""
    ride_id = clip_id.split("_")[0]

    video_remote_path = f"clips/{ride_id}/front/{clip_id}.mp4"
    telemetry_remote_path = f"clips/{ride_id}/telemetry/{clip_id}.csv"

    # 1. Download and move Video
    try:
        cached_video = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=video_remote_path,
        )
        shutil.copy(cached_video, video_dir / f"{clip_id}.mp4")
    except Exception as e:
        print(f"[-] Failed to download video for {clip_id}: {e}")

    # 2. Download and move Telemetry
    try:
        cached_csv = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=telemetry_remote_path,
        )
        shutil.copy(cached_csv, telemetry_dir / f"{clip_id}.csv")
    except Exception as e:
        print(f"[-] Failed to download telemetry for {clip_id}: {e}")

def main():
    print(f"Starting download of {len(CLIP_IDS)} clips from {REPO_ID}...")
    video_dir, telemetry_dir = setup_directories(DESTINATION_DIR)

    for clip_id in tqdm(CLIP_IDS, desc="Downloading MOTOR batch"):
        download_clip_pair(clip_id, video_dir, telemetry_dir)

    print("\nDownload complete!")
    print(f"Videos saved to:    {video_dir.resolve()}")
    print(f"Telemetry saved to: {telemetry_dir.resolve()}")

if __name__ == "__main__":
    main()