import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# --- Configuration ---
REPO_ID = "varunpaturkar/MOTOR"
REPO_TYPE = "dataset"
DESTINATION_DIR = Path("./data")

# Top 3 largest (longest) clips from the dataset
CLIP_IDS = [
    "15_008", 
    "08_107", 
    "14_043"
]

def setup_directories(base_dir: Path):
    video_dir = base_dir / "raw_video"
    telemetry_dir = base_dir / "raw_imu"
    video_dir.mkdir(parents=True, exist_ok=True)
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    return video_dir, telemetry_dir

def download_clip_pair(clip_id: str, video_dir: Path, telemetry_dir: Path):
    ride_id = clip_id.split("_")[0]

    video_remote_path = f"clips/{ride_id}/front/{clip_id}.mp4"
    telemetry_remote_path = f"clips/{ride_id}/telemetry/{clip_id}.csv"

    print(f"\nFetching Video for {clip_id}...")
    try:
        cached_video = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=video_remote_path,
        )
        shutil.copy(cached_video, video_dir / f"{clip_id}.mp4")
    except Exception as e:
        print(f"[-] Failed to download video for {clip_id}: {e}")

    print(f"Fetching Telemetry for {clip_id}...")
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
    print(f"Starting download of {len(CLIP_IDS)} long-duration clips from {REPO_ID}...")
    video_dir, telemetry_dir = setup_directories(DESTINATION_DIR)

    for clip_id in CLIP_IDS:
        download_clip_pair(clip_id, video_dir, telemetry_dir)

    print("\nDownload complete!")
    print(f"Videos saved to:    {video_dir.resolve()}")
    print(f"Telemetry saved to: {telemetry_dir.resolve()}")

if __name__ == "__main__":
    main()
