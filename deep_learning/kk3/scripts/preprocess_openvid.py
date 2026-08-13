"""
Preprocess OpenVid-1M subset for K3 training.

Downloads 1K videos from OpenVid-1M part0, extracts 4 frames per video,
and uploads to lv12/MultiModalDataset as 'openvid' config.

Storage: 1K videos × 4 frames × ~50KB = ~200MB

Prerequisites:
    1. Download OpenVid_part0.zip manually (requires ~20GB disk space)
    2. pip install decord opencv-python datasets huggingface-hub
    3. huggingface-cli login (with write access to lv12/MultiModalDataset)

Usage:
    python scripts/preprocess_openvid.py \
        --zip-path /path/to/OpenVid_part0.zip \
        --output-dir data/openvid_preprocessed \
        --max-videos 5000
"""

import os
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm
import click
import cv2
import numpy as np
from datasets import Dataset, DatasetDict, Image as HFImage, load_dataset
import tempfile


def extract_zip_subset(zip_path: str, output_dir: str, max_files: int) -> list:
    """Extract first max_files videos from ZIP archive."""
    click.secho(f"📦 Extracting {max_files} videos from {zip_path}...", fg="cyan")

    extracted = []
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        video_files = [f for f in zf.namelist() if f.endswith('.mp4')][:max_files]

        for video_file in tqdm(video_files, desc="Extracting"):
            try:
                zf.extract(video_file, output_dir)
                # Move to flat directory (ZIP might have subdirs)
                src = os.path.join(output_dir, video_file)
                dst = os.path.join(output_dir, os.path.basename(video_file))
                if src != dst:
                    shutil.move(src, dst)
                extracted.append(os.path.basename(video_file))
            except Exception as e:
                click.secho(f"⚠ Failed to extract {video_file}: {e}", fg="yellow")
                continue

    click.secho(f"✓ Extracted {len(extracted)} videos", fg="green")
    return extracted


def extract_frames_from_video(video_path: str, num_frames: int = 4, frame_size: int = 112, random_sample: bool = False) -> list:
    """
    Extract frames from video using OpenCV.

    Args:
        random_sample: If True, sample frames with temporal jittering for augmentation
    """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return None

        # Uniform sampling (with optional jitter for augmentation)
        if random_sample and total_frames > num_frames * 2:
            # Random sampling with jitter (for training augmentation)
            segment_len = total_frames // num_frames
            indices = []
            for i in range(num_frames):
                start = i * segment_len
                end = min((i + 1) * segment_len, total_frames - 1)
                indices.append(np.random.randint(start, end + 1))
            indices = np.array(indices, dtype=int)
        else:
            # Fixed uniform sampling (for preprocessing/eval)
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # Convert BGR to RGB and resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (frame_size, frame_size))
            frames.append(frame)

        cap.release()

        # Ensure we got all frames
        if len(frames) != num_frames:
            return None

        return frames

    except Exception as e:
        return None


def load_captions_from_hf(video_filenames: list) -> dict:
    """Load captions from OpenVid-1M metadata on HuggingFace."""
    click.secho("📥 Loading captions from HuggingFace...", fg="cyan")

    # Load the metadata we already have
    ds = load_dataset("nkp37/OpenVid-1M", split="train", streaming=True)

    # Create filename -> caption mapping
    caption_map = {}
    processed = 0

    for sample in tqdm(ds, desc="Loading captions", total=len(video_filenames)):
        if sample['video'] in video_filenames:
            caption_map[sample['video']] = {
                'caption': sample['caption'],
                'aesthetic_score': sample['aesthetic score'],
                'motion_score': sample['motion score'],
                'temporal_consistency': sample['temporal consistency score'],
                'duration': sample['seconds'],
                'fps': sample['fps'],
            }
            processed += 1

        if processed >= len(video_filenames):
            break

    click.secho(f"✓ Found captions for {len(caption_map)} videos", fg="green")
    return caption_map


def preprocess_videos(
    video_dir: str,
    output_dir: str,
    caption_map: dict,
    num_frames: int = 4,
    frame_size: int = 112
):
    """Process all videos and create HuggingFace dataset."""
    click.secho(f"🎬 Processing videos from {video_dir}...", fg="cyan")

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    dataset_rows = []
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])

    for idx, video_file in enumerate(tqdm(video_files, desc="Processing videos")):
        # Skip if no caption
        if video_file not in caption_map:
            continue

        video_path = os.path.join(video_dir, video_file)

        # Extract frames
        frames = extract_frames_from_video(video_path, num_frames, frame_size)

        if frames is None:
            click.secho(f"⚠ Failed to process {video_file}", fg="yellow")
            continue

        # Save frames as JPEGs
        frame_paths = []
        for f_idx, frame in enumerate(frames):
            frame_path = os.path.join(frames_dir, f"{idx:06d}_frame{f_idx}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_paths.append(frame_path)

        # Add to dataset
        metadata = caption_map[video_file]
        dataset_rows.append({
            'video_id': video_file.replace('.mp4', ''),
            'caption': metadata['caption'],
            'frame_0': frame_paths[0],
            'frame_1': frame_paths[1],
            'frame_2': frame_paths[2],
            'frame_3': frame_paths[3],
            'aesthetic_score': metadata['aesthetic_score'],
            'motion_score': metadata['motion_score'],
            'temporal_consistency': metadata['temporal_consistency'],
            'duration': metadata['duration'],
            'fps': metadata['fps'],
        })

    click.secho(f"✓ Processed {len(dataset_rows)} videos", fg="green")
    return dataset_rows


@click.command()
@click.option("--zip-path", type=click.Path(exists=True), required=True,
              help="Path to OpenVid_part0.zip")
@click.option("--output-dir", type=click.Path(), default="data/openvid_preprocessed",
              help="Output directory for preprocessed data")
@click.option("--max-videos", type=int, default=1000,
              help="Number of videos to process")
@click.option("--num-frames", type=int, default=4,
              help="Frames to extract per video")
@click.option("--frame-size", type=int, default=112,
              help="Frame resolution (square)")
@click.option("--push-to-hub", is_flag=True,
              help="Push to lv12/MultiModalDataset after processing")
@click.option("--skip-extraction", is_flag=True,
              help="Skip ZIP extraction (if already extracted)")
def main(zip_path, output_dir, max_videos, num_frames, frame_size, push_to_hub, skip_extraction):
    """Preprocess OpenVid-1M subset for K3 training."""

    click.secho("🚀 OpenVid-1M Preprocessing Pipeline", fg="cyan", bold=True)
    click.secho(f"Target: {max_videos} videos, {num_frames} frames @ {frame_size}x{frame_size}", fg="cyan")

    os.makedirs(output_dir, exist_ok=True)
    video_dir = os.path.join(output_dir, "videos_raw")

    # Step 1: Extract videos from ZIP
    if skip_extraction:
        click.secho("⏭ Skipping extraction (--skip-extraction)", fg="yellow")
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')][:max_videos]
    else:
        video_files = extract_zip_subset(zip_path, video_dir, max_videos)

    if len(video_files) == 0:
        click.secho("❌ No videos found!", fg="red")
        return

    # Step 2: Load captions from HuggingFace
    caption_map = load_captions_from_hf(video_files)

    # Step 3: Process videos and extract frames
    dataset_rows = preprocess_videos(video_dir, output_dir, caption_map, num_frames, frame_size)

    if len(dataset_rows) == 0:
        click.secho("❌ No videos successfully processed!", fg="red")
        return

    # Step 4: Create HuggingFace Dataset
    click.secho("📊 Creating HuggingFace dataset...", fg="cyan")
    dataset = Dataset.from_list(dataset_rows)

    # Convert frame paths to Image features (enables automatic loading)
    for i in range(num_frames):
        dataset = dataset.cast_column(f'frame_{i}', HFImage())

    # Create train/val split (95/5)
    click.secho("✂️ Splitting into train/val...", fg="cyan")
    split = dataset.train_test_split(test_size=0.05, seed=42)

    dataset_dict = DatasetDict({
        'train': split['train'],
        'validation': split['test']
    })

    click.secho(f"✓ Train: {len(split['train'])} samples", fg="green")
    click.secho(f"✓ Val: {len(split['test'])} samples", fg="green")

    # Step 5: Save locally
    local_path = os.path.join(output_dir, "dataset")
    click.secho(f"💾 Saving to {local_path}...", fg="cyan")
    dataset_dict.save_to_disk(local_path)

    # Step 6: Push to HuggingFace Hub
    if push_to_hub:
        click.secho("🚀 Pushing to lv12/MultiModalDataset...", fg="cyan")
        try:
            dataset_dict.push_to_hub(
                "lv12/MultiModalDataset",
                config_name="openvid",
                commit_message=f"Add OpenVid subset: {len(dataset_rows)} videos, {num_frames} frames each"
            )
            click.secho("✓ Pushed to HuggingFace Hub!", fg="green", bold=True)
        except Exception as e:
            click.secho(f"❌ Failed to push: {e}", fg="red")
            click.secho("Dataset saved locally. You can push manually later.", fg="yellow")

    # Cleanup raw videos to save space
    click.secho("🧹 Cleaning up raw videos...", fg="cyan")
    shutil.rmtree(video_dir)

    click.secho("✅ Done! Dataset ready for training.", fg="green", bold=True)
    click.secho(f"📁 Location: {local_path}", fg="cyan")

    # Print size estimate
    frames_size = sum(os.path.getsize(os.path.join(output_dir, "frames", f))
                     for f in os.listdir(os.path.join(output_dir, "frames")))
    click.secho(f"💾 Dataset size: {frames_size / 1e9:.2f} GB", fg="cyan")


if __name__ == "__main__":
    main()
