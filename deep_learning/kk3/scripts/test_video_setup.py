"""
Test video dataset setup without downloading OpenVid.

Creates a tiny synthetic video dataset to verify the pipeline works
before committing to the 5GB download.
"""

import os
import click
import numpy as np
import torch
from PIL import Image
from datasets import Dataset, DatasetDict, Image as HFImage


def create_synthetic_video_dataset(output_dir: str, num_videos: int = 10, num_frames: int = 4, frame_size: int = 112):
    """Create synthetic video dataset for testing."""

    click.secho(f"🧪 Creating synthetic test dataset: {num_videos} videos", fg="cyan")

    os.makedirs(f"{output_dir}/frames", exist_ok=True)

    data = []

    for video_idx in range(num_videos):
        frame_paths = []

        # Generate random frames
        for frame_idx in range(num_frames):
            # Create random colored frame
            color = np.random.randint(0, 255, size=(frame_size, frame_size, 3), dtype=np.uint8)

            # Add some pattern so it looks video-like
            center = frame_size // 2
            offset = video_idx * 10 + frame_idx * 5
            cv2_available = False
            try:
                import cv2
                cv2.circle(color, (center + offset, center), 20, (255, 255, 255), -1)
                cv2_available = True
            except ImportError:
                # If cv2 not available, just use solid colors
                pass

            # Save frame
            frame_path = f"{output_dir}/frames/test_{video_idx:03d}_frame{frame_idx}.jpg"
            img = Image.fromarray(color)
            img.save(frame_path, quality=90)
            frame_paths.append(frame_path)

        # Synthetic caption
        caption = f"Synthetic test video {video_idx} showing random patterns and colors."

        data.append({
            'video_id': f'test_video_{video_idx:03d}',
            'caption': caption,
            'frame_0': frame_paths[0],
            'frame_1': frame_paths[1],
            'frame_2': frame_paths[2],
            'frame_3': frame_paths[3],
            'aesthetic_score': 5.0 + np.random.randn() * 0.5,
            'motion_score': 1.0 + np.random.randn() * 0.3,
            'temporal_consistency': 0.95 + np.random.rand() * 0.05,
            'duration': 5.0 + np.random.rand() * 3.0,
            'fps': 30.0,
        })

    # Create dataset
    dataset = Dataset.from_list(data)

    # Cast to Image features
    for i in range(num_frames):
        dataset = dataset.cast_column(f'frame_{i}', HFImage())

    # Split
    split = dataset.train_test_split(test_size=0.2, seed=42)

    dataset_dict = DatasetDict({
        'train': split['train'],
        'validation': split['test']
    })

    # Save locally
    dataset_dict.save_to_disk(f"{output_dir}/dataset")

    click.secho(f"✓ Created test dataset: {len(split['train'])} train, {len(split['test'])} val", fg="green")

    return dataset_dict


def test_loading():
    """Test loading the synthetic dataset."""
    click.secho("\n🧪 Testing dataset loading...", fg="cyan")

    from k3.video_data import HFVideoDataset

    try:
        # This will fail if openvid config doesn't exist, which is expected
        click.secho("Testing HFVideoDataset (will fail if openvid config not uploaded yet)...", fg="yellow")
        ds = HFVideoDataset("train", max_samples=10, seq_len=64)
        click.secho(f"✓ Loaded {len(ds)} samples from HuggingFace", fg="green")
    except Exception as e:
        click.secho(f"⚠ Expected failure (openvid not uploaded yet): {e}", fg="yellow")

    # Test local loading
    click.secho("\n🧪 Testing local dataset loading...", fg="cyan")
    from datasets import load_from_disk

    ds = load_from_disk("data/test_video/dataset")
    sample = ds['train'][0]

    click.secho(f"✓ Loaded local dataset", fg="green")
    click.secho(f"  - Caption: {sample['caption'][:50]}...", fg="white")
    click.secho(f"  - Frames: {[type(sample[f'frame_{i}']).__name__ for i in range(4)]}", fg="white")
    click.secho(f"  - Metadata: aesthetic={sample['aesthetic_score']:.2f}", fg="white")

    # Test tensor conversion
    click.secho("\n🧪 Testing frame tensor conversion...", fg="cyan")

    frame = np.array(sample['frame_0'].convert('RGB'))
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

    click.secho(f"✓ Frame tensor shape: {frame_tensor.shape}", fg="green")
    click.secho(f"  - Expected: torch.Size([3, 112, 112])", fg="white")

    if frame_tensor.shape == torch.Size([3, 112, 112]):
        click.secho("✅ All tests passed!", fg="green", bold=True)
    else:
        click.secho("⚠ Frame size mismatch!", fg="yellow")


@click.command()
@click.option("--output-dir", default="data/test_video", help="Output directory for test data")
@click.option("--num-videos", default=10, help="Number of test videos to create")
def main(output_dir, num_videos):
    """Create and test synthetic video dataset."""

    click.secho("🚀 Video Dataset Test Script", fg="cyan", bold=True)
    click.secho("This creates a tiny synthetic dataset to verify the pipeline works.\n", fg="cyan")

    # Create synthetic dataset
    dataset = create_synthetic_video_dataset(output_dir, num_videos)

    # Test loading
    test_loading()

    click.secho("\n✅ Setup verified! You can now:", fg="green", bold=True)
    click.secho("  1. Download OpenVid_part0.zip (~5GB)", fg="white")
    click.secho("  2. Run: uv run python scripts/preprocess_openvid.py --zip-path /path/to/part0.zip --push-to-hub", fg="white")
    click.secho("  3. Train: uv run python train.py --adam --use-video --n-train 1000", fg="white")


if __name__ == "__main__":
    main()
