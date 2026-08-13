# Video Dataset Setup for K3

Guide to adding video support using OpenVid-1M subset.

## Overview

- **Dataset**: 1K videos from OpenVid-1M
- **Storage**: ~200MB preprocessed (4 frames per video @ 112x112)
- **Location**: lv12/MultiModalDataset (openvid config)
- **Modalities**: Text + Images + Audio + **Video** (4 total)

---

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install decord opencv-python
   ```

2. **HuggingFace authentication:**
   ```bash
   huggingface-cli login
   # Use token with write access to lv12/MultiModalDataset
   ```

3. **Disk space:**
   - ~5GB for OpenVid_part0.zip download
   - ~3GB for temporary extracted videos
   - ~200MB for final preprocessed dataset

---

## Step 1: Download OpenVid-1M Part 0

OpenVid-1M is 12.4TB total, but we only need part0 (~5GB):

### Option A: Direct Download (if available)
Check the OpenVid-1M repository for direct download links to part0.zip

### Option B: Manual Download
Visit the OpenVid-1M dataset page and download `OpenVid_part0.zip`:
https://huggingface.co/datasets/nkp37/OpenVid-1M

**Note**: The full dataset is 12.4TB across 73 parts. We only need part0.

---

## Step 2: Preprocess Videos

Run the preprocessing script:

```bash
# Basic usage (1K videos)
uv run python scripts/preprocess_openvid.py \
    --zip-path /path/to/OpenVid_part0.zip \
    --output-dir data/openvid_preprocessed \
    --push-to-hub

# Custom options
uv run python scripts/preprocess_openvid.py \
    --zip-path /path/to/OpenVid_part0.zip \
    --output-dir data/openvid_preprocessed \
    --max-videos 1000 \
    --num-frames 4 \
    --frame-size 112 \
    --push-to-hub
```

**What it does:**
1. Extracts 1K videos from the ZIP
2. Loads captions from HuggingFace metadata
3. Extracts 4 frames per video (uniformly sampled)
4. Resizes to 112×112 RGB
5. Saves as HuggingFace Dataset with Image features
6. Pushes to lv12/MultiModalDataset as "openvid" config

**Output:**
```
✓ Extracted 1000 videos
✓ Found captions for 1000 videos
✓ Processed 1000 videos
✓ Train: 950 samples
✓ Val: 50 samples
✓ Pushed to HuggingFace Hub!
💾 Dataset size: 0.20 GB
```

---

## Step 3: Train with Video Data

Once preprocessing is complete, train with `--use-video`:

```bash
# Text + Images + Audio + Video (all modalities)
uv run python train.py --adam --use-audio --use-video --n-train 5000 --epochs 1

# Video only (for testing)
uv run python train.py --adam --use-video --n-train 1000 --epochs 1

# All modalities with larger dataset
uv run python train.py --adam --use-audio --use-video --n-train 10000 --epochs 2
```

**Dataset composition with all flags:**
- Text: `lv12/MultiModalDataset/fineweb`
- Images: `lv12/MultiModalDataset/coco`
- Audio: `lv12/MultiModalDataset/audioset`
- Video: `lv12/MultiModalDataset/openvid` ← New!

---

## Verification

Check that the dataset was uploaded:

```python
from datasets import load_dataset

# Load video dataset
ds = load_dataset("lv12/MultiModalDataset", "openvid", split="train")
print(f"Total videos: {len(ds)}")

# Check first sample
sample = ds[0]
print(f"Caption: {sample['caption'][:100]}...")
print(f"Frames: {[sample[f'frame_{i}'] for i in range(4)]}")
print(f"Metadata: aesthetic={sample['aesthetic_score']:.2f}, motion={sample['motion_score']:.2f}")
```

Expected output:
```
Total videos: 950
Caption: In the video, a man is seen in a living room setting, standing in front of a window with blinds...
Frames: [<PIL.JpegImagePlugin.JpegImageFile>, ...]
Metadata: aesthetic=5.43, motion=1.57
```

---

## Dataset Structure

```python
{
  'video_id': str,              # Original filename (without .mp4)
  'caption': str,               # Video description
  'frame_0': Image,             # First frame (112×112 RGB)
  'frame_1': Image,             # Second frame
  'frame_2': Image,             # Third frame
  'frame_3': Image,             # Fourth frame
  'aesthetic_score': float,     # 4.51-8.76
  'motion_score': float,        # Amount of motion
  'temporal_consistency': float,# 0.83-1.0
  'duration': float,            # Video length in seconds
  'fps': float,                 # Original frame rate
}
```

---

## Troubleshooting

### "No videos found!"
- Check that OpenVid_part0.zip path is correct
- Ensure ZIP contains .mp4 files

### "Failed to extract frames"
- Install decord: `pip install decord`
- Check that videos aren't corrupted

### "Failed to push to HuggingFace"
- Verify authentication: `huggingface-cli whoami`
- Check write access to lv12/MultiModalDataset
- Try pushing manually:
  ```python
  from datasets import load_from_disk
  ds = load_from_disk("data/openvid_preprocessed/dataset")
  ds.push_to_hub("lv12/MultiModalDataset", config_name="openvid")
  ```

### Out of memory during preprocessing
- Reduce `--max-videos` (try 500)
- Process in smaller batches
- Close other applications

### Slow preprocessing
- Preprocessing ~1000 videos takes 10-20 minutes
- Use SSD storage if available
- Increase `--num-workers` in script (if you modify it)

---

## Cleanup

After successful upload to HuggingFace:

```bash
# Remove raw videos (saves ~3GB)
rm -rf data/openvid_preprocessed/videos_raw

# Remove local dataset copy (if you want)
rm -rf data/openvid_preprocessed/dataset

# Keep only the frames for local backup
# data/openvid_preprocessed/frames (~200MB)
```

---

## Future Improvements

1. **More videos**: Increase `--max-videos` or download additional parts
2. **Frame augmentation**: Random frame sampling each epoch
3. **Higher resolution**: Increase `--frame-size` (requires more storage)
4. **Temporal modeling**: Add temporal transformer layers to vision encoder
5. **Video-specific tasks**: Action recognition, video captioning benchmarks

---

## Cost Estimate

- **Download time**: 30-60 min (depends on connection)
- **Preprocessing time**: 10-20 min for 1K videos
- **Upload time**: 5-10 min
- **Storage (HF)**: ~200MB (well under free tier 50GB limit)
- **Training overhead**: Minimal (same as image dataset)
