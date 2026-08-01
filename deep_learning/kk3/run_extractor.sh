#!/bin/bash

# Add FFmpeg 7 to PATH (prioritize over system FFmpeg 4)
export PATH="$HOME/bin:$PATH"

# Add CUDA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/usr/local/cuda-12.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"

# Verify FFmpeg version
echo "🎬 FFmpeg version:"
ffmpeg -version 2>&1 | head -1
echo ""

# Get the token from the stored location
TOKEN_FILE="$HOME/.cache/huggingface/token"

if [ ! -f "$TOKEN_FILE" ]; then
    echo "❌ Token file not found at $TOKEN_FILE"
    echo ""
    echo "Please login first:"
    echo "  hf auth login"
    exit 1
fi

# Read and export the token
export HF_TOKEN=$(cat "$TOKEN_FILE")

echo "🔐 Authentication:"
echo "  ✓ Token loaded"
echo "  ✓ CUDA libs: /usr/local/cuda-12.3/targets/x86_64-linux/lib"
echo ""

# Set HuggingFace cache to bumblebee1
export HF_HOME="/media/starscream/bumblebee1/hf_cache"
export HF_DATASETS_CACHE="/media/starscream/bumblebee1/hf_cache/datasets"
export TRANSFORMERS_CACHE="/media/starscream/bumblebee1/hf_cache/transformers"
export HF_HUB_CACHE="/media/starscream/bumblebee1/hf_cache/hub"

mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"

echo "💾 Disk space on bumblebee1:"
df -h /media/starscream/bumblebee1 | tail -1
echo ""

# Run extractor
uv run python extractor.py \
    --datasets fineweb,coco,audioset \
    --output-dir /media/starscream/bumblebee1/lv12_multimodal_dataset \
    --num-workers 16 \
    --push-to-hub \
    --hub-name lv12/MultiModalDataset
