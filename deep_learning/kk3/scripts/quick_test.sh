#!/bin/bash
# Quick test run to verify everything works

set -e

echo "🧪 Quick test run (100 steps, all modalities)..."

# Force cache to wheeljack1
export HF_HOME="/media/starscream/wheeljack1/.cache/huggingface"
export HF_DATASETS_CACHE="/media/starscream/wheeljack1/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/media/starscream/wheeljack1/.cache/huggingface/transformers"

# Quick test with minimal settings for memory
uv run python train.py \
    --adam \
    --use-audio \
    --use-video \
    --n-train 200 \
    --n-eval 20 \
    --batch-size 1 \
    --epochs 1

echo ""
echo "✅ Test complete! If this ran without errors, you're ready for full training."
