#!/bin/bash
# Quick test run to verify everything works

set -e

echo "🧪 Quick test run (100 steps, all modalities)..."

# Force cache to wheeljack1
export HF_HOME="/media/starscream/wheeljack1/.cache/huggingface"
export HF_DATASETS_CACHE="/media/starscream/wheeljack1/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/media/starscream/wheeljack1/.cache/huggingface/transformers"

# Minimal config for low-memory GPUs
uv run python train.py \
    --adam \
    --n-train 500 \
    --batch-size 8 \
    --active-experts 2 \
    --total-experts 16 \
    --epochs 1

echo ""
echo "✅ Test complete! If this ran without errors, you're ready for full training."
