#!/bin/bash
# Train with video dataset only (testing)

set -e

echo "🎬 Training with video dataset only..."
echo "This is for testing video integration before full multimodal training."
echo ""

# Force cache to wheeljack1
export HF_HOME="/media/starscream/wheeljack1/.cache/huggingface"
export HF_DATASETS_CACHE="/media/starscream/wheeljack1/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/media/starscream/wheeljack1/.cache/huggingface/transformers"

# Train with video only
uv run python train.py \
    --adam \
    --use-video \
    --n-train 1000 \
    --n-eval 100 \
    --batch-size 8 \
    --epochs 1 \
    --active-experts 4 \
    --total-experts 128

echo ""
echo "✅ Video-only training complete!"
echo "Next: Try full multimodal training with scripts/train_all_modalities.sh"
