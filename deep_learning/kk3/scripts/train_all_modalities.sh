#!/bin/bash
# Train with ALL modalities: text, images, audio, video

set -e

echo "🚀 Training with all 4 modalities..."
echo "Datasets:"
echo "  - Text: lv12/MultiModalDataset/fineweb"
echo "  - Images: lv12/MultiModalDataset/coco"
echo "  - Audio: lv12/MultiModalDataset/audioset"
echo "  - Video: lv12/MultiModalDataset/openvid"
echo ""

# Force cache to wheeljack1
export HF_HOME="/media/starscream/wheeljack1/.cache/huggingface"
export HF_DATASETS_CACHE="/media/starscream/wheeljack1/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/media/starscream/wheeljack1/.cache/huggingface/transformers"

# Full multimodal training
uv run python train.py \
    --adam \
    --use-audio \
    --use-video \
    --n-train 5000 \
    --n-eval 100 \
    --batch-size 8 \
    --epochs 2 \
    --active-experts 4 \
    --total-experts 128

echo ""
echo "✅ Multimodal training complete!"
echo "Check wandb for training curves: https://wandb.ai"
echo "Checkpoints saved in: checkpoints/"
