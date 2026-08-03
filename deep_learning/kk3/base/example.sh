#!/bin/bash
# Example workflow: Initialize K3 from pretrained weights and train

set -e

echo "🔄 Step 1: Transfer pretrained weights to K3"
uv run python -m base.transfer_weights \
    --source smollm2-360m \
    --output checkpoints/k3_pretrained_init.pt

echo ""
echo "✅ Pretrained initialization complete!"
echo ""
echo "🚀 Step 2: Train from pretrained checkpoint"
echo "   Run: python train.py --resume checkpoints/k3_pretrained_init.pt --epochs 10"
echo ""
echo "Or try different sources:"
echo "  - smollm2-135m (smaller, 2T tokens)"
echo "  - smollm2-360m (recommended, 4T tokens)"
echo "  - qwen2.5-0.5b (larger, multilingual)"
