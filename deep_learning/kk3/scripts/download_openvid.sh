#!/bin/bash
# Download OpenVid_part0.zip using wheeljack1 storage

set -e

# Configure cache to use wheeljack1
export HF_HOME="/media/starscream/wheeljack1/.cache/huggingface"
export HF_DATASETS_CACHE="/media/starscream/wheeljack1/.cache/huggingface/datasets"

echo "🚀 Downloading OpenVid_part0.zip to wheeljack1"
echo "Cache location: $HF_HOME"
echo "Free space: $(df -h /media/starscream/wheeljack1 | tail -1 | awk '{print $4}')"
echo ""

# Download to wheeljack1
cd /media/starscream/wheeljack1/learn/university_of_toronto/learning_algorithms/deep_learning/kk3

hf download \
    --repo-type dataset \
    --local-dir data/openvid_raw \
    nkp37/OpenVid-1M \
    OpenVid_part0.zip

echo ""
echo "✓ Download complete!"
echo "File location: data/openvid_raw/OpenVid_part0.zip"
echo "File size: $(du -h data/openvid_raw/OpenVid_part0.zip | cut -f1)"
