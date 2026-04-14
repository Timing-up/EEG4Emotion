#!/usr/bin/env bash
# Train DG-MAGNet on SEED dataset with LOSO protocol
# Usage: bash scripts/train_seed.sh [--variant full] [--gpu 0]

set -euo pipefail

VARIANT="${1:-full}"
GPU="${2:-0}"
CONFIG="configs/seed_loso.yaml"

echo "=============================="
echo "DG-MAGNet SEED LOSO Training"
echo "Variant: ${VARIANT}"
echo "GPU: ${GPU}"
echo "Config: ${CONFIG}"
echo "=============================="

CUDA_VISIBLE_DEVICES=${GPU} python -m code.train \
    --config ${CONFIG} \
    --variant ${VARIANT}
