#!/usr/bin/env bash
# Run full ablation study on SEED dataset
# Trains all 5 variants sequentially and saves results
# Usage: bash scripts/run_ablation.sh [--gpu 0]

set -euo pipefail

GPU="${1:-0}"
CONFIG="configs/ablation.yaml"
VARIANTS=("vanilla" "isgd_only" "isgd_adv" "isgd_adv_mi" "full")

echo "======================================"
echo "DG-MAGNet Ablation Study"
echo "Variants: ${VARIANTS[*]}"
echo "GPU: ${GPU}"
echo "======================================"

for VARIANT in "${VARIANTS[@]}"; do
    echo ""
    echo ">>> Starting variant: ${VARIANT}"
    echo "--------------------------------------"

    CUDA_VISIBLE_DEVICES=${GPU} python -m code.train \
        --config ${CONFIG} \
        --variant ${VARIANT} \
        2>&1 | tee "outputs/logs/ablation_${VARIANT}.log"

    echo ">>> Finished variant: ${VARIANT}"
done

echo ""
echo "======================================"
echo "All ablation variants complete."
echo "Results saved to outputs/figures/"
echo "======================================"
