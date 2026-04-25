#!/usr/bin/env bash
# Run 2 ResNet-18 training runs: baseline and strong (seed=42 only).
# Architecture variability check — multi-seed robustness already shown with ViT-S.
# 40 epochs each (~5-6h total on RTX 3050).
#
# Usage:
#   chmod +x scripts/run_resnet18_all.sh
#   nohup scripts/run_resnet18_all.sh > /tmp/resnet18_all.log 2>&1 &

set -e
PYTHON="$(dirname "$0")/../venv/bin/python"
SCRIPT="$(dirname "$0")/train_resnet18.py"

echo "=== ResNet-18 TopoLoss Training — 2 runs (arch variability check) ==="
echo "Started: $(date)"
echo ""

run() {
    local alpha=$1 seed=$2 run_id=$3
    echo "──────────────────────────────────────────"
    echo "  alpha=$alpha  seed=$seed  run_id=$run_id"
    echo "  Started: $(date)"
    $PYTHON $SCRIPT --alpha "$alpha" --seed "$seed" --run-id "$run_id" --epochs 40
    echo "  Finished: $(date)"
    echo ""
}

# Baseline (α=0.0)
run 0.0  42  resnet18_baseline_s42

# TopoStrong (α=1.0)
run 1.0  42  resnet18_strong_s42

echo "=== All runs complete: $(date) ==="
