#!/usr/bin/env bash
# Queue all 9 ResNet-18 training runs sequentially.
# Each run: ~2-3h on RTX 3050. Total: ~18-27h overnight.
#
# Usage:
#   chmod +x scripts/run_resnet18_all.sh
#   nohup scripts/run_resnet18_all.sh > /tmp/resnet18_all.log 2>&1 &

set -e
PYTHON="$(dirname "$0")/../venv/bin/python"
SCRIPT="$(dirname "$0")/train_resnet18.py"

echo "=== ResNet-18 TopoLoss Training — all 9 runs ==="
echo "Started: $(date)"
echo ""

run() {
    local alpha=$1 seed=$2 run_id=$3
    echo "──────────────────────────────────────────"
    echo "  alpha=$alpha  seed=$seed  run_id=$run_id"
    echo "  Started: $(date)"
    $PYTHON $SCRIPT --alpha "$alpha" --seed "$seed" --run-id "$run_id"
    echo "  Finished: $(date)"
    echo ""
}

# Baseline (α=0.0)
run 0.0  42  resnet18_baseline_s42
run 0.0  123 resnet18_baseline_s123
run 0.0  456 resnet18_baseline_s456

# TopoWeak (α=0.1)
run 0.1  42  resnet18_weak_s42
run 0.1  123 resnet18_weak_s123
run 0.1  456 resnet18_weak_s456

# TopoStrong (α=1.0)
run 1.0  42  resnet18_strong_s42
run 1.0  123 resnet18_strong_s123
run 1.0  456 resnet18_strong_s456

echo "=== All runs complete: $(date) ==="
