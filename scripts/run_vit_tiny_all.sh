#!/usr/bin/env bash
# ViT-Ti/16 alpha sweep to validate the weight-norm scaling hypothesis.
# Predicted optimal α ≈ 0.38 (from attn.proj RMS norm ratio vs ViT-S).
# Runs α ∈ {0.0, 0.1, 0.3, 1.0}, seed=42, 40 epochs each (~5-6h total).
#
# Uses existing ../topo/scripts/run_experiment.py infrastructure.
#
# Usage:
#   chmod +x scripts/run_vit_tiny_all.sh
#   nohup scripts/run_vit_tiny_all.sh > /tmp/vit_tiny_all.log 2>&1 &

set -e
TOPO_DIR="$(dirname "$0")/../../topo"
PYTHON="$(dirname "$0")/../venv/bin/python"
SCRIPT="$TOPO_DIR/scripts/run_experiment.py"
CONFIG="$TOPO_DIR/configs/vit_ti16.yaml"

echo "=== ViT-Ti/16 alpha sweep — weight-norm scaling hypothesis ==="
echo "Started: $(date)"
echo ""

run() {
    local alpha=$1 run_id=$2
    echo "──────────────────────────────────────────"
    echo "  alpha=$alpha  run_id=$run_id"
    echo "  Started: $(date)"
    cd "$TOPO_DIR"
    $PYTHON scripts/run_experiment.py \
        --config configs/vit_ti16.yaml \
        --alpha "$alpha" \
        --run_id "$run_id" \
        --epochs 40
    cd - > /dev/null
    echo "  Finished: $(date)"
    echo ""
}

run 0.0  vit_ti_baseline_s42
run 0.1  vit_ti_weak_s42
run 0.3  vit_ti_mid_s42
run 1.0  vit_ti_strong_s42

echo "=== All runs complete: $(date) ==="
