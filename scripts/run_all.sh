#!/usr/bin/env bash
# Master run script — Topo Monosemanticity Research
#
# Kill and re-run at any time — training auto-resumes from latest checkpoint.
#
# Usage:
#   nohup bash scripts/run_all.sh > logs/nohup_out.log 2>&1 &
#   tail -f logs/nohup_out.log
#
# Monitor from phone: https://ntfy.sh/toposae-c9499b67
# (or install the ntfy app and subscribe to: toposae-c9499b67)
#
set -euo pipefail

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT="/home/toposae/toposae"
VENV="$PROJECT/venv"
PYTHON="$VENV/bin/python"
RUNNER="$PROJECT/scripts/run_experiment.py"
POST="$PROJECT/scripts/post_analysis.py"
LOGS="$PROJECT/logs"

export NTFY_TOPIC="toposae-c9499b67"   # monitor at https://ntfy.sh/toposae-c9499b67

mkdir -p "$LOGS"
MASTER_LOG="$LOGS/run_all_$(date +%Y%m%d_%H%M%S).log"

# ── Helpers ────────────────────────────────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

fmt_mins() {
    local secs=$1
    local m=$(( secs / 60 ))
    if (( m < 60 )); then echo "${m}m"; else echo "$(( m/60 ))h$(( m%60 ))m"; fi
}

# Send to ntfy.sh — silent if curl missing or network down
ntfy() {
    local title="$1" msg="$2" priority="${3:-default}" tags="${4:-}"
    curl -s --max-time 5 \
        -H "Title: $title" \
        -H "Priority: $priority" \
        ${tags:+-H "Tags: $tags"} \
        -d "$msg" \
        "https://ntfy.sh/$NTFY_TOPIC" > /dev/null 2>&1 || true
}

# Run one experiment phase, streaming output to master log + dedicated log
run_phase() {
    local label="$1"; shift
    local exp_log="$LOGS/${label}_$(date +%Y%m%d_%H%M%S).log"
    log ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "  PHASE: $label"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    "$PYTHON" "$@" 2>&1 | tee -a "$exp_log" | tee -a "$MASTER_LOG"
    log "  PHASE DONE: $label"
}

# ── Sanity checks ──────────────────────────────────────────────────────────────
log "============================= RUN ALL =================================="
log "Project: $PROJECT"
log "Log:     $MASTER_LOG"
log "ntfy:    https://ntfy.sh/$NTFY_TOPIC"

if [ ! -f "$PYTHON" ]; then
    log "ERROR: venv missing at $VENV"
    ntfy "💥 STARTUP ERROR" "venv missing at $VENV" "urgent" "rotating_light"
    exit 1
fi
if [ ! -d "$PROJECT/data/imagenet-100/train" ]; then
    log "ERROR: ImageNet-100 not found"
    ntfy "💥 STARTUP ERROR" "ImageNet-100 data missing" "urgent" "rotating_light"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU info")
log "GPU: $GPU_INFO"

WALL_START=$(date +%s)
ntfy "🚀 Run All Started" \
    "GPU: $GPU_INFO
Pipeline: 11 training runs (ViT-S/16, ImageNet-100)
EXP A: 9 runs (3 seeds × 3 α)
EXP B: 2 runs (α=0.01 and 0.5)
+ Layerwise + Post-analysis
Est. total: ~62 hrs" \
    "high" "rocket"

# ─────────────────────────────────────────────────────────────────────────────
# EXP A: ViT-S/16 multi-seed  (9 runs: seeds 42,123,456 × α 0.0,0.1,1.0)
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "══════════════════════════════════════════════════════════════"
log "  EXP A: ViT-S/16 multi-seed  (9 runs, ~45 hrs)"
log "══════════════════════════════════════════════════════════════"
ntfy "📦 EXP A Starting" \
    "ViT-S/16 multi-seed
9 runs: seeds 42, 123, 456 × α 0.0, 0.1, 1.0
Each run: 90 epochs, ~5 hrs
Est phase total: ~45 hrs" \
    "default" "package"

EXP_A_START=$(date +%s)
COMPLETED_A=0
TOTAL_A=9
RUN_TIMES_A=()

for SEED in 42 123 456; do
    for ALPHA in 0.0 0.1 1.0; do
        LABEL=$(python3 -c "
m={0.0:'baseline',0.1:'topo_weak',1.0:'topo_strong'}
print(m.get($ALPHA, 'alpha$ALPHA'))
")
        RUN_ID="${LABEL}_s${SEED}"
        RUN_NUM=$(( COMPLETED_A + 1 ))

        log ""
        log "  ── EXP A run $RUN_NUM/$TOTAL_A: $RUN_ID (α=$ALPHA seed=$SEED) ──"
        ntfy "▶ EXP A  $RUN_NUM/$TOTAL_A" \
            "Starting: $RUN_ID
α=$ALPHA  seed=$SEED  90 epochs
$(( COMPLETED_A )) done so far" \
            "default" "arrow_forward"

        RUN_START=$(date +%s)
        if ! "$PYTHON" "$RUNNER" \
            --config "$PROJECT/configs/vit_s16.yaml" \
            --seeds "$SEED" \
            --alphas "$ALPHA" \
            --epochs 90 \
            2>&1 | tee -a "$LOGS/expa_${RUN_ID}_$(date +%Y%m%d_%H%M%S).log" | tee -a "$MASTER_LOG"; then
            ntfy "💥 EXP A CRASHED on $RUN_ID" \
                "Run $RUN_NUM/$TOTAL_A failed. Check log: $MASTER_LOG" \
                "urgent" "rotating_light"
            log "  ERROR: $RUN_ID failed. Continuing to next run."
        fi
        RUN_END=$(date +%s)
        RUN_DUR=$(( RUN_END - RUN_START ))
        RUN_TIMES_A+=("$RUN_DUR")
        COMPLETED_A=$(( COMPLETED_A + 1 ))

        # Compute average time + ETA for remaining runs
        RUNS_LEFT=$(( TOTAL_A - COMPLETED_A ))
        N_TIMES=${#RUN_TIMES_A[@]}
        SUM=0
        for t in "${RUN_TIMES_A[@]}"; do SUM=$(( SUM + t )); done
        AVG_RUN=$(( SUM / N_TIMES ))
        ETA_A=$(( AVG_RUN * RUNS_LEFT ))

        log "  EXP A: $COMPLETED_A/$TOTAL_A done  avg=$(fmt_mins $AVG_RUN)/run  eta=$(fmt_mins $ETA_A)"
        if (( RUNS_LEFT > 0 )); then
            ntfy "✅ EXP A  $COMPLETED_A/$TOTAL_A done" \
                "$RUN_ID finished in $(fmt_mins $RUN_DUR)
Avg/run: $(fmt_mins $AVG_RUN)
Remaining: $RUNS_LEFT runs (~$(fmt_mins $ETA_A))" \
                "default" "white_check_mark"
        fi
    done
done

EXP_A_DUR=$(( $(date +%s) - EXP_A_START ))
log "  EXP A complete in $(fmt_mins $EXP_A_DUR)"
ntfy "🏁 EXP A Complete" \
    "All 9 multi-seed runs done
Total: $(fmt_mins $EXP_A_DUR)
Up next: EXP B (2 α-sweep runs)" \
    "high" "checkered_flag"

# ─────────────────────────────────────────────────────────────────────────────
# EXP B: α sweep extras  (2 new runs: α=0.01 and α=0.5, seed=42)
# α=0.0, 0.1, 1.0 at seed=42 already done above — skipped by resume logic
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "══════════════════════════════════════════════════════════════"
log "  EXP B: α sweep extras  (2 new runs: α=0.01 and 0.5)"
log "══════════════════════════════════════════════════════════════"
ntfy "📦 EXP B Starting" \
    "α sweep: 2 new runs at seed=42
topo_001_s42 (α=0.01) and topo_05_s42 (α=0.5)
~10 hrs total" \
    "default" "package"

EXP_B_START=$(date +%s)
COMPLETED_B=0
TOTAL_B=2

for ALPHA in 0.01 0.5; do
    LABEL=$(python3 -c "
m={0.01:'topo_001',0.5:'topo_05'}
print(m.get($ALPHA, 'alpha$ALPHA'))
")
    RUN_ID="${LABEL}_s42"
    COMPLETED_B=$(( COMPLETED_B + 1 ))

    log ""
    log "  ── EXP B run $COMPLETED_B/$TOTAL_B: $RUN_ID (α=$ALPHA) ──"
    ntfy "▶ EXP B  $COMPLETED_B/$TOTAL_B" \
        "Starting: $RUN_ID  α=$ALPHA  seed=42" \
        "default" "arrow_forward"

    RUN_START=$(date +%s)
    if ! "$PYTHON" "$RUNNER" \
        --config "$PROJECT/configs/vit_s16.yaml" \
        --seeds 42 \
        --alphas "$ALPHA" \
        --epochs 90 \
        2>&1 | tee -a "$LOGS/expb_${RUN_ID}_$(date +%Y%m%d_%H%M%S).log" | tee -a "$MASTER_LOG"; then
        ntfy "💥 EXP B CRASHED on $RUN_ID" \
            "Check log: $MASTER_LOG" "urgent" "rotating_light"
        log "  ERROR: $RUN_ID failed. Continuing."
    fi
    RUN_DUR=$(( $(date +%s) - RUN_START ))
    ntfy "✅ EXP B  $COMPLETED_B/$TOTAL_B done" \
        "$RUN_ID finished in $(fmt_mins $RUN_DUR)" \
        "default" "white_check_mark"
done

EXP_B_DUR=$(( $(date +%s) - EXP_B_START ))
log "  EXP B complete in $(fmt_mins $EXP_B_DUR)"
ntfy "🏁 EXP B Complete" \
    "All 5 α values covered (incl. seed=42 from EXP A)
Total: $(fmt_mins $EXP_B_DUR)
Up next: Layer-wise SAE analysis" \
    "high" "checkered_flag"

# ─────────────────────────────────────────────────────────────────────────────
# LAYERWISE: Per-layer SAE on all 5 seed=42 runs
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "══════════════════════════════════════════════════════════════"
log "  LAYERWISE: Per-layer SAE analysis (5 runs × 12 layers)"
log "══════════════════════════════════════════════════════════════"
ntfy "📊 Layerwise Starting" \
    "SAE per layer for 5 seed=42 runs
Single forward pass collects all 12 layers at once
Est: ~2 hrs" \
    "default" "bar_chart"

LW_START=$(date +%s)
if ! run_phase "layerwise" \
    "$RUNNER" \
    --config "$PROJECT/configs/vit_s16.yaml" \
    --seeds 42 \
    --alphas 0.0 0.01 0.1 0.5 1.0 \
    --skip-training \
    --layerwise; then
    ntfy "💥 LAYERWISE CRASHED" "Check log: $MASTER_LOG" "urgent" "rotating_light"
fi
LW_DUR=$(( $(date +%s) - LW_START ))
ntfy "🏁 Layerwise Complete" "$(fmt_mins $LW_DUR)" "high" "checkered_flag"

# ─────────────────────────────────────────────────────────────────────────────
# POST: Multi-seed aggregation, α sweep plots, final report
# ─────────────────────────────────────────────────────────────────────────────
log ""
log "══════════════════════════════════════════════════════════════"
log "  POST: Aggregation + plots + RESULTS_FINAL.md"
log "══════════════════════════════════════════════════════════════"
ntfy "📈 Post-analysis Starting" \
    "Aggregating multi-seed stats
Generating α sweep + layer-wise plots" \
    "default" "chart_with_upwards_trend"

POST_START=$(date +%s)
if ! "$PYTHON" "$POST" 2>&1 | tee -a "$LOGS/post_$(date +%Y%m%d_%H%M%S).log" | tee -a "$MASTER_LOG"; then
    ntfy "💥 POST CRASHED" "Check log: $MASTER_LOG" "urgent" "rotating_light"
fi
POST_DUR=$(( $(date +%s) - POST_START ))

# ─────────────────────────────────────────────────────────────────────────────
# ALL DONE
# ─────────────────────────────────────────────────────────────────────────────
WALL_TOTAL=$(( $(date +%s) - WALL_START ))
log ""
log "============================= ALL DONE ================================="
log "Total wall time: $(fmt_mins $WALL_TOTAL)"
log "Results:  $PROJECT/results/RESULTS_FINAL.md"
log "JSON:     $PROJECT/results/json/"
log "Figures:  $PROJECT/results/figures/"
log "Full log: $MASTER_LOG"

ntfy "🎉 ALL DONE" \
    "Total wall time: $(fmt_mins $WALL_TOTAL)
11 training runs complete
Results: results/RESULTS_FINAL.md
Figures: results/figures/ (3 plots)" \
    "high" "tada"
