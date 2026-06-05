#!/bin/bash
# =============================================================================
# run_experiments.sh — Run all MAD-Former comparison experiments
#
# Usage:
#   bash run_experiments.sh                # Run all (sequential)
#   bash run_experiments.sh --dry-run      # Print plan without running
#   bash run_experiments.sh --resume       # Resume from checkpoints (default)
#   bash run_experiments.sh --no-resume    # Force restart all
# =============================================================================
set -e

# ---- Configuration ----
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
DATA_DIR="data"
OUTPUT_DIR="./output"
EPOCHS=200

# Auto-detect GPU VRAM to set batch size
BATCH_SIZE=2
if command -v nvidia-smi &> /dev/null; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | sed 's/[^0-9]//g' || echo "0")
    VRAM_MB=${VRAM_MB:-0}
    if [ "$VRAM_MB" -ge 30000 ] 2>/dev/null; then
        BATCH_SIZE=8
        echo "[INFO] 32GB+ VRAM (RTX 5090) → batch_size=$BATCH_SIZE"
    elif [ "$VRAM_MB" -ge 20000 ] 2>/dev/null; then
        BATCH_SIZE=4
        echo "[INFO] 24GB VRAM → batch_size=$BATCH_SIZE"
    else
        echo "[INFO] VRAM=$VRAM_MB MB → batch_size=$BATCH_SIZE"
    fi
else
    echo "[INFO] No GPU detected → batch_size=$BATCH_SIZE"
fi

DRY_RUN=false
RESUME_FLAG="--resume"

# ---- Parse args ----
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --no-resume) RESUME_FLAG="--no-resume" ;;
        --resume)    RESUME_FLAG="--resume" ;;
        --device=*)  DEVICE="${arg#*=}" ;;
        --gpu=*)     DEVICE="cuda:${arg#*=}" ;;
        --batch-size=*) BATCH_SIZE="${arg#*=}" ;;
    esac
done

# ---- Experiment Matrix ----
# Format: "dataset task"
EXPERIMENTS=(
    # 18F-AV45
    "18F-AV45 AD_HC"
    "18F-AV45 EMCI_LMCI"
    "18F-AV45 HC_MCI"
    "18F-AV45 HC_ALL_MCI"
    # 18F-AV1451
    "18F-AV1451 AD_HC"
    "18F-AV1451 HC_MCI"
    # 18F-FBB
    "18F-FBB HC_MCI"
    # 18F-FDG
    "18F-FDG AD_HC"
    "18F-FDG EMCI_LMCI"
    "18F-FDG HC_MCI"
    "18F-FDG HC_ALL_MCI"
)

FOLDS=(0 1 2 3 4)

TOTAL=$(( ${#EXPERIMENTS[@]} * ${#FOLDS[@]} ))
CURRENT=0
FAILURES=()
COMPLETED=0
SKIPPED=0

# ---- Header ----
echo "=============================================================================="
echo " MAD-Former Full Experiment Pipeline"
echo "=============================================================================="
echo " Experiments: ${#EXPERIMENTS[@]} tasks × ${#FOLDS[@]} folds = $TOTAL runs"
echo " Device:      $DEVICE"
echo " Epochs:      $EPOCHS"
echo " Data dir:    $DATA_DIR"
echo " Output dir:  $OUTPUT_DIR"
echo " Resume:      $RESUME_FLAG"
echo "=============================================================================="
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would execute the following commands:"
    echo ""
    for entry in "${EXPERIMENTS[@]}"; do
        read -r DATASET TASK <<< "$entry"
        DATA_PATH="$DATA_DIR/$DATASET"
        for FOLD in "${FOLDS[@]}"; do
            echo "  python train.py --dataset $DATASET --task $TASK --fold $FOLD --data-dir $DATA_PATH --device $DEVICE $RESUME_FLAG"
        done
    done
    echo ""
    echo "  python aggregate_results.py --output-dir $OUTPUT_DIR --data-dir $DATA_DIR"
    exit 0
fi

# ---- Main Loop ----
START_TIME=$(date +%s)

for entry in "${EXPERIMENTS[@]}"; do
    read -r DATASET TASK <<< "$entry"
    DATA_PATH="$DATA_DIR/$DATASET"

    for FOLD in "${FOLDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        EXP_NAME="${DATASET}_${TASK}_fold${FOLD}"
        RESULT_FILE="$OUTPUT_DIR/results/${EXP_NAME}.json"

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "[$CURRENT/$TOTAL] $DATASET | $TASK | Fold $FOLD"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # Check if already completed
        if [ -f "$RESULT_FILE" ] && [[ "$RESUME_FLAG" == "--resume" ]]; then
            echo "[SKIP] Already completed → $RESULT_FILE"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Run training
        set +e  # Don't exit on failure
        $PYTHON train.py \
            --dataset "$DATASET" \
            --task "$TASK" \
            --fold "$FOLD" \
            --data-dir "$DATA_PATH" \
            --output-dir "$OUTPUT_DIR" \
            --device "$DEVICE" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            $RESUME_FLAG

        EXIT_CODE=$?
        set -e

        if [ $EXIT_CODE -eq 0 ]; then
            COMPLETED=$((COMPLETED + 1))
            echo "[OK] $EXP_NAME completed successfully."
        else
            FAILURES+=("$EXP_NAME")
            echo "[FAIL] $EXP_NAME exited with code $EXIT_CODE"
        fi
    done
done

# ---- Summary ----
END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "=============================================================================="
echo " Pipeline Complete"
echo "=============================================================================="
echo " Total:    $TOTAL"
echo " Done:     $COMPLETED"
echo " Skipped:  $SKIPPED"
echo " Failed:   ${#FAILURES[@]}"
echo " Time:     ${ELAPSED} minutes"
echo ""

if [ ${#FAILURES[@]} -gt 0 ]; then
    echo "Failed experiments:"
    for f in "${FAILURES[@]}"; do
        echo "  - $f"
    done
    echo ""
    echo "To re-run failures, use: bash run_experiments.sh --resume"
fi

# ---- Aggregate Results ----
echo "Aggregating results..."
$PYTHON aggregate_results.py --output-dir "$OUTPUT_DIR" --data-dir "$DATA_DIR"

echo ""
echo "All done! Results saved to $OUTPUT_DIR/results/"
