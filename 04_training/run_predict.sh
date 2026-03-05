#!/bin/bash
##############################################################
# Run cfm-predict for all test compounds across 5 folds
# Uses OpenMP-parallelized cfm-predict (batch mode, 32 threads)
##############################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="cfmid-final"
N_FOLDS=5
NUM_THREADS=$(nproc 2>/dev/null || echo 4)

echo "=================================================================="
echo "  CFM-ID Batch Prediction — 5 Folds (OpenMP: $NUM_THREADS threads)"
echo "=================================================================="

for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    FOLD_DIR="$SCRIPT_DIR/fold_${FOLD}"
    TEST_FILE="$FOLD_DIR/test_compounds.txt"
    TRAINED_MODEL="$FOLD_DIR/param_output.log"
    PREDICT_DIR="$FOLD_DIR/predictions"
    BASELINE_DIR="$FOLD_DIR/baseline_predictions"

    echo ""
    echo "  Fold $FOLD:"

    if [ ! -f "$TRAINED_MODEL" ]; then
        echo "    SKIP: param_output.log not found"
        continue
    fi

    mkdir -p "$PREDICT_DIR" "$BASELINE_DIR"

    # Create batch input file: "id smiles" per line
    BATCH_FILE="$FOLD_DIR/test_batch.txt"
    grep -v "^#" "$TEST_FILE" | grep -v "^$" | awk -F'\t' '{print $1" "$2}' > "$BATCH_FILE"
    N_TEST=$(wc -l < "$BATCH_FILE")
    echo "    Test compounds: $N_TEST"

    # --- Trained model prediction (OpenMP parallel inside cfm-predict) ---
    echo "    Predicting with trained model..."
    START=$(date +%s)
    docker run --rm \
        --cpus=$NUM_THREADS \
        -e OMP_NUM_THREADS=$NUM_THREADS \
        -e OMP_SCHEDULE=dynamic \
        -v "$FOLD_DIR:/cfmid/public/predict" \
        -w /cfmid/public/predict \
        "$IMAGE_NAME" \
        cfm-predict test_batch.txt 0.001 \
        param_output.log config.txt 0 \
        predictions/
    END=$(date +%s)
    N_PRED=$(ls "$PREDICT_DIR"/*.log 2>/dev/null | wc -l)
    echo "    Trained: $N_PRED files ($((END - START))s)"

    # --- Baseline model prediction (OpenMP parallel) ---
    echo "    Predicting with baseline model..."
    START=$(date +%s)
    docker run --rm \
        --cpus=$NUM_THREADS \
        -e OMP_NUM_THREADS=$NUM_THREADS \
        -e OMP_SCHEDULE=dynamic \
        -v "$FOLD_DIR:/cfmid/public/predict" \
        -w /cfmid/public/predict \
        "$IMAGE_NAME" \
        cfm-predict test_batch.txt 0.001 \
        pretrained/param_output.log config.txt 0 \
        baseline_predictions/
    END=$(date +%s)
    N_BASE=$(ls "$BASELINE_DIR"/*.log 2>/dev/null | wc -l)
    echo "    Baseline: $N_BASE files ($((END - START))s)"

    # Cleanup
    rm -f "$BATCH_FILE"

done

echo ""
echo "=================================================================="
echo "  Prediction complete! Run: python evaluate_5fold.py"
echo "=================================================================="
