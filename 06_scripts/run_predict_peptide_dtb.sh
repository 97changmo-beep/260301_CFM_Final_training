#!/bin/bash
##############################################################
# Run cfm-predict for DTB-modified peptide candidates
# Uses full_model (final) param
# Parallelizes across multiple Docker containers
##############################################################
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="cfmid-final"

# Params
PARAM_FILE="$PROJECT_DIR/04_training/full_model/param_output.log"
CONFIG_FILE="$PROJECT_DIR/02_config/config.txt"

# Peptide DTB directory
PEPTIDE_DIR="$PROJECT_DIR/05_evaluation/peptide_dtb"
PRED_DIR="$PEPTIDE_DIR/predictions"
mkdir -p "$PRED_DIR"

# Parallelization: use 16 containers × 2 threads each = 32 threads
TOTAL_CORES=$(nproc 2>/dev/null || echo 32)
MAX_PARALLEL=16
THREADS_PER_CONTAINER=$((TOTAL_CORES / MAX_PARALLEL))
if [ "$THREADS_PER_CONTAINER" -lt 1 ]; then
    THREADS_PER_CONTAINER=1
fi

echo "=================================================================="
echo "  Peptide DTB — CFM-ID Prediction (Full Model)"
echo "  9220 candidates in $(ls "$PEPTIDE_DIR"/batch_chunk_*.txt | wc -l) chunks"
echo "  Parallelization: $MAX_PARALLEL containers × $THREADS_PER_CONTAINER threads"
echo "=================================================================="

START_TOTAL=$(date +%s)

# Function to run a single chunk
run_chunk() {
    local CHUNK_FILE="$1"
    local CHUNK_NAME=$(basename "$CHUNK_FILE")
    local N_LINES=$(wc -l < "$CHUNK_FILE")
    local START=$(date +%s)

    docker run --rm \
        --cpus=$THREADS_PER_CONTAINER \
        -e OMP_NUM_THREADS=$THREADS_PER_CONTAINER \
        -v "$PEPTIDE_DIR:/cfmid/public/predict" \
        -v "$PARAM_FILE:/cfmid/public/predict/param.log:ro" \
        -v "$CONFIG_FILE:/cfmid/public/predict/config.txt:ro" \
        -w /cfmid/public/predict \
        "$IMAGE_NAME" \
        cfm-predict "$CHUNK_NAME" 0.001 \
        param.log config.txt 0 \
        predictions/ \
        > /dev/null 2>&1

    local END=$(date +%s)
    echo "  ✓ $CHUNK_NAME ($N_LINES) — $((END-START))s"
}

export -f run_chunk
export PEPTIDE_DIR PARAM_FILE CONFIG_FILE IMAGE_NAME THREADS_PER_CONTAINER

# Run all chunks in parallel (max $MAX_PARALLEL at once)
echo ""
echo "Starting parallel prediction..."
ls "$PEPTIDE_DIR"/batch_chunk_*.txt | \
    xargs -P $MAX_PARALLEL -I {} bash -c 'run_chunk "$@"' _ {}

END_TOTAL=$(date +%s)
N_PRED=$(ls "$PRED_DIR"/*.log 2>/dev/null | wc -l)

echo ""
echo "=================================================================="
echo "  Peptide DTB Prediction Complete!"
echo "  Total predictions: $N_PRED"
echo "  Total time: $((END_TOTAL - START_TOTAL))s"
echo "  Output: $PRED_DIR"
echo "=================================================================="
