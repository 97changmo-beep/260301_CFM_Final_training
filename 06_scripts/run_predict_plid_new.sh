#!/bin/bash
##############################################################
# Run cfm-predict for NEW PLID SMILES (not in 529 or peptide)
# Full model param, 32 threads via parallel Docker containers
##############################################################
set -e

PROJECT_DIR="/home/rheelab/바탕화면/Changmo_FINAL/260301_CFM_Final_training"
IMAGE_NAME="cfmid-final"
PARAM_FILE="$PROJECT_DIR/04_training/full_model/param_output.log"
CONFIG_FILE="$PROJECT_DIR/02_config/config.txt"

PRED_DIR="$PROJECT_DIR/05_evaluation/plid_new_predictions"
mkdir -p "$PRED_DIR/predictions"

TOTAL_CORES=$(nproc 2>/dev/null || echo 32)
MAX_PARALLEL=16
THREADS_PER=$((TOTAL_CORES / MAX_PARALLEL))

N_CHUNKS=$(ls "$PRED_DIR"/batch_chunk_*.txt 2>/dev/null | wc -l)
N_SMILES=$(cat "$PRED_DIR"/batch_chunk_*.txt 2>/dev/null | wc -l)

echo "=================================================================="
echo "  PLID New SMILES — CFM-ID Prediction (Full Model)"
echo "  $N_SMILES SMILES in $N_CHUNKS chunks"
echo "  $MAX_PARALLEL containers × $THREADS_PER threads"
echo "=================================================================="

run_chunk() {
    local CHUNK_FILE="$1"
    local CHUNK_NAME=$(basename "$CHUNK_FILE")
    local N_LINES=$(wc -l < "$CHUNK_FILE")
    local START=$(date +%s)

    docker run --rm \
        --cpus=$THREADS_PER \
        -e OMP_NUM_THREADS=$THREADS_PER \
        -v "$PRED_DIR:/cfmid/public/predict" \
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
export PRED_DIR PARAM_FILE CONFIG_FILE IMAGE_NAME THREADS_PER

START_TOTAL=$(date +%s)

ls "$PRED_DIR"/batch_chunk_*.txt | \
    xargs -P $MAX_PARALLEL -I {} bash -c 'run_chunk "$@"' _ {}

END_TOTAL=$(date +%s)
N_PRED=$(ls "$PRED_DIR/predictions/"*.log 2>/dev/null | wc -l)

echo ""
echo "=================================================================="
echo "  PLID New Predictions Complete!"
echo "  Total: $N_PRED predictions in $((END_TOTAL - START_TOTAL))s"
echo "=================================================================="
