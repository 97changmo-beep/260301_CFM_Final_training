#!/bin/bash
##############################################################
# Run cfm-predict with NeoME V8 param for:
#   1. 143 library compounds (top1_eval) — 6087 SMILES
#   2. 529 feature candidates (eval_529) — 6097 SMILES
##############################################################
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="cfmid-final"
NUM_THREADS=$(nproc 2>/dev/null || echo 4)

PARAM_NEOME="$SCRIPT_DIR/../Param_NeoME_V8/NeoME_V8_param_output.log"
CONFIG_NEOME="$SCRIPT_DIR/../Param_NeoME_V8/NeoME_V8_param_config.txt"

echo "=================================================================="
echo "  NeoME V8 — CFM-ID Prediction"
echo "  143 library (6087 SMILES) + 529 features (6097 SMILES)"
echo "  OpenMP: $NUM_THREADS threads"
echo "=================================================================="

# ── Part 1: 143 Library (top1_eval) ──
MODEL_DIR="$SCRIPT_DIR/top1_eval/neome_v8"
PRED_DIR="$MODEL_DIR/predictions"
N_EXISTING=$(ls "$PRED_DIR"/*.log 2>/dev/null | wc -l)
echo ""
echo "=== 143 Library — NeoME V8 (existing: $N_EXISTING) ==="
START_ALL=$(date +%s)

for CHUNK_FILE in "$MODEL_DIR"/batch_chunk_*.txt; do
    CHUNK_NAME=$(basename "$CHUNK_FILE")
    N_LINES=$(wc -l < "$CHUNK_FILE")
    echo -n "  $CHUNK_NAME ($N_LINES)... "
    START=$(date +%s)
    docker run --rm \
        --cpus=$NUM_THREADS \
        -e OMP_NUM_THREADS=$NUM_THREADS \
        -v "$MODEL_DIR:/cfmid/public/predict" \
        -v "$PARAM_NEOME:/cfmid/public/predict/param_neome.log:ro" \
        -v "$CONFIG_NEOME:/cfmid/public/predict/config_neome.txt:ro" \
        -w /cfmid/public/predict \
        "$IMAGE_NAME" \
        cfm-predict "$CHUNK_NAME" 0.001 \
        param_neome.log config_neome.txt 0 \
        predictions/
    END=$(date +%s)
    echo "$((END-START))s"
done
N_PRED=$(ls "$PRED_DIR"/*.log 2>/dev/null | wc -l)
END_ALL=$(date +%s)
echo "  143 Library DONE: $N_PRED predictions ($((END_ALL-START_ALL))s total)"

# ── Part 2: 529 Features ──
MODEL_DIR="$SCRIPT_DIR/eval_529/neome_v8"
PRED_DIR="$MODEL_DIR/predictions"
N_EXISTING=$(ls "$PRED_DIR"/*.log 2>/dev/null | wc -l)
echo ""
echo "=== 529 Features — NeoME V8 (existing: $N_EXISTING) ==="
START_ALL=$(date +%s)

for CHUNK_FILE in "$MODEL_DIR"/batch_chunk_*.txt; do
    CHUNK_NAME=$(basename "$CHUNK_FILE")
    N_LINES=$(wc -l < "$CHUNK_FILE")
    echo -n "  $CHUNK_NAME ($N_LINES)... "
    START=$(date +%s)
    docker run --rm \
        --cpus=$NUM_THREADS \
        -e OMP_NUM_THREADS=$NUM_THREADS \
        -v "$MODEL_DIR:/cfmid/public/predict" \
        -v "$PARAM_NEOME:/cfmid/public/predict/param_neome.log:ro" \
        -v "$CONFIG_NEOME:/cfmid/public/predict/config_neome.txt:ro" \
        -w /cfmid/public/predict \
        "$IMAGE_NAME" \
        cfm-predict "$CHUNK_NAME" 0.001 \
        param_neome.log config_neome.txt 0 \
        predictions/
    END=$(date +%s)
    echo "$((END-START))s"
done
N_PRED=$(ls "$PRED_DIR"/*.log 2>/dev/null | wc -l)
END_ALL=$(date +%s)
echo "  529 Features DONE: $N_PRED predictions ($((END_ALL-START_ALL))s total)"

echo ""
echo "=================================================================="
echo "  NeoME V8 — All predictions complete!"
echo "  143: $SCRIPT_DIR/top1_eval/neome_v8/predictions/"
echo "  529: $SCRIPT_DIR/eval_529/neome_v8/predictions/"
echo "=================================================================="
