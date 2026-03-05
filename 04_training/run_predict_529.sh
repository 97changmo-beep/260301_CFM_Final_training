#!/bin/bash
##############################################################
# Run cfm-predict for 529 feature candidates × 3 models
# 6097 unique SMILES, 13 chunks of 500
##############################################################
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="$SCRIPT_DIR/eval_529"
IMAGE_NAME="cfmid-final"
NUM_THREADS=$(nproc 2>/dev/null || echo 4)

# Param paths
PARAM_DEFAULT="/trained_models_cfmid4.0/cfmid4/[M+H]+/param_output.log"
CONFIG_DEFAULT="/trained_models_cfmid4.0/cfmid4/[M+H]+/param_config.txt"
PARAM_JJY="$SCRIPT_DIR/../Param_JJY/param_output.log"
PARAM_JJY_CONFIG="$SCRIPT_DIR/../Param_JJY/param_config.txt"
PARAM_FULL="$SCRIPT_DIR/full_model/param_output.log"
CONFIG_FULL="$SCRIPT_DIR/config.txt"

echo "=================================================================="
echo "  529 Feature Candidates — CFM-ID Prediction"
echo "  6097 unique SMILES × 3 models"
echo "  OpenMP: $NUM_THREADS threads"
echo "=================================================================="

# ── Model 1: CFM-ID Default ──
MODEL="cfm_default"
MODEL_DIR="$EVAL_DIR/$MODEL"
PRED_DIR="$MODEL_DIR/predictions"
N_EXISTING=$(ls "$PRED_DIR"/*.log 2>/dev/null | wc -l)
echo ""
echo "=== $MODEL (existing: $N_EXISTING) ==="
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
        -w /cfmid/public/predict \
        "$IMAGE_NAME" \
        cfm-predict "$CHUNK_NAME" 0.001 \
        "$PARAM_DEFAULT" "$CONFIG_DEFAULT" 0 \
        predictions/
    END=$(date +%s)
    echo "$((END-START))s"
done
N_PRED=$(ls "$PRED_DIR"/*.log 2>/dev/null | wc -l)
END_ALL=$(date +%s)
echo "  $MODEL DONE: $N_PRED predictions ($((END_ALL-START_ALL))s total)"

# ── Model 2: Param_JJY ──
MODEL="param_jjy"
MODEL_DIR="$EVAL_DIR/$MODEL"
PRED_DIR="$MODEL_DIR/predictions"
N_EXISTING=$(ls "$PRED_DIR"/*.log 2>/dev/null | wc -l)
echo ""
echo "=== $MODEL (existing: $N_EXISTING) ==="
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
        -v "$PARAM_JJY:/cfmid/public/predict/param_jjy.log:ro" \
        -v "$PARAM_JJY_CONFIG:/cfmid/public/predict/config_jjy.txt:ro" \
        -w /cfmid/public/predict \
        "$IMAGE_NAME" \
        cfm-predict "$CHUNK_NAME" 0.001 \
        param_jjy.log config_jjy.txt 0 \
        predictions/
    END=$(date +%s)
    echo "$((END-START))s"
done
N_PRED=$(ls "$PRED_DIR"/*.log 2>/dev/null | wc -l)
END_ALL=$(date +%s)
echo "  $MODEL DONE: $N_PRED predictions ($((END_ALL-START_ALL))s total)"

# ── Model 3: Full Model (Final) ──
MODEL="full_model"
MODEL_DIR="$EVAL_DIR/$MODEL"
PRED_DIR="$MODEL_DIR/predictions"
N_EXISTING=$(ls "$PRED_DIR"/*.log 2>/dev/null | wc -l)
echo ""
echo "=== $MODEL (existing: $N_EXISTING) ==="
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
        -v "$PARAM_FULL:/cfmid/public/predict/param_full.log:ro" \
        -v "$CONFIG_FULL:/cfmid/public/predict/config_full.txt:ro" \
        -w /cfmid/public/predict \
        "$IMAGE_NAME" \
        cfm-predict "$CHUNK_NAME" 0.001 \
        param_full.log config_full.txt 0 \
        predictions/
    END=$(date +%s)
    echo "$((END-START))s"
done
N_PRED=$(ls "$PRED_DIR"/*.log 2>/dev/null | wc -l)
END_ALL=$(date +%s)
echo "  $MODEL DONE: $N_PRED predictions ($((END_ALL-START_ALL))s total)"

echo ""
echo "=================================================================="
echo "  All predictions complete!"
echo "  Results in: $EVAL_DIR/{cfm_default,param_jjy,full_model}/predictions/"
echo "=================================================================="
