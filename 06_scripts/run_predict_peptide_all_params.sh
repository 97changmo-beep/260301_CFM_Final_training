#!/bin/bash
##############################################################
# Run cfm-predict for DTB-modified peptide candidates
# with cfm_default, param_jjy, and neome_v8 params
# All 3 models run in parallel (~10 containers each)
##############################################################
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="cfmid-final"

PEPTIDE_DIR="$PROJECT_DIR/05_evaluation/peptide_dtb"

# Param/config for each model
# cfm_default: uses Docker-internal paths (no -v mount needed for params)
DEFAULT_PARAM="/trained_models_cfmid4.0/cfmid4/[M+H]+/param_output.log"
DEFAULT_CONFIG="/trained_models_cfmid4.0/cfmid4/[M+H]+/param_config.txt"

JJY_PARAM="$PROJECT_DIR/02_config/Param_JJY/param_output.log"
JJY_CONFIG="$PROJECT_DIR/02_config/Param_JJY/param_config.txt"

V8_PARAM="$PROJECT_DIR/02_config/Param_NeoME_V8/NeoME_V8_param_output.log"
V8_CONFIG="$PROJECT_DIR/02_config/Param_NeoME_V8/NeoME_V8_param_config.txt"

echo "=================================================================="
echo "  Peptide DTB — Multi-Model Prediction"
echo "  3 models × 9220 candidates, running in parallel"
echo "=================================================================="

# ─── Helper: run one chunk for cfm_default ───
run_default_chunk() {
    local CHUNK_FILE="$1"
    local MODEL_DIR="$2"
    local CHUNK_NAME=$(basename "$CHUNK_FILE")
    local START=$(date +%s)

    docker run --rm \
        --cpus=1 -e OMP_NUM_THREADS=1 \
        -v "$MODEL_DIR:/cfmid/public/predict" \
        -w /cfmid/public/predict \
        "$IMAGE_NAME" \
        cfm-predict "$CHUNK_NAME" 0.001 \
        "/trained_models_cfmid4.0/cfmid4/[M+H]+/param_output.log" \
        "/trained_models_cfmid4.0/cfmid4/[M+H]+/param_config.txt" 0 \
        predictions/ \
        > /dev/null 2>&1

    local END=$(date +%s)
    echo "  [cfm_default] $CHUNK_NAME — $((END-START))s"
}

# ─── Helper: run one chunk for external param models ───
run_external_chunk() {
    local CHUNK_FILE="$1"
    local MODEL_DIR="$2"
    local PARAM_PATH="$3"
    local CONFIG_PATH="$4"
    local MODEL_LABEL="$5"
    local CHUNK_NAME=$(basename "$CHUNK_FILE")
    local START=$(date +%s)

    docker run --rm \
        --cpus=1 -e OMP_NUM_THREADS=1 \
        -v "$MODEL_DIR:/cfmid/public/predict" \
        -v "$PARAM_PATH:/cfmid/public/predict/param.log:ro" \
        -v "$CONFIG_PATH:/cfmid/public/predict/config.txt:ro" \
        -w /cfmid/public/predict \
        "$IMAGE_NAME" \
        cfm-predict "$CHUNK_NAME" 0.001 \
        param.log config.txt 0 \
        predictions/ \
        > /dev/null 2>&1

    local END=$(date +%s)
    echo "  [$MODEL_LABEL] $CHUNK_NAME — $((END-START))s"
}

export -f run_default_chunk run_external_chunk
export IMAGE_NAME

# ─── Setup directories ───
for model in cfm_default param_jjy neome_v8; do
    MODEL_DIR="$PEPTIDE_DIR/$model"
    mkdir -p "$MODEL_DIR/predictions"
    for f in "$PEPTIDE_DIR"/batch_chunk_*.txt; do
        cp "$f" "$MODEL_DIR/"
    done
done

CONTAINERS_PER=10
START_TOTAL=$(date +%s)

# ─── cfm_default (background) ───
(
    echo ""
    echo "=== cfm_default: Starting ==="
    S=$(date +%s)
    ls "$PEPTIDE_DIR/cfm_default"/batch_chunk_*.txt | \
        xargs -P $CONTAINERS_PER -I {} bash -c \
        'run_default_chunk "$@"' _ {} "$PEPTIDE_DIR/cfm_default"
    E=$(date +%s)
    N=$(ls "$PEPTIDE_DIR/cfm_default/predictions/"*.log 2>/dev/null | wc -l)
    echo "=== cfm_default: DONE — $N predictions ($((E-S))s) ==="
) &
PID1=$!

# ─── param_jjy (background) ───
(
    echo ""
    echo "=== param_jjy: Starting ==="
    S=$(date +%s)
    ls "$PEPTIDE_DIR/param_jjy"/batch_chunk_*.txt | \
        xargs -P $CONTAINERS_PER -I {} bash -c \
        'run_external_chunk "$@"' _ {} "$PEPTIDE_DIR/param_jjy" "$JJY_PARAM" "$JJY_CONFIG" "param_jjy"
    E=$(date +%s)
    N=$(ls "$PEPTIDE_DIR/param_jjy/predictions/"*.log 2>/dev/null | wc -l)
    echo "=== param_jjy: DONE — $N predictions ($((E-S))s) ==="
) &
PID2=$!

# ─── neome_v8 (background) ───
(
    echo ""
    echo "=== neome_v8: Starting ==="
    S=$(date +%s)
    ls "$PEPTIDE_DIR/neome_v8"/batch_chunk_*.txt | \
        xargs -P $CONTAINERS_PER -I {} bash -c \
        'run_external_chunk "$@"' _ {} "$PEPTIDE_DIR/neome_v8" "$V8_PARAM" "$V8_CONFIG" "neome_v8"
    E=$(date +%s)
    N=$(ls "$PEPTIDE_DIR/neome_v8/predictions/"*.log 2>/dev/null | wc -l)
    echo "=== neome_v8: DONE — $N predictions ($((E-S))s) ==="
) &
PID3=$!

# Wait for all
wait $PID1 $PID2 $PID3

END_TOTAL=$(date +%s)
echo ""
echo "=================================================================="
echo "  All 3 models complete! Total: $((END_TOTAL - START_TOTAL))s"
for model in cfm_default param_jjy neome_v8; do
    N=$(ls "$PEPTIDE_DIR/$model/predictions/"*.log 2>/dev/null | wc -l)
    echo "  $model: $N predictions"
done
echo "=================================================================="
