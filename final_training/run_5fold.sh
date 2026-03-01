#!/bin/bash
##############################################################
# CFM-ID Final Training — 5-Repeat 80/10/10 Scaffold Split
# OpenMP 32-thread parallelized training
#
# Usage:
#   bash run_5fold.sh [NUM_THREADS] [--dry-run]
#
# Examples:
#   bash run_5fold.sh          # use all cores, run all folds
#   bash run_5fold.sh 32       # use 32 threads
#   bash run_5fold.sh --dry-run  # show commands without executing
#   bash run_5fold.sh 32 --dry-run
##############################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="cfmid-final"
N_FOLDS=5
DRY_RUN=false

# Parse arguments
NUM_THREADS=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        [0-9]*) NUM_THREADS="$arg" ;;
    esac
done

# Auto-detect CPU threads
TOTAL_THREADS=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)
PHYSICAL_CORES=$(lscpu 2>/dev/null | grep "^Core(s) per socket:" | awk '{print $NF}' || echo "")
SOCKETS=$(lscpu 2>/dev/null | grep "^Socket(s):" | awk '{print $NF}' || echo "")
if [ -n "$PHYSICAL_CORES" ] && [ -n "$SOCKETS" ]; then
    TOTAL_PHYSICAL=$((PHYSICAL_CORES * SOCKETS))
else
    TOTAL_PHYSICAL=$TOTAL_THREADS
fi

NUM_THREADS=${NUM_THREADS:-$TOTAL_THREADS}

if [ "$NUM_THREADS" -gt "$TOTAL_THREADS" ]; then
    echo "WARNING: Requested $NUM_THREADS threads, only $TOTAL_THREADS available."
    NUM_THREADS=$TOTAL_THREADS
fi

# Determine OMP_PLACES
if [ "$NUM_THREADS" -gt "$TOTAL_PHYSICAL" ]; then
    OMP_PLACES_VAL="threads"
    HT_MODE="yes"
else
    OMP_PLACES_VAL="cores"
    HT_MODE="no"
fi

CPUSET_END=$((NUM_THREADS - 1))
CPUSET_RANGE="0-${CPUSET_END}"

echo "=================================================================="
echo "  CFM-ID Final Training — 5-Repeat Scaffold Split"
echo "=================================================================="
echo "  Physical cores: $TOTAL_PHYSICAL"
echo "  Logical threads: $TOTAL_THREADS"
echo "  OpenMP threads: $NUM_THREADS"
echo "  Hyperthreading: $HT_MODE"
echo "  CPU pinning: $CPUSET_RANGE"
echo "  Dry run: $DRY_RUN"
echo "=================================================================="

# Check Docker image
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo ""
    echo "ERROR: Docker image '$IMAGE_NAME' not found."
    echo "Build it first: cd docker && docker build -t $IMAGE_NAME ."
    exit 1
fi
echo "Docker image: $IMAGE_NAME (OK)"

# Loop over folds
for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    FOLD_DIR="$SCRIPT_DIR/fold_${FOLD}"
    CONTAINER_NAME="cfmid_final_fold_${FOLD}"

    echo ""
    echo "=================================================================="
    echo "  FOLD ${FOLD} / $((N_FOLDS - 1))"
    echo "=================================================================="

    # Check if already completed
    if [ -f "$FOLD_DIR/param_output.log" ]; then
        FILESIZE=$(stat -c%s "$FOLD_DIR/param_output.log" 2>/dev/null || stat -f%z "$FOLD_DIR/param_output.log" 2>/dev/null || echo 0)
        if [ "$FILESIZE" -gt 1000000 ]; then
            echo "  SKIP: param_output.log already exists (${FILESIZE} bytes)"
            echo "  Delete it to re-train: rm $FOLD_DIR/param_output.log"
            continue
        fi
    fi

    # Validate training files
    for f in input_molecules.txt features.txt config.txt; do
        if [ ! -f "$FOLD_DIR/$f" ]; then
            echo "  ERROR: Missing $FOLD_DIR/$f"
            echo "  Run: python prepare_training.py first"
            exit 1
        fi
    done

    if [ ! -d "$FOLD_DIR/spectra" ] || [ -z "$(ls -A "$FOLD_DIR/spectra" 2>/dev/null)" ]; then
        echo "  ERROR: No spectra files in $FOLD_DIR/spectra/"
        exit 1
    fi

    N_MOLECULES=$(head -1 "$FOLD_DIR/input_molecules.txt")
    N_SPECTRA=$(ls "$FOLD_DIR/spectra/" 2>/dev/null | wc -l)
    echo "  Molecules: $N_MOLECULES (train+val)"
    echo "  Spectra: $N_SPECTRA"

    # Check pretrained model
    PRETRAINED="pretrained/param_output.log"
    if [ -f "$FOLD_DIR/$PRETRAINED" ]; then
        echo "  Pretrained: CFM-ID 4.0 default [M+H]+ (OK)"
    else
        echo "  WARNING: No pretrained model at $FOLD_DIR/$PRETRAINED"
        echo "  Training will start from random initialization."
    fi

    # Prepare tmp_data directory
    mkdir -p "$FOLD_DIR/tmp_data"

    # Build cfm-train command
    CFM_CMD="cd /cfmid/public/train && cfm-train"
    CFM_CMD="$CFM_CMD -i input_molecules.txt"
    CFM_CMD="$CFM_CMD -f features.txt"
    CFM_CMD="$CFM_CMD -c config.txt"
    CFM_CMD="$CFM_CMD -p spectra/"
    CFM_CMD="$CFM_CMD -g 1"
    CFM_CMD="$CFM_CMD -l tmp_data/status.log"
    if [ -f "$FOLD_DIR/$PRETRAINED" ]; then
        CFM_CMD="$CFM_CMD -w $PRETRAINED"
    fi

    DOCKER_CMD="docker run -d \
        --name $CONTAINER_NAME \
        --cpus=$NUM_THREADS \
        --cpuset-cpus=\"$CPUSET_RANGE\" \
        -e OMP_NUM_THREADS=$NUM_THREADS \
        -e OMP_PROC_BIND=close \
        -e OMP_PLACES=\"$OMP_PLACES_VAL\" \
        -e OMP_SCHEDULE=\"dynamic\" \
        -e OMP_STACKSIZE=\"8M\" \
        -v \"$FOLD_DIR:/cfmid/public/train\" \
        $IMAGE_NAME \
        sh -c \"$CFM_CMD\""

    if [ "$DRY_RUN" = true ]; then
        echo ""
        echo "  [DRY RUN] Would execute:"
        echo "  $DOCKER_CMD"
        continue
    fi

    # Remove existing container if any
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

    echo ""
    echo "  Starting training..."
    echo "  Container: $CONTAINER_NAME"
    echo "  Start time: $(date '+%Y-%m-%d %H:%M:%S')"

    # Run Docker
    eval "$DOCKER_CMD"

    # Wait for container to finish
    echo "  Monitoring (docker logs -f $CONTAINER_NAME)..."
    echo "  Ctrl+C to detach (training continues in background)"
    echo ""

    docker wait "$CONTAINER_NAME" || true

    # Check result
    echo ""
    echo "  End time: $(date '+%Y-%m-%d %H:%M:%S')"
    if [ -f "$FOLD_DIR/param_output.log" ]; then
        FILESIZE=$(stat -c%s "$FOLD_DIR/param_output.log" 2>/dev/null || stat -f%z "$FOLD_DIR/param_output.log" 2>/dev/null || echo 0)
        echo "  Result: param_output.log (${FILESIZE} bytes)"
    else
        echo "  WARNING: param_output.log not found. Check logs:"
        echo "    docker logs $CONTAINER_NAME"
    fi

    # Save container logs
    docker logs "$CONTAINER_NAME" > "$FOLD_DIR/training.log" 2>&1 || true
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

done

echo ""
echo "=================================================================="
echo "  All folds complete!"
echo "=================================================================="

# Summary
COMPLETED=0
for FOLD in $(seq 0 $((N_FOLDS - 1))); do
    FOLD_DIR="$SCRIPT_DIR/fold_${FOLD}"
    if [ -f "$FOLD_DIR/param_output.log" ]; then
        FILESIZE=$(stat -c%s "$FOLD_DIR/param_output.log" 2>/dev/null || stat -f%z "$FOLD_DIR/param_output.log" 2>/dev/null || echo 0)
        if [ "$FILESIZE" -gt 1000000 ]; then
            echo "  Fold $FOLD: COMPLETE (${FILESIZE} bytes)"
            COMPLETED=$((COMPLETED + 1))
        else
            echo "  Fold $FOLD: INCOMPLETE (${FILESIZE} bytes)"
        fi
    else
        echo "  Fold $FOLD: NOT STARTED"
    fi
done

echo ""
echo "  Completed: $COMPLETED / $N_FOLDS folds"

if [ "$COMPLETED" -eq "$N_FOLDS" ]; then
    echo ""
    echo "  Next step: python evaluate_5fold.py"
fi
