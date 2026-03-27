#!/bin/bash
# End-to-end pipeline orchestrator.
# Usage: bash scripts/run_pipeline.sh [--skip-pseudo]
#
# Stages 0 and 3 run inline (fast, CPU-only).
# Stages 1, 2, 4, 4b are submitted as Slurm jobs with dependency chaining.

set -euo pipefail

PROJECT_DIR="$HOME/PantanalSpeciesMonitoring"
SKIP_PSEUDO=${1:-""}
NUM_PSEUDO_ROUNDS=${NUM_PSEUDO_ROUNDS:-2}

cd "$PROJECT_DIR"
mkdir -p outputs/logs

# ── Load modules for inline stages ──
module purge
module load miniconda3/latest
eval "$(conda shell.bash hook)"
conda activate pantanal

# ══════════════════════════════════════════════
# Stage 0: Data Preparation (inline, ~5 min)
# ══════════════════════════════════════════════
echo "=== Stage 0: Data Preparation ==="
if [ -f outputs/segments.csv ]; then
    echo "segments.csv already exists, skipping Stage 0"
    echo "  (delete outputs/segments.csv to regenerate)"
else
    python src/prepare_data.py --config configs/default.yaml \
        --set stage0.skip_silence_filter=true
fi

# ══════════════════════════════════════════════
# Stage 1: Embedding Extraction (GPU, ~6-12h)
# ══════════════════════════════════════════════
echo "=== Stage 1: Embedding Extraction ==="
EMBED_JOB=$(sbatch --parsable scripts/extract_embeddings.sh)
echo "Submitted embedding job: $EMBED_JOB"

# ══════════════════════════════════════════════
# Stage 2+3: Clustering + Feature Extraction
#   (CPU, high-mem, depends on Stage 1)
# ══════════════════════════════════════════════
echo "=== Stage 2+3: Clustering + Features (depends on $EMBED_JOB) ==="
CLUSTER_JOB=$(sbatch --parsable --dependency=afterok:$EMBED_JOB scripts/cluster.sh)
echo "Submitted clustering job: $CLUSTER_JOB"

# ══════════════════════════════════════════════
# Stage 4: Classifier Training
#   (GPU, depends on Stage 2+3, fan-out over folds)
# ══════════════════════════════════════════════
echo "=== Stage 4: Classifier Training (depends on $CLUSTER_JOB) ==="
LAST_TRAIN_JOBS=""
for FOLD in 0 1 2 3 4; do
    TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$CLUSTER_JOB \
        --export=ALL,FOLD=$FOLD scripts/train_classifier.sh)
    LAST_TRAIN_JOBS="${LAST_TRAIN_JOBS}:${TRAIN_JOB}"
    echo "  Fold $FOLD: $TRAIN_JOB"
done

# ══════════════════════════════════════════════
# Stage 4b: Pseudo-Labeling (optional)
# ══════════════════════════════════════════════
if [ "$SKIP_PSEUDO" != "--skip-pseudo" ]; then
    echo "=== Stage 4b: Pseudo-Labeling ==="

    # Round 1: Perch logit bootstrap
    PL_JOB=$(sbatch --parsable --dependency=afterok${LAST_TRAIN_JOBS} \
        --export=ALL,ROUND=1 scripts/pseudo_label.sh)
    echo "  Round 1: $PL_JOB"

    # Round 2+: self-training with retrain
    PREV_JOB=$PL_JOB
    for ROUND in $(seq 2 $NUM_PSEUDO_ROUNDS); do
        for FOLD in 0 1 2 3 4; do
            PL_JOB=$(sbatch --parsable --dependency=afterok:$PREV_JOB \
                --export=ALL,ROUND=$ROUND,FOLD=$FOLD \
                scripts/pseudo_label.sh)
            echo "  Round $ROUND fold $FOLD: $PL_JOB"
        done
        PREV_JOB=$PL_JOB
    done
else
    echo "=== Pseudo-labeling skipped (--skip-pseudo) ==="
fi

echo ""
echo "Pipeline submitted. Monitor with: squeue -u $USER"
