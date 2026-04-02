#!/bin/bash
#SBATCH --job-name=distill_embed
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/distill_embed_%j.log
#SBATCH --error=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/distill_embed_%j.log

# ── Project paths ──
PROJECT_DIR="$HOME/PantanalSpeciesMonitoring"
ENV_NAME="pantanal"

mkdir -p "$PROJECT_DIR/outputs/logs"

# ── Load modules ──
module purge
module load cuda/12.8.0-dh6b
module load miniconda3/latest

# ── Activate conda ──
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

cd "$PROJECT_DIR"

# ── Fix CuDNN: prefer pip-installed NVIDIA libs over system module ──
NVIDIA_SITE="$(python -c 'import nvidia.cudnn; print(nvidia.cudnn.__path__[0])' 2>/dev/null)/lib"
if [[ -d "$NVIDIA_SITE" ]]; then
    export LD_LIBRARY_PATH="$NVIDIA_SITE:${LD_LIBRARY_PATH:-}"
    echo "Prepended pip cuDNN to LD_LIBRARY_PATH: $NVIDIA_SITE"
fi

# ── Suppress cosmetic TF / abseil warnings ──
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=1
export KAGGLEHUB_VERBOSITY=error

# ── Print environment info ──
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'N/A')"
echo "---"

# ── Stage 0b: prepare distill segments (fast, CPU) ──
DISTILL_SEGMENTS="${DISTILL_SEGMENTS_CSV:-outputs/distill_segments.csv}"
N_EXISTING=0
if [[ -f "$PROJECT_DIR/$DISTILL_SEGMENTS" ]]; then
    N_EXISTING=$(tail -n +2 "$PROJECT_DIR/$DISTILL_SEGMENTS" | wc -l)
fi

if [[ "$N_EXISTING" -gt 0 ]]; then
    echo "distill_segments.csv already exists ($N_EXISTING rows) — skipping Stage 0b."
else
    echo "Running Stage 0b: prepare_distill_data..."
    python src/prepare_distill_data.py
    N_EXISTING=$(tail -n +2 "$PROJECT_DIR/$DISTILL_SEGMENTS" 2>/dev/null | wc -l || echo 0)
    if [[ "$N_EXISTING" -eq 0 ]]; then
        echo "ERROR: prepare_distill_data produced 0 segments — aborting."
        exit 1
    fi
fi

# ── Stage 1b: embed distill segments ──
echo "Running Stage 1b: embedding $N_EXISTING distill segments..."
python src/extract_embeddings.py \
    --set outputs.segments_csv="$DISTILL_SEGMENTS" \
    --set outputs.embeddings_h5=outputs/embeddings/distill_embeddings.h5

echo "Distill embedding extraction complete (job=$SLURM_JOB_ID)."
