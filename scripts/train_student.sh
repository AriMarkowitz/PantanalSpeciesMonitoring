#!/bin/bash
#SBATCH --job-name=student_distill
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/student_%j.log
#SBATCH --error=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/student_%j.log

PROJECT_DIR="$HOME/PantanalSpeciesMonitoring"
ENV_NAME="pantanal"

mkdir -p "$PROJECT_DIR/outputs/logs"

module purge
module load cuda/12.8.0-dh6b
module load miniconda3/latest

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

cd "$PROJECT_DIR"

# ── Node-local mel cache (avoids 46GB hit on home quota) ────────────────────
# Build the cache to /tmp/$SLURM_JOB_ID — disappears when the job ends.
# Adds ~30 min wall time at job start; makes per-epoch ~10× faster than
# live decode and avoids touching the home filesystem.
export MELS_CACHE_DIR="/tmp/${SLURM_JOB_ID:-$$}/mels"
mkdir -p "$MELS_CACHE_DIR"
trap 'rm -rf "$MELS_CACHE_DIR" 2>/dev/null || true' EXIT
echo "MELS_CACHE_DIR=$MELS_CACHE_DIR"
df -h /tmp | tail -1

# Fix CuDNN
NVIDIA_SITE="$(python -c 'import nvidia.cudnn; print(nvidia.cudnn.__path__[0])' 2>/dev/null)/lib"
if [[ -d "$NVIDIA_SITE" ]]; then
    export LD_LIBRARY_PATH="$NVIDIA_SITE:${LD_LIBRARY_PATH:-}"
fi

export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Python: $(python --version)"
echo "---"

FOLD=${FOLD:-0}

# ── Step 1: Build mel cache to node-local /tmp ──────────────────────────────
# --distill builds primary AND distill caches; train_student loads both.
echo "=== Caching mels to $MELS_CACHE_DIR ==="
python src/cache_mels.py --distill
PRIMARY="$MELS_CACHE_DIR/embeddings.h5.mels.npy"
DISTILL="$MELS_CACHE_DIR/distill_embeddings.h5.mels.npy"
for p in "$PRIMARY" "$DISTILL"; do
    if [ ! -f "$p" ]; then
        echo "ERROR: expected cache file not found: $p" >&2
        echo "       train_student loads primary AND distill — abort early." >&2
        exit 1
    fi
done
echo "Cache built. Disk on /tmp:"
df -h /tmp | tail -1
du -sh "$MELS_CACHE_DIR"
echo "---"

# ── Step 2: Train student ───────────────────────────────────────────────────
echo "Training student embedder, fold=$FOLD"
python src/train_student.py ${OVERRIDE:+--set "$OVERRIDE"}

echo "Student distillation complete (job=$SLURM_JOB_ID)."
