#!/bin/bash
#SBATCH --job-name=student_emb
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/student_emb_%j.log
#SBATCH --error=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/student_emb_%j.log

# Extract student embeddings for all training segments using cached mels.
# GPU-accelerated, ~10 min for 358k segments.

PROJECT_DIR="$HOME/PantanalSpeciesMonitoring"
ENV_NAME="pantanal"

mkdir -p "$PROJECT_DIR/outputs/logs"

module purge
module load cuda/12.8.0-dh6b
module load miniconda3/latest

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

cd "$PROJECT_DIR"

NVIDIA_SITE="$(python -c 'import nvidia.cudnn; print(nvidia.cudnn.__path__[0])' 2>/dev/null)/lib"
if [[ -d "$NVIDIA_SITE" ]]; then
    export LD_LIBRARY_PATH="$NVIDIA_SITE:${LD_LIBRARY_PATH:-}"
fi

export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ── Node-local mel cache ───────────────────────────────────────────────────
export MELS_CACHE_DIR="/tmp/${SLURM_JOB_ID:-$$}/mels"
mkdir -p "$MELS_CACHE_DIR"
trap 'rm -rf "$MELS_CACHE_DIR" 2>/dev/null || true' EXIT

echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "MELS_CACHE_DIR=$MELS_CACHE_DIR"
df -h /tmp | tail -1
echo "---"

# ── Step 1: Build mel cache (primary + distill) to /tmp ────────────────────
echo "=== Caching mels to $MELS_CACHE_DIR ==="
python src/cache_mels.py --distill
du -sh "$MELS_CACHE_DIR"
echo "---"

# ── Step 2: Extract student embeddings ─────────────────────────────────────
CKPT=${STUDENT_CKPT:-""}
python src/extract_student_embeddings.py ${CKPT:+--ckpt "$CKPT"}

echo "Student embedding extraction complete (job=$SLURM_JOB_ID)."
