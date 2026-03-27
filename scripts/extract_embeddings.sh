#!/bin/bash
#SBATCH --job-name=perch_embed
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/embed_%j.log
#SBATCH --error=outputs/logs/embed_%j.log

# ── Project paths ──
PROJECT_DIR="$HOME/PantanalSpeciesMonitoring"
ENV_NAME="pantanal"

# ── Create log directory ──
mkdir -p "$PROJECT_DIR/outputs/logs"

# ── Load modules ──
module purge
module load cuda/12.8.0-dh6b
module load miniconda3/latest

# ── Activate conda ──
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ── Fix CuDNN: prefer pip-installed NVIDIA libs over system module ──
# tensorflow[and-cuda] bundles cuDNN 9.3; the cuda module ships 9.1 which is too old.
NVIDIA_SITE="$(python -c 'import nvidia.cudnn; print(nvidia.cudnn.__path__[0])' 2>/dev/null)/lib"
if [[ -d "$NVIDIA_SITE" ]]; then
    export LD_LIBRARY_PATH="$NVIDIA_SITE:${LD_LIBRARY_PATH:-}"
    echo "Prepended pip cuDNN to LD_LIBRARY_PATH: $NVIDIA_SITE"
fi

# ── Suppress cosmetic TF / abseil warnings ──
export TF_ENABLE_ONEDNN_OPTS=0          # silence oneDNN info messages
export TF_CPP_MIN_LOG_LEVEL=1           # hide TF INFO-level C++ logs (keep WARN+)
export KAGGLEHUB_VERBOSITY=error        # silence kagglehub upgrade nag

# ── Print environment info ──
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'N/A')"
echo "---"

cd "$PROJECT_DIR"

# ── Run embedding extraction ──
# Supports OVERRIDE env var for config overrides, e.g.:
#   OVERRIDE="stage1.batch_size=128" sbatch scripts/extract_embeddings.sh
python src/extract_embeddings.py --config configs/default.yaml

echo "Embedding extraction complete (job=$SLURM_JOB_ID)."
