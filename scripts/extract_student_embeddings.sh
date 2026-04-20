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

echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "---"

# Auto-select best student checkpoint
CKPT=${STUDENT_CKPT:-""}
python src/extract_student_embeddings.py ${CKPT:+--ckpt "$CKPT"}

# Clean up mel caches (48GB) — no longer needed after extraction
echo "Cleaning up mel caches..."
rm -f outputs/embeddings/embeddings.h5.mels.npy
rm -f outputs/embeddings/distill_embeddings.h5.mels.npy
echo "Student embedding extraction complete (job=$SLURM_JOB_ID)."
