#!/bin/bash
#SBATCH --job-name=cache_mels
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/cache_mels_%j.log
#SBATCH --error=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/cache_mels_%j.log

# Pre-cache mel spectrograms into embeddings HDF5.
# CPU-only, I/O-heavy — uses 16 workers for parallel audio loading.
# Run once before student distillation training.

PROJECT_DIR="$HOME/PantanalSpeciesMonitoring"
ENV_NAME="pantanal"

mkdir -p "$PROJECT_DIR/outputs/logs"

module purge
module load miniconda3/latest
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

export PYTHONUNBUFFERED=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "---"

cd "$PROJECT_DIR"

python src/cache_mels.py --distill --num-workers 16

echo "Mel caching complete (job=$SLURM_JOB_ID)."
