#!/bin/bash
#SBATCH --job-name=supcon_proj
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/supcon_%j.log
#SBATCH --error=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/supcon_%j.log

# Stage 1.5: Supervised contrastive projection of Perch embeddings.
# Trains a linear projection head so that same-species embeddings
# cluster tightly before HDBSCAN/GMM in Stage 2.

PROJECT_DIR="$HOME/PantanalSpeciesMonitoring"
ENV_NAME="pantanal"

mkdir -p "$PROJECT_DIR/outputs/logs"

module purge
module load miniconda3/latest
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Python: $(python --version)"
echo "---"

cd "$PROJECT_DIR"

python src/supcon_project.py

echo "SupCon projection training complete (job=$SLURM_JOB_ID)."
