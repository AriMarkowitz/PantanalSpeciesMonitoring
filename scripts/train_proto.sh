#!/bin/bash
#SBATCH --job-name=proto_probe
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/proto_%j.log
#SBATCH --error=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/proto_%j.log

# Prototypical probing head on frozen Student spatial embeddings.
#
# Usage:
#   sbatch scripts/train_proto.sh                  # fold 0
#   FOLD=2 sbatch scripts/train_proto.sh           # specific fold

PROJECT_DIR="$HOME/PantanalSpeciesMonitoring"
ENV_NAME="pantanal"

mkdir -p "$PROJECT_DIR/outputs/logs"

module purge
module load cuda/12.8.0-dh6b
module load miniconda3/latest

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

cd "$PROJECT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Fold: ${FOLD:-0}"
echo "---"

export FOLD="${FOLD:-0}"

python src/train_proto.py --config configs/default.yaml

echo "Proto probe training complete (job=$SLURM_JOB_ID, fold=$FOLD)."
