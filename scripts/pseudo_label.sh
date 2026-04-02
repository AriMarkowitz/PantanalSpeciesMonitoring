#!/bin/bash
#SBATCH --job-name=pseudo_label
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/pseudo_%j.log
#SBATCH --error=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/pseudo_%j.log

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

export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3

ROUND=${ROUND:-1}
CHECKPOINT=${CHECKPOINT:-""}

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Round: $ROUND"
echo "Checkpoint: ${CHECKPOINT:-none}"
echo "---"

if [[ "$ROUND" -eq 1 ]]; then
    python src/pseudo_label.py --round 1
else
    if [[ -z "$CHECKPOINT" ]]; then
        echo "ERROR: CHECKPOINT env var required for round >= 2"
        exit 1
    fi
    python src/pseudo_label.py --round "$ROUND" --checkpoint "$CHECKPOINT"
fi

echo "Pseudo-labeling round $ROUND complete (job=$SLURM_JOB_ID)."
