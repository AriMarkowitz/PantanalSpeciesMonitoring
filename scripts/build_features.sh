#!/bin/bash
#SBATCH --job-name=build_features
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/build_features_%j.log
#SBATCH --error=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/build_features_%j.log

# ── Project paths ──
PROJECT_DIR="$HOME/PantanalSpeciesMonitoring"
ENV_NAME="pantanal"

# ── Create log directory ──
mkdir -p "$PROJECT_DIR/outputs/logs"

# ── Load modules (CPU only) ──
module purge
module load miniconda3/latest

# ── Activate conda ──
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# ── Print environment info ──
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Python: $(python --version)"
echo "---"

cd "$PROJECT_DIR"

# ── Stage 3: Feature Extraction ──
echo "=== Stage 3: Feature Extraction ==="
python src/build_features.py ${STUDENT:+--student}

echo "Feature extraction complete (job=$SLURM_JOB_ID)."
