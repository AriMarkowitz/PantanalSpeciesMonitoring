#!/bin/bash
#SBATCH --job-name=nmf_per_class
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/nmf_pc_%j.log
#SBATCH --error=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/nmf_pc_%j.log

# Per-species NMFk dictionary learning + pseudoinverse projection.
# Produces outputs/nmf_per_class/{W_all,W_all_pinv,species_boundaries,nmf_features}.npy
# and a species_info.csv with per-species k_sp + diagnostics.
#
# Usage:
#   sbatch scripts/nmf_per_class.sh                      # build dicts + project primary
#   STEP=build sbatch scripts/nmf_per_class.sh           # dicts only
#   STEP=project sbatch scripts/nmf_per_class.sh         # project only (dicts must exist)
#   STEP=project SOURCE=distill sbatch scripts/nmf_per_class.sh

PROJECT_DIR="$HOME/PantanalSpeciesMonitoring"
ENV_NAME="pantanal"

mkdir -p "$PROJECT_DIR/outputs/logs"

module purge
module load miniconda3/latest

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

cd "$PROJECT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Step: ${STEP:-all}"
echo "Source: ${SOURCE:-primary}"
echo "---"

# sklearn benefits from OMP/MKL parallelism at per-species NMF scale
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

STEP=${STEP:-all}
SOURCE=${SOURCE:-primary}

case "$STEP" in
    build)
        python src/nmf_per_class.py build-dicts
        ;;
    project)
        python src/nmf_per_class.py project --source "$SOURCE"
        ;;
    all)
        python src/nmf_per_class.py build-dicts
        python src/nmf_per_class.py project --source "$SOURCE"
        ;;
    *)
        echo "Unknown STEP: $STEP (expected build|project|all)"
        exit 1
        ;;
esac

echo "NMF per-class stage complete (job=$SLURM_JOB_ID)."
