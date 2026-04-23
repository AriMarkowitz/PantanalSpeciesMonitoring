#!/bin/bash
#SBATCH --job-name=nmf_global
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/nmf_global_%j.log
#SBATCH --error=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/nmf_global_%j.log

# Global NMFk on a stratified sample of student embeddings.
# Produces outputs/nmf_global/{W,W_pinv,H_all,k_sweep.csv,build_info.json}.
#
# Usage:
#   sbatch scripts/nmf_global.sh                 # build + project
#   STEP=build sbatch scripts/nmf_global.sh      # fit dictionary only
#   STEP=project sbatch scripts/nmf_global.sh    # project only (W must exist)

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
echo "---"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

STEP=${STEP:-all}

case "$STEP" in
    build)   python src/nmf_global.py build ;;
    project) python src/nmf_global.py project ;;
    all)     python src/nmf_global.py all ;;
    *)       echo "Unknown STEP: $STEP (expected build|project|all)"; exit 1 ;;
esac

echo "NMF global stage complete (job=$SLURM_JOB_ID)."
