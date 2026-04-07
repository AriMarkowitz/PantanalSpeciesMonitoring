#!/bin/bash
#SBATCH --job-name=motif_cluster
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/cluster_%j.log
#SBATCH --error=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/cluster_%j.log

# ── Project paths ──
PROJECT_DIR="$HOME/PantanalSpeciesMonitoring"
ENV_NAME="pantanal"

# ── Create log directory ──
mkdir -p "$PROJECT_DIR/outputs/logs"

# ── Load modules (CPU only — no CUDA needed) ──
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

# ── Auto-detect distill embeddings ──
DISTILL_H5="${DISTILL_H5:-outputs/embeddings/distill_embeddings.h5}"
DISTILL_ARGS=""
if [[ -f "$PROJECT_DIR/$DISTILL_H5" ]]; then
    echo "Distill embeddings found: $DISTILL_H5 — merging into clustering."
    DISTILL_ARGS="--with-distill --distill-h5 $DISTILL_H5"
else
    echo "No distill embeddings found at $DISTILL_H5 — running primary only."
fi

# ── Auto-detect SupCon projection ──
SUPCON_ARGS=""
if [[ -f "$PROJECT_DIR/outputs/prototypes/supcon_W.npy" ]]; then
    echo "SupCon projection found — clustering in projected space."
    SUPCON_ARGS="--supcon"
else
    echo "No SupCon projection — clustering in raw Perch space."
fi

# ── Stage 2: Clustering ──
echo "=== Stage 2: Motif Discovery ==="
python src/cluster.py --force $DISTILL_ARGS $SUPCON_ARGS

# ── Stage 3: Feature Extraction (runs on same node, fast) ──
echo "=== Stage 3: Feature Extraction ==="
python src/build_features.py

echo "Clustering + feature extraction complete (job=$SLURM_JOB_ID)."
