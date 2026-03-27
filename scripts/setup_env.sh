#!/bin/bash
# One-time environment setup for PantanalSpeciesMonitoring.
# Run interactively (not via sbatch): bash scripts/setup_env.sh

set -euo pipefail

PROJECT_DIR="$HOME/PantanalSpeciesMonitoring"
ENV_NAME="pantanal"

module purge
module load cuda/12.8.0-dh6b
module load miniconda3/latest
eval "$(conda shell.bash hook)"

if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Activate with: conda activate $ENV_NAME"
    echo "To recreate, run: conda env remove -n $ENV_NAME"
    exit 0
fi

echo "Creating conda environment from environment.yml"
conda env create -f "$PROJECT_DIR/environment.yml"
conda activate "$ENV_NAME"

# Verify GPU access
echo "=== Verification ==="
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'TensorFlow {tf.__version__}: {len(gpus)} GPU(s) detected')
if not gpus:
    print('WARNING: No GPUs detected. Perch embedding extraction requires GPU.')
    print('Check CUDA module compatibility.')
"

python -c "
from perch_hoplite.zoo import model_configs
print('perch_hoplite imported successfully')
print('Available models:', list(model_configs.MODEL_CONFIGS.keys()) if hasattr(model_configs, 'MODEL_CONFIGS') else 'check manually')
"

echo "=== Setup complete ==="
echo "Activate with: conda activate $ENV_NAME"
