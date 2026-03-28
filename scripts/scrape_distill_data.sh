#!/bin/bash
#SBATCH --job-name=distill_scrape
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/distill_scrape_%j.log
#SBATCH --error=outputs/logs/distill_scrape_%j.log

# ── Project paths ──
PROJECT_DIR="$HOME/PantanalSpeciesMonitoring"
ENV_NAME="pantanal"

# ── Runtime defaults (override via environment variables at sbatch) ──
OUTPUT_DIR="${OUTPUT_DIR:-data/distill_audio}"
MANIFEST_PATH="${MANIFEST_PATH:-data/distill_manifest.csv}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# iNat defaults
INAT_ENABLE="${INAT_ENABLE:-1}"
INAT_MAX_FILES="${INAT_MAX_FILES:-8000}"
INAT_SPLITS="${INAT_SPLITS:-train}"
INAT_SUPERCATEGORIES="${INAT_SUPERCATEGORIES:-aves,amphibia,insecta,mammalia,reptilia}"
INAT_BBOX="${INAT_BBOX:--25,-60,-10,-45}"
INAT_MIN_DURATION="${INAT_MIN_DURATION:-2.0}"

# BirdSet defaults
BIRDSET_ENABLE="${BIRDSET_ENABLE:-1}"
BIRDSET_CONFIG="${BIRDSET_CONFIG:-PER}"
BIRDSET_SPLIT="${BIRDSET_SPLIT:-train}"
BIRDSET_MAX_FILES="${BIRDSET_MAX_FILES:-3000}"
BIRDSET_ALLOWLIST="${BIRDSET_ALLOWLIST:-}"
BIRDSET_ISOLATED="${BIRDSET_ISOLATED:-1}"
BIRDSET_VENV_DIR="${BIRDSET_VENV_DIR:-$PROJECT_DIR/.venv_birdset_scrape}"

mkdir -p "$PROJECT_DIR/outputs/logs"

# ── Load modules ──
module purge
module load miniconda3/latest

# ── Activate conda ──
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

cd "$PROJECT_DIR"

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-N/A}"
echo "Python: $(python --version)"
echo "Output dir: $OUTPUT_DIR"
echo "Manifest: $MANIFEST_PATH"
echo "iNat enabled: $INAT_ENABLE"
echo "BirdSet enabled: $BIRDSET_ENABLE"
echo "BirdSet isolated venv: $BIRDSET_ISOLATED"
echo "---"

# BirdSet path requires datasets package.
# To avoid dependency churn in the main conda env, use an isolated venv by default.
INAT_PYTHON="python"
BIRDSET_PYTHON="python"

if [[ "$BIRDSET_ENABLE" == "1" ]]; then
    if [[ "$BIRDSET_ISOLATED" == "1" ]]; then
        echo "Preparing isolated BirdSet venv: $BIRDSET_VENV_DIR"
        if [[ ! -d "$BIRDSET_VENV_DIR" ]]; then
            python -m venv "$BIRDSET_VENV_DIR"
        fi
        BIRDSET_PYTHON="$BIRDSET_VENV_DIR/bin/python"

        if ! "$BIRDSET_PYTHON" -c "import datasets, soundfile, tqdm, resampy" >/dev/null 2>&1; then
            "$BIRDSET_PYTHON" -m pip install --upgrade pip
            "$BIRDSET_PYTHON" -m pip install "datasets<=3.6.0" soundfile tqdm resampy
        fi
    else
        if ! python -c "import datasets" >/dev/null 2>&1; then
            echo "Installing required package for BirdSet in main env: datasets<=3.6.0"
            pip install "datasets<=3.6.0"
        fi
    fi
fi

COMMON_ARGS=(
    --output-dir "$OUTPUT_DIR"
    --manifest-path "$MANIFEST_PATH"
    --log-level INFO
)

if [[ "$SKIP_EXISTING" == "1" ]]; then
    COMMON_ARGS+=(--skip-existing)
fi

if [[ "$INAT_ENABLE" == "1" ]]; then
    # Skip iNat entirely if the manifest already has iNat rows
    INAT_DONE=0
    if [[ -f "$PROJECT_DIR/$MANIFEST_PATH" ]] && grep -q '^inat,' "$PROJECT_DIR/$MANIFEST_PATH" 2>/dev/null; then
        INAT_COUNT=$(grep -c '^inat,' "$PROJECT_DIR/$MANIFEST_PATH" || true)
        echo "[1/2] iNat already collected ($INAT_COUNT rows in manifest) — skipping."
        INAT_DONE=1
    fi

    if [[ "$INAT_DONE" == "0" ]]; then
        echo "[1/2] Collecting iNat subset..."
        "$INAT_PYTHON" src/scrape_distill_data.py \
            --inat \
            --inat-max-files "$INAT_MAX_FILES" \
            --inat-splits "$INAT_SPLITS" \
            --inat-supercategories "$INAT_SUPERCATEGORIES" \
            --bbox="$INAT_BBOX" \
            --inat-min-duration "$INAT_MIN_DURATION" \
            "${COMMON_ARGS[@]}"
    fi
fi

if [[ "$BIRDSET_ENABLE" == "1" ]]; then
    # Skip BirdSet entirely if it already has enough rows in the manifest
    BIRDSET_DONE=0
    if [[ -f "$PROJECT_DIR/$MANIFEST_PATH" ]] && grep -q '^birdset,' "$PROJECT_DIR/$MANIFEST_PATH" 2>/dev/null; then
        BIRDSET_COUNT=$(grep -c '^birdset,' "$PROJECT_DIR/$MANIFEST_PATH" || true)
        if [[ "$BIRDSET_COUNT" -ge "$BIRDSET_MAX_FILES" ]]; then
            echo "[2/2] BirdSet already collected ($BIRDSET_COUNT rows in manifest, target=$BIRDSET_MAX_FILES) — skipping."
            BIRDSET_DONE=1
        else
            echo "[2/2] BirdSet partially collected ($BIRDSET_COUNT/$BIRDSET_MAX_FILES rows) — resuming..."
        fi
    fi

    if [[ "$BIRDSET_DONE" == "0" ]]; then
        BIRDSET_ARGS=(
            --birdset
            --birdset-config "$BIRDSET_CONFIG"
            --birdset-split "$BIRDSET_SPLIT"
            --birdset-max-files "$BIRDSET_MAX_FILES"
        )
        if [[ -n "$BIRDSET_ALLOWLIST" ]]; then
            BIRDSET_ARGS+=(--birdset-species-allowlist "$BIRDSET_ALLOWLIST")
        fi

        "$BIRDSET_PYTHON" src/scrape_distill_data.py \
            "${BIRDSET_ARGS[@]}" \
            "${COMMON_ARGS[@]}"
    fi
fi

echo "Distillation data scrape complete (job=${SLURM_JOB_ID:-N/A})."
