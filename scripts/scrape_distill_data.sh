#!/bin/bash
#SBATCH --job-name=distill_scrape
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/distill_scrape_%j.log
#SBATCH --error=/users/admarkowitz/PantanalSpeciesMonitoring/outputs/logs/distill_scrape_%j.log

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

# BirdSet defaults — geo streaming mode (preferred)
BIRDSET_ENABLE="${BIRDSET_ENABLE:-1}"
# All 10 configs; geo-streaming filters to Pantanal bbox before downloading
BIRDSET_CONFIGS="${BIRDSET_CONFIGS:-PER,NES,UHH,HSN,NBP,POW,SSW,SNE,XCM,XCL}"
BIRDSET_SPLIT="${BIRDSET_SPLIT:-train}"
BIRDSET_GEO_MAX_PER_CONFIG="${BIRDSET_GEO_MAX_PER_CONFIG:-5000}"   # per config, geo-filtered
BIRDSET_GEO_BBOX="${BIRDSET_GEO_BBOX:--22,-62,-10,-44}"            # Pantanal region
BIRDSET_REQUIRE_DETECTED_EVENTS="${BIRDSET_REQUIRE_DETECTED_EVENTS:-1}"
# PER and NES are region-specific configs — take all clips, no bbox filter needed
BIRDSET_NO_BBOX_CONFIGS="${BIRDSET_NO_BBOX_CONFIGS:-PER,NES}"
TRAIN_CSV="${TRAIN_CSV:-data/train.csv}"
# Legacy non-geo mode settings (only used if BIRDSET_GEO_MODE=0)
BIRDSET_MAX_FILES="${BIRDSET_MAX_FILES:-3000}"
BIRDSET_ALLOWLIST="${BIRDSET_ALLOWLIST:-}"
BIRDSET_GEO_MODE="${BIRDSET_GEO_MODE:-1}"
BIRDSET_ISOLATED="${BIRDSET_ISOLATED:-1}"
BIRDSET_VENV_DIR="${BIRDSET_VENV_DIR:-$PROJECT_DIR/.venv_birdset_scrape}"
TAXONOMY_CACHE_DIR="${TAXONOMY_CACHE_DIR:-data}"
REPAIR_MANIFEST="${REPAIR_MANIFEST:-0}"

mkdir -p "$PROJECT_DIR/outputs/logs"

# ── Load modules ──
module purge
module load miniconda3/latest

# ── Activate conda ──
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

cd "$PROJECT_DIR"

# Route HuggingFace cache to /tmp (scratch) to avoid filling home quota.
# Each BirdSet config downloads ~3-4GB of shards that can be discarded after.
export HF_DATASETS_CACHE="/tmp/hf_datasets_${SLURM_JOB_ID:-$$}"
export HF_HOME="/tmp/hf_home_${SLURM_JOB_ID:-$$}"
mkdir -p "$HF_DATASETS_CACHE" "$HF_HOME"
echo "HF cache → $HF_DATASETS_CACHE"

# Pass HF token if set in environment (avoids rate-limit warnings in isolated venv)
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
    echo "HF_TOKEN: set (authenticated)"
elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
    echo "HF_TOKEN: loaded from ~/.cache/huggingface/token"
else
    echo "HF_TOKEN: not set (unauthenticated — rate limits may apply)"
fi

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-N/A}"
echo "Python: $(python --version)"
echo "Output dir: $OUTPUT_DIR"
echo "Manifest: $MANIFEST_PATH"
echo "iNat enabled: $INAT_ENABLE (max=$INAT_MAX_FILES, bbox=$INAT_BBOX)"
echo "BirdSet enabled: $BIRDSET_ENABLE (configs: $BIRDSET_CONFIGS)"
echo "BirdSet geo mode: $BIRDSET_GEO_MODE (bbox=$BIRDSET_GEO_BBOX, max/config=$BIRDSET_GEO_MAX_PER_CONFIG)"
echo "BirdSet require detected_events: $BIRDSET_REQUIRE_DETECTED_EVENTS"
echo "Train CSV for XC dedup: $TRAIN_CSV"
echo "BirdSet isolated venv: $BIRDSET_ISOLATED"
echo "Repair manifest: $REPAIR_MANIFEST"
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
    --taxonomy-cache-dir "$TAXONOMY_CACHE_DIR"
    --log-level INFO
)

if [[ "$SKIP_EXISTING" == "1" ]]; then
    COMMON_ARGS+=(--skip-existing)
fi

# ── Step 0: Repair existing manifest BirdSet labels if requested ──────────────
if [[ "$REPAIR_MANIFEST" == "1" ]]; then
    echo "[0/2] Repairing BirdSet integer labels in manifest..."
    "$BIRDSET_PYTHON" src/scrape_distill_data.py \
        --repair-manifest \
        "${COMMON_ARGS[@]}"
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
    BIRDSET_COUNT=0
    if [[ -f "$PROJECT_DIR/$MANIFEST_PATH" ]]; then
        BIRDSET_COUNT=$(grep -c '^birdset,' "$PROJECT_DIR/$MANIFEST_PATH" 2>/dev/null || true)
    fi

    if [[ "$BIRDSET_GEO_MODE" == "1" ]]; then
        # ── Geographic streaming mode (preferred) ──────────────────────────
        N_CONFIGS=$(echo "$BIRDSET_CONFIGS" | tr ',' '\n' | wc -l)
        BIRDSET_TARGET=$(( BIRDSET_GEO_MAX_PER_CONFIG * N_CONFIGS ))
        echo "[2/2] BirdSet geo-streaming: configs=$BIRDSET_CONFIGS, target~$BIRDSET_TARGET clips"

        BIRDSET_ARGS=(
            --birdset-geo
            --birdset-configs "$BIRDSET_CONFIGS"
            --birdset-split "$BIRDSET_SPLIT"
            --birdset-geo-max-per-config "$BIRDSET_GEO_MAX_PER_CONFIG"
            --birdset-geo-bbox="$BIRDSET_GEO_BBOX"
            --train-csv "$TRAIN_CSV"
        )
        if [[ "$BIRDSET_REQUIRE_DETECTED_EVENTS" == "1" ]]; then
            BIRDSET_ARGS+=(--birdset-require-detected-events)
        fi
        if [[ -n "$BIRDSET_NO_BBOX_CONFIGS" ]]; then
            BIRDSET_ARGS+=(--birdset-no-bbox-configs "$BIRDSET_NO_BBOX_CONFIGS")
        fi

        "$BIRDSET_PYTHON" src/scrape_distill_data.py \
            "${BIRDSET_ARGS[@]}" \
            "${COMMON_ARGS[@]}"
    else
        # ── Legacy non-streaming mode ──────────────────────────────────────
        N_CONFIGS=$(echo "$BIRDSET_CONFIGS" | tr ',' '\n' | wc -l)
        BIRDSET_TARGET=$(( BIRDSET_MAX_FILES * N_CONFIGS ))
        if [[ "$BIRDSET_COUNT" -ge "$BIRDSET_TARGET" ]]; then
            echo "[2/2] BirdSet already collected ($BIRDSET_COUNT rows, target=$BIRDSET_TARGET) — skipping."
        else
            echo "[2/2] BirdSet (legacy): $BIRDSET_COUNT/$BIRDSET_TARGET rows, resuming..."
            BIRDSET_ARGS=(
                --birdset
                --birdset-configs "$BIRDSET_CONFIGS"
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
fi

echo "Distillation data scrape complete (job=${SLURM_JOB_ID:-N/A})."
