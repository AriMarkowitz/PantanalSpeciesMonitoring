#!/bin/bash
#
# Upload artifacts to Kaggle, run inference notebook, and submit.
#
# Usage:
#   bash scripts/submit.sh
#
# Requirements: kaggle CLI configured (kaggle.json or ~/.kaggle/kaggle.json)

set -e

PROJECT_DIR="$HOME/PantanalSpeciesMonitoring"
KAGGLE_DS="$PROJECT_DIR/outputs/kaggle_dataset/pantanal-artifacts"
NOTEBOOK_DIR="$PROJECT_DIR/notebooks"
KERNEL_SLUG="arimarkowitz/birdclef-2026-pantanal-inference"
COMPETITION="birdclef-2026"

# ── Step 1: Package artifacts ────────────────────────────────────────────────
echo "=== Step 1: Packaging artifacts ==="
bash "$PROJECT_DIR/scripts/package_kaggle_dataset.sh"
echo ""

echo "Artifacts to upload:"
du -sh "$KAGGLE_DS"/*
echo ""

# ── Step 2: Upload dataset ───────────────────────────────────────────────────
echo "=== Step 2: Uploading dataset to Kaggle ==="

# Create dataset-metadata.json if it doesn't exist
if [ ! -f "$KAGGLE_DS/dataset-metadata.json" ]; then
    cat > "$KAGGLE_DS/dataset-metadata.json" <<EOF
{
  "title": "Pantanal Artifacts",
  "id": "arimarkowitz/pantanal-artifacts",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
fi

# Try version update first, fall back to create
if kaggle datasets status arimarkowitz/pantanal-artifacts 2>/dev/null | grep -qi "ready"; then
    kaggle datasets version -p "$KAGGLE_DS" -m "Updated artifacts" --dir-mode zip
else
    kaggle datasets create -p "$KAGGLE_DS" --dir-mode zip
fi
echo ""

# Wait for dataset to process
echo "Waiting for dataset to finish processing..."
for i in $(seq 1 30); do
    sleep 10
    STATUS=$(kaggle datasets status arimarkowitz/pantanal-artifacts 2>/dev/null || echo "unknown")
    if echo "$STATUS" | grep -qi "ready"; then
        echo "Dataset ready."
        break
    fi
    echo "  Still processing... (${i}/30)"
done
echo ""

# ── Step 3: Push and run notebook ────────────────────────────────────────────
echo "=== Step 3: Pushing notebook (triggers run) ==="
kaggle kernels push -p "$NOTEBOOK_DIR"
echo ""

# Wait for notebook to complete
echo "Waiting for notebook to complete..."
NOTEBOOK_COMPLETE=false
for i in $(seq 1 60); do
    sleep 30
    STATUS=$(kaggle kernels status "$KERNEL_SLUG" 2>&1)
    echo "  [$i] $STATUS"
    if echo "$STATUS" | grep -qi "complete"; then
        echo "Notebook finished successfully."
        NOTEBOOK_COMPLETE=true
        break
    fi
    if echo "$STATUS" | grep -qi "error\|cancel"; then
        echo "ERROR: Notebook failed. Check Kaggle for details."
        exit 1
    fi
done

if [ "$NOTEBOOK_COMPLETE" != "true" ]; then
    echo "ERROR: Notebook did not complete within 30 minutes."
    exit 1
fi

# ── Step 4: Submit via UI ────────────────────────────────────────────────────
echo ""
echo "=== Step 4: Submit via Kaggle UI ==="
echo ""
echo "Notebook ran successfully. Submit to competition:"
echo ""
echo "  1. Go to: https://www.kaggle.com/code/arimarkowitz/birdclef-2026-pantanal-inference"
echo "  2. Click 'Output' tab → latest version"
echo "  3. Click 'Submit to Competition'"
echo ""
echo "=== Done! ==="
