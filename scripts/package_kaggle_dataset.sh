#!/bin/bash
# Package artifacts for Kaggle dataset upload (probing branch).
#
# Bundles: Student encoder + PrototypicalHead checkpoint + config + taxonomy
#          + EfficientNet-B1-BirdSet backbone (for offline inference).
#
# The old motif/GMM/SupCon artifacts are no longer needed — the probing
# pipeline goes mel → student → spatial → probe → logits, end of.

set -e

PROJECT_DIR="$HOME/PantanalSpeciesMonitoring"
OUT_DIR="$PROJECT_DIR/outputs/kaggle_dataset/pantanal-artifacts"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/efficientnet_b1_birdset"

echo "Copying artifacts..."

# ── Student checkpoint ──────────────────────────────────────────────────────
# Pick the most recently modified student checkpoint. Student training writes
# files like "337377_fold0_cos=0.9196_ep47.pt" into outputs/checkpoints/student/.
BEST_STUDENT=$(ls -t "$PROJECT_DIR"/outputs/checkpoints/student/*.pt 2>/dev/null | head -1)
if [ -z "$BEST_STUDENT" ]; then
    echo "ERROR: no student checkpoint found under outputs/checkpoints/student/*.pt"
    exit 1
fi
echo "Selected student checkpoint: $BEST_STUDENT"
cp "$BEST_STUDENT" "$OUT_DIR/student_best.pt"

# ── Prototypical probe checkpoint ───────────────────────────────────────────
# Pick the most recently modified probe fold checkpoint. Proto training writes
# fold{N}_best.pt into outputs/checkpoints/proto_probe/.
BEST_PROBE=$(ls -t "$PROJECT_DIR"/outputs/checkpoints/proto_probe/fold*_best.pt 2>/dev/null | head -1)
if [ -z "$BEST_PROBE" ]; then
    echo "ERROR: no probe checkpoint found under outputs/checkpoints/proto_probe/fold*_best.pt"
    echo "       Run: sbatch scripts/train_proto.sh"
    exit 1
fi
echo "Selected probe checkpoint: $BEST_PROBE"
cp "$BEST_PROBE" "$OUT_DIR/proto_probe_best.pt"

# ── EfficientNet backbone (for offline inference) ───────────────────────────
cp "$PROJECT_DIR/outputs/efficientnet_b1_birdset/"* "$OUT_DIR/efficientnet_b1_birdset/"

# ── Config + taxonomy ──────────────────────────────────────────────────────
cp "$PROJECT_DIR/configs/default.yaml" "$OUT_DIR/config.yaml"
cp "$PROJECT_DIR/data/taxonomy.csv" "$OUT_DIR/"

echo ""
echo "Artifact sizes:"
du -sh "$OUT_DIR"/*
echo "---"
du -sh "$OUT_DIR"

echo ""
echo "To upload to Kaggle:"
echo "  cd $PROJECT_DIR/outputs/kaggle_dataset"
echo "  kaggle datasets create -p pantanal-artifacts        # first time"
echo "  kaggle datasets version -p pantanal-artifacts -m 'msg'   # subsequent"
