#!/bin/bash
# Package artifacts for Kaggle dataset upload.
# Run this after training completes. Creates a tar.gz for upload.

PROJECT_DIR="$HOME/PantanalSpeciesMonitoring"
OUT_DIR="$PROJECT_DIR/outputs/kaggle_dataset/pantanal-artifacts"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/efficientnet_b1_birdset"

echo "Copying artifacts..."

# Student checkpoint
cp "$PROJECT_DIR/outputs/checkpoints/student/337377_fold0_cos=0.9196_ep47.pt" \
   "$OUT_DIR/student_best.pt"

# Classifier checkpoint — pick the most recently modified best_val_auc checkpoint
# across all training runs (not a hardcoded job ID, which goes stale on retrain).
BEST_CLS=$(ls -t "$PROJECT_DIR"/outputs/checkpoints/*/best_val_auc*.pt 2>/dev/null | head -1)
if [ -z "$BEST_CLS" ]; then
    echo "ERROR: no classifier checkpoint found under outputs/checkpoints/*/best_val_auc*.pt"
    exit 1
fi
echo "Selected classifier checkpoint: $BEST_CLS"
cp "$BEST_CLS" "$OUT_DIR/classifier_best.pt"

# Prototypes and projection
cp "$PROJECT_DIR/outputs/prototypes/global_prototypes.npz" "$OUT_DIR/"
cp "$PROJECT_DIR/outputs/prototypes/species_gmms.pkl" "$OUT_DIR/"
cp "$PROJECT_DIR/outputs/prototypes/supcon_W.npy" "$OUT_DIR/"

# EfficientNet backbone (for offline inference)
cp "$PROJECT_DIR/outputs/efficientnet_b1_birdset/"* "$OUT_DIR/efficientnet_b1_birdset/"

# Per-class NMF dictionaries (optional — only if built).
NMF_DIR="$PROJECT_DIR/outputs/nmf_per_class"
if [ -f "$NMF_DIR/W_all_pinv.npy" ]; then
    echo "Including per-class NMF dictionaries"
    cp "$NMF_DIR/W_all.npy"               "$OUT_DIR/"
    cp "$NMF_DIR/W_all_pinv.npy"          "$OUT_DIR/"
    cp "$NMF_DIR/species_boundaries.npy"  "$OUT_DIR/"
    cp "$NMF_DIR/species_order.json"      "$OUT_DIR/"
else
    echo "No per-class NMF dictionaries at $NMF_DIR — skipping"
fi

# Config and taxonomy
cp "$PROJECT_DIR/configs/default.yaml" "$OUT_DIR/config.yaml"
cp "$PROJECT_DIR/data/taxonomy.csv" "$OUT_DIR/"

echo "Artifact sizes:"
du -sh "$OUT_DIR"/*
echo "---"
du -sh "$OUT_DIR"

echo ""
echo "To upload to Kaggle:"
echo "  cd $PROJECT_DIR/outputs/kaggle_dataset"
echo "  kaggle datasets create -p pantanal-artifacts"
echo "Or tar it up:"
echo "  tar czf pantanal-artifacts.tar.gz -C $PROJECT_DIR/outputs/kaggle_dataset pantanal-artifacts"
