#!/bin/bash
# Train 4-model ensemble and evaluate on old-challenge-split test data
# Models 1&2 should already be training. This script trains models 3&4,
# then assembles ensemble and evaluates.
set -e

cd "/workspaces/snomed-ct-entity-linking/2nd Place"

echo "=== Training Model 3: fold=0, balanced weights ==="
python src/main.py --config-name=old_challenge_fold0 2>&1 | grep -E "Epoch|saving|chunked"

echo "=== Training Model 4: fold=3, boosted weights ==="
python src/main.py --config-name=old_challenge_fold3 2>&1 | grep -E "Epoch|saving|chunked"

echo "=== All training complete ==="

# Find best checkpoints from each output directory
echo "Collecting best checkpoints..."
ENSEMBLE_DIR="data/first_stage_ensemble"
rm -rf "$ENSEMBLE_DIR"
mkdir -p "$ENSEMBLE_DIR"

# Model 1: split=all, default weights
BEST1=$(ls -d output/02-16/*/models/S* 2>/dev/null | sort | tail -1)
echo "Model 1: $BEST1"
cp -r "$BEST1" "$ENSEMBLE_DIR/model1"

# Model 2: split=all, boosted weights
BEST2=$(ls -d output_boost/*/models/S* 2>/dev/null | sort | tail -1)
echo "Model 2: $BEST2"
cp -r "$BEST2" "$ENSEMBLE_DIR/model2"

# Model 3: fold=0, balanced weights
BEST3=$(ls -d output_fold0/*/models/S* 2>/dev/null | sort | tail -1)
echo "Model 3: $BEST3"
cp -r "$BEST3" "$ENSEMBLE_DIR/model3"

# Model 4: fold=3, boosted weights
BEST4=$(ls -d output_fold3/*/models/S* 2>/dev/null | sort | tail -1)
echo "Model 4: $BEST4"
cp -r "$BEST4" "$ENSEMBLE_DIR/model4"

echo "=== Running ensemble inference ==="
