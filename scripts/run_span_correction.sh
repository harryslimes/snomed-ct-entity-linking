#!/bin/bash
# End-to-end pipeline for span correction model

set -e  # Exit on error

echo "================================================================"
echo "Span Correction Pipeline"
echo "================================================================"

# Configuration
DATA_DIR="data/old-challenge-split"
KIRI_PRED="$DATA_DIR/kiri_super_pred.csv"
GOLD_ANNOTATIONS="$DATA_DIR/train_annotations.csv"
NOTES="$DATA_DIR/train_notes.csv"
SPAN_CORRECTION_DATA="data/span_correction"
OUTPUT_DIR="span_correction/output"

# Step 1: Generate training data
echo ""
echo "Step 1: Generating training data..."
echo "================================================================"

if [ ! -f "$SPAN_CORRECTION_DATA/train_fold_0.csv" ]; then
    python -m span_correction.data_prep \
        --kiri-pred "$KIRI_PRED" \
        --gold "$GOLD_ANNOTATIONS" \
        --notes "$NOTES" \
        --output "$SPAN_CORRECTION_DATA/training_data_full.csv" \
        --context-size 200 \
        --iou-threshold 0.3 \
        --max-delta 100

    echo "Training data generated. Now split into folds..."
    # TODO: Add fold splitting logic here if needed
    # For now, assume folds are pre-split
else
    echo "Training data already exists. Skipping..."
fi

# Step 2: Train model (4-fold CV)
echo ""
echo "Step 2: Training models (4-fold cross-validation)..."
echo "================================================================"

for fold in 0 1 2 3; do
    echo ""
    echo "Training fold $fold..."

    if [ ! -f "$OUTPUT_DIR/fold_$fold/best_model.pt" ]; then
        python -m span_correction.main split=$fold
    else
        echo "Model for fold $fold already trained. Skipping..."
    fi
done

# Step 3: Threshold tuning (on validation fold 0)
echo ""
echo "Step 3: Tuning p_exist threshold on validation set..."
echo "================================================================"

VAL_PRED="$DATA_DIR/kiri_super_pred_val_fold_0.csv"
VAL_GOLD="$DATA_DIR/val_annotations_fold_0.csv"
VAL_NOTES="$DATA_DIR/val_notes_fold_0.csv"
MODEL_CHECKPOINT="$OUTPUT_DIR/fold_0/best_model.pt"

python -m span_correction.inference \
    --kiri-pred "$VAL_PRED" \
    --notes "$VAL_NOTES" \
    --model "$MODEL_CHECKPOINT" \
    --output "$OUTPUT_DIR/threshold_tuning_output.csv" \
    --tune-threshold \
    --gold "$VAL_GOLD" \
    --batch-size 32

# Extract best threshold from results
BEST_THRESHOLD=$(python -c "import pandas as pd; df = pd.read_csv('threshold_tuning_results.csv'); print(df.loc[df['iou'].idxmax(), 'threshold'])")
echo "Best threshold: $BEST_THRESHOLD"

# Step 4: Inference on test set
echo ""
echo "Step 4: Running inference on test set..."
echo "================================================================"

TEST_PRED="$DATA_DIR/kiri_super_pred_test.csv"
TEST_NOTES="$DATA_DIR/test_notes.csv"
TEST_OUTPUT="$OUTPUT_DIR/corrected_test_predictions.csv"

python -m span_correction.inference \
    --kiri-pred "$TEST_PRED" \
    --notes "$TEST_NOTES" \
    --model "$MODEL_CHECKPOINT" \
    --output "$TEST_OUTPUT" \
    --threshold "$BEST_THRESHOLD" \
    --batch-size 32

# Step 5: Evaluation (if test gold annotations available)
echo ""
echo "Step 5: Evaluation..."
echo "================================================================"

TEST_GOLD="$DATA_DIR/test_annotations.csv"

if [ -f "$TEST_GOLD" ]; then
    python -m span_correction.evaluate \
        --baseline "$TEST_PRED" \
        --corrected "$TEST_OUTPUT" \
        --gold "$TEST_GOLD" \
        --output-report "$OUTPUT_DIR/evaluation_report.csv"
else
    echo "Test gold annotations not available. Skipping evaluation..."
fi

echo ""
echo "================================================================"
echo "Pipeline complete!"
echo "================================================================"
echo "Corrected predictions: $TEST_OUTPUT"
echo "Competition format: ${TEST_OUTPUT%.csv}_submission.csv"
