#!/bin/bash
# Example: Evaluate a trained linear probe checkpoint

# Configuration
CHECKPOINT="./results/linear_probe_example/linear_probe.pth"
MODEL="dinov3_vitl16"
TEST_BASE_DIR="./datasets/test"
CATEGORY="car"
OUTPUT_DIR="./results/evaluation_example"

# Run evaluation
python evaluate_trained_model.py \
    --checkpoint $CHECKPOINT \
    --model $MODEL \
    --test_base_dir $TEST_BASE_DIR \
    --test_category $CATEGORY \
    --max_test_samples 500 \
    --batch_size 32 \
    --img_size 224 \
    --output_dir $OUTPUT_DIR

echo "Evaluation complete! Results saved to: $OUTPUT_DIR"
