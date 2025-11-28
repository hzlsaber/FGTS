#!/bin/bash
# Example: Zero-shot deepfake detection

# Configuration
MODEL="dinov3_vits16"
TRAIN_GENERATOR="progan"  # Options: progan, ldm, sd14
REFERENCE_DATASET="../datasets/train/${TRAIN_GENERATOR}"
CATEGORY="car"
TOKEN_STRATEGY="all"  # Options: all, cls, reg, patch, cls+reg, top_fisher, auto_fisher, custom_indices

# Test dataset configuration
TEST_MODE="AIGCDetectionBenchmark"  # Options: so-fake-ood, GenImage, AIGCDetectionBenchmark

# Set TEST_BASE_DIR and REFERENCE_FAKE_TYPE according to TEST_MODE
# IMPORTANT: REFERENCE_FAKE_TYPE should match the generator used in training-free test
# Note: VAL_DATASET uses the same generator as REFERENCE_DATASET
if [ "$TEST_MODE" = "GenImage" ]; then
    TEST_BASE_DIR="/mnt/nas_d/zhenglin/GenImage/1"
    REFERENCE_FAKE_TYPE="1_fake_sd14"
elif [ "$TEST_MODE" = "AIGCDetectionBenchmark" ]; then
    TEST_BASE_DIR="/mnt/nas_d/zhenglin/AIGCDetectionBenchmark/AIGCDetectionBenchMark/test"
    REFERENCE_FAKE_TYPE="1_fake"
else
    TEST_BASE_DIR="../datasets/test"
    REFERENCE_FAKE_TYPE="1_fake_ldm"
fi

# Validation dataset uses the same generator as reference dataset
VAL_DATASET="../datasets/val/${TRAIN_GENERATOR}"

OUTPUT_DIR="../results/${MODEL}_${TOKEN_STRATEGY}_${TEST_MODE}_training_free"
OUTPUT_DIR_FISHER="../results/${MODEL}_${TOKEN_STRATEGY}_${TEST_MODE}_training_free_fisher"
OUTPUT_DIR_CUSTOM="../results/${MODEL}_${TOKEN_STRATEGY}_${TEST_MODE}_training_free_custom"

# Mode 1: Standard token strategies (all, cls, reg, patch, cls+reg)
# Run zero-shot detection
# python ../training_free_test.py \
#     --model $MODEL \
#     --reference_dataset $REFERENCE_DATASET \
#     --reference_category $CATEGORY \
#     --reference_fake_type $REFERENCE_FAKE_TYPE \
#     --test_base_dir $TEST_BASE_DIR \
#     --test_category $CATEGORY \
#     --test_mode $TEST_MODE \
#     --token_strategy $TOKEN_STRATEGY \
#     --max_reference 1000 \
#     --max_test 500 \
#     --batch_size 32 \
#     --img_size 224 \
#     --output_dir $OUTPUT_DIR

# Mode 2: Auto Fisher mode (automatically compute Fisher scores and select top-k patch tokens)
# This mode:
# 1. Computes Fisher scores on reference data
# 2. Automatically filters out CLS and register tokens
# 3. Selects top-k patch tokens only
CUDA_VISIBLE_DEVICES="1" python ../training_free_test.py \
    --model $MODEL \
    --reference_dataset $REFERENCE_DATASET \
    --reference_category $CATEGORY \
    --reference_fake_type $REFERENCE_FAKE_TYPE \
    --test_base_dir $TEST_BASE_DIR \
    --test_category $CATEGORY \
    --test_mode $TEST_MODE \
    --token_strategy auto_fisher \
    --top_k 10 \
    --max_reference 1000 \
    --max_test 6000 \
    --batch_size 32 \
    --img_size 224 \
    --output_dir $OUTPUT_DIR_FISHER

echo "Zero-shot detection complete! Results saved to: $OUTPUT_DIR_FISHER"

# Mode 3: Use pre-computed Fisher scores
# If you already have Fisher scores from linear probe training:
# FISHER_SCORES="./results/linear_probe_example/fisher_scores.npy"
# python ../training_free_test.py \
#     --model $MODEL \
#     --reference_dataset $REFERENCE_DATASET \
#     --reference_category $CATEGORY \
#     --reference_fake_type $REFERENCE_FAKE_TYPE \
#     --test_base_dir $TEST_BASE_DIR \
#     --test_category $CATEGORY \
#     --test_mode $TEST_MODE \
#     --token_strategy top_fisher \
#     --fisher_scores_path $FISHER_SCORES \
#     --top_k 10 \
#     --max_reference 1000 \
#     --max_test 500 \
#     --batch_size 32 \
#     --img_size 224 \
#     --output_dir $OUTPUT_DIR_CUSTOM

# Mode 4: Custom token indices
# Manually specify which token indices to use
# This is useful when you want to test specific tokens identified from previous experiments
# Example: Use tokens 187, 200, 18, 5 (comma-separated list)
# python ../training_free_test.py \
#     --model $MODEL \
#     --reference_dataset $REFERENCE_DATASET \
#     --reference_category $CATEGORY \
#     --reference_fake_type $REFERENCE_FAKE_TYPE \
#     --test_base_dir $TEST_BASE_DIR \
#     --test_category $CATEGORY \
#     --test_mode $TEST_MODE \
#     --token_strategy custom_indices \
#     --custom_token_indices "187,200,18,5,120,156,78,234,45,67" \
#     --max_reference 1000 \
#     --max_test 500 \
#     --batch_size 32 \
#     --img_size 224 \
#     --output_dir $OUTPUT_DIR_CUSTOM
#
# Note: Token indices must be valid for the model
# - For ViT-S/16 (224x224): 0 (CLS) + 4 (register, if any) + 196 (14x14 patches) = 200 or 201 tokens total
# - Token 0: CLS token
# - Tokens 1-4: Register tokens (DINOv2/v3 models with registers)
# - Remaining tokens: Patch tokens
# You can find optimal token indices from Fisher score analysis or linear probe training


# ============================================================
# Example configurations for different test modes
# ============================================================

# For GenImage dataset:
# TEST_MODE="GenImage"
# TEST_BASE_DIR="/mnt/nas_d/zhenglin/GenImage/1"
# CATEGORY=""  # Not used for GenImage mode

# For AIGCDetectionBenchmark dataset:
# TEST_MODE="AIGCDetectionBenchmark"
# TEST_BASE_DIR="/mnt/nas_d/zhenglin/AIGCDetectionBenchmark/AIGCDetectionBenchMark/test"
# CATEGORY=""  # Not used for AIGCDetectionBenchmark mode
