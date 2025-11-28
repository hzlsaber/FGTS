#!/bin/bash
# Example: Train a linear probe with different token selection strategies

# Configuration
MODEL="dinov3_vit_7b"
TRAIN_GENERATOR="progan"  # Options: progan, ldm, sd14
CATEGORY="car"
TOKEN_STRATEGY="auto_fisher"  # Options: all, cls, reg, patch, cls+reg, top_fisher, auto_fisher
TOP_K=1

# Test dataset configuration
TEST_MODE="so-fake-ood"  # Options: so-fake-ood, GenImage, AIGCDetectionBenchmark

# Set TEST_BASE_DIR and TRAIN_FAKE_TYPE according to TEST_MODE
# IMPORTANT: TRAIN_FAKE_TYPE should match the generator used in training-free test
# Note: VAL_DATASET uses the same generator as TRAIN_DATASET
if [ "$TEST_MODE" = "GenImage" ]; then
    TEST_BASE_DIR="/mnt/nas_d/zhenglin/GenImage/1"
    TRAIN_FAKE_TYPE="1_fake_sd14"
elif [ "$TEST_MODE" = "AIGCDetectionBenchmark" ]; then
    TEST_BASE_DIR="/mnt/nas_d/zhenglin/AIGCDetectionBenchmark/AIGCDetectionBenchMark/test"
    TRAIN_FAKE_TYPE="1_fake"
else
    TEST_BASE_DIR="../datasets/test"
    TRAIN_FAKE_TYPE="1_fake_ldm"
fi

# Default train/val datasets (local FGTS paths)
TRAIN_DATASET="../datasets/train/${TRAIN_GENERATOR}"
VAL_DATASET="../datasets/val/${TRAIN_GENERATOR}"
MAX_TRAIN_SAMPLES=1000
MAX_VAL_SAMPLES=500


OUTPUT_DIR_FISHER="../results/${MODEL}_${TOKEN_STRATEGY}_${TEST_MODE}_linear_probe_fisher"
OUTPUT_DIR="../results/${MODEL}_${TOKEN_STRATEGY}_${TEST_MODE}_linear_probe"
OUTPUT_DIR_CUSTOM="../results/${MODEL}_${TOKEN_STRATEGY}_${TEST_MODE}_linear_probe_custom"
# Mode 1: Standard token strategies (all, cls, reg, patch, cls+reg)
# Train linear probe with all tokens

# python ../linear_probe_with_fisher.py \
#     --model $MODEL \
#     --train_dataset $TRAIN_DATASET \
#     --train_category $CATEGORY \
#     --train_fake_type $TRAIN_FAKE_TYPE \
#     --test_base_dir $TEST_BASE_DIR \
#     --test_category $CATEGORY \
#     --test_mode $TEST_MODE \
#     --token_strategy all \
#     --max_train_samples 1000 \
#     --max_test_samples 500 \
#     --num_epochs 50 \
#     --lr 0.01 \
#     --batch_size 32 \
#     --img_size 224 \
#     --output_dir $OUTPUT_DIR

# Mode 2: Auto Fisher mode (automatically compute Fisher scores and select top-k patch tokens)
# This mode:
# 1. Computes Fisher scores on training data
# 2. Automatically filters out CLS and register tokens
# 3. Selects top-k patch tokens only
# Recommended for best performance with DINOv3 models
# CUDA_VISIBLE_DEVICES="0" python ../linear_probe_with_fisher.py \
#     --model $MODEL \
#     --train_dataset $TRAIN_DATASET \
#     --val_dataset $VAL_DATASET \
#     --train_category $CATEGORY \
#     --train_fake_type $TRAIN_FAKE_TYPE \
#     --val_fake_type $TRAIN_FAKE_TYPE \
#     --test_base_dir $TEST_BASE_DIR \
#     --test_category $CATEGORY \
#     --test_mode $TEST_MODE \
#     --token_strategy auto_fisher \
#     --top_k $TOP_K \
#     --max_train_samples $MAX_TRAIN_SAMPLES \
#     --max_test_samples 6000 \
#     --num_epochs 50 \
#     --lr 0.01 \
#     --batch_size 32 \
#     --img_size 224 \
#     --output_dir "../results/${MODEL}_${TOKEN_STRATEGY}_${TEST_MODE}_linear_probe_fisher"
# echo "Training complete! Results saved to: $OUTPUT_DIR_FISHER"

# Mode 3: Use pre-computed Fisher scores
# If you already have Fisher scores from previous training:
# FISHER_SCORES="./results/previous_run/fisher_scores.npy"

# Mode 4: Use custom token indices
# Specify the custom token indices you want to use
CUSTOM_INDICES="100"

echo "Training with custom token indices: $CUSTOM_INDICES"

python ../linear_probe_with_fisher.py \
    --model $MODEL \
    --train_dataset $TRAIN_DATASET \
    --val_dataset $VAL_DATASET \
    --train_category $CATEGORY \
    --train_fake_type $TRAIN_FAKE_TYPE \
    --val_fake_type $TRAIN_FAKE_TYPE \
    --test_base_dir $TEST_BASE_DIR \
    --test_category $CATEGORY \
    --test_mode $TEST_MODE \
    --token_strategy custom_indices \
    --custom_token_indices $CUSTOM_INDICES \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --max_test_samples 500 \
    --max_val_samples $MAX_VAL_SAMPLES \
    --num_epochs 50 \
    --lr 0.01 \
    --batch_size 32 \
    --img_size 224 \
    --output_dir $OUTPUT_DIR_CUSTOM

echo "Completed training with custom indices. Results saved to: $OUTPUT_DIR_CUSTOM"

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

# ------------------------------------------------------------
# Mode 5: Evaluate using pre-trained UFD linear probe (skip training)
# ------------------------------------------------------------
# UFD_PROBE_CKPT="/home/zhenglin/UniversalFakeDetect/results/AIGCDetectionBenchmark/custom_AIGC_benchmark_2/linear_probe.pth"

# if [ -f "$UFD_PROBE_CKPT" ]; then
#     echo "Evaluating with existing UFD probe: $UFD_PROBE_CKPT"
#     CUDA_VISIBLE_DEVICES="0" python ../linear_probe_with_fisher.py \
#         --model $MODEL \
#         --train_dataset $TRAIN_DATASET \
#         --val_dataset $VAL_DATASET \
#         --train_category $CATEGORY \
#         --train_fake_type $TRAIN_FAKE_TYPE \
#         --val_fake_type $TRAIN_FAKE_TYPE \
#         --test_base_dir $TEST_BASE_DIR \
#         --test_category $CATEGORY \
#         --test_mode $TEST_MODE \
#         --token_strategy custom_indices \
#         --custom_token_indices "187,200,18,5,186,199,188,32,17,173" \
#         --max_train_samples $MAX_TRAIN_SAMPLES \
#         --max_val_samples $MAX_VAL_SAMPLES \
#         --max_test_samples 500 \
#         --num_epochs 0 \
#         --lr 0.01 \
#         --batch_size 32 \
#         --img_size 224 \
#         --output_dir "../results/${MODEL}_ufd_probe_eval" \
#         --probe_checkpoint "$UFD_PROBE_CKPT"
# else
#     echo "[Warn] UFD probe checkpoint not found: $UFD_PROBE_CKPT"
# fi
