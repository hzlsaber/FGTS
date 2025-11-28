#!/bin/bash
# 测试所有数据选择策略

# Configuration
MODEL="dinov3_vit_7b"
TRAIN_GENERATOR="progan"
CATEGORY="car"
TOKEN_STRATEGY="auto_fisher"
TOP_K=10
TEST_MODE="AIGCDetectionBenchmark"
TEST_BASE_DIR="/mnt/nas_d/zhenglin/AIGCDetectionBenchmark/AIGCDetectionBenchMark/test"
TRAIN_FAKE_TYPE="1_fake"  # Will be overridden for each strategy

# Validation dataset
VAL_DATASET="../datasets/val/${TRAIN_GENERATOR}"

# Strategies to test
STRATEGIES=("quality_first" "diversity_first" "balanced" "first_n")

echo "========================================"
echo "Testing All Data Selection Strategies"
echo "Model: $MODEL"
echo "Category: $CATEGORY"
echo "Strategies: ${STRATEGIES[@]}"
echo "========================================"
echo ""

# Test each strategy
for strategy in "${STRATEGIES[@]}"; do
    echo ""
    echo "========================================"
    echo "Testing strategy: $strategy"
    echo "========================================"
    echo ""

    TRAIN_DATASET="../datasets/train/${TRAIN_GENERATOR}"
    TRAIN_FAKE_TYPE="1_fake_${strategy}"
    OUTPUT_DIR="../results/${MODEL}_${TOKEN_STRATEGY}_${TEST_MODE}_${strategy}"

    echo "Training dataset: $TRAIN_DATASET"
    echo "Fake type: $TRAIN_FAKE_TYPE"
    echo "Output directory: $OUTPUT_DIR"
    echo ""

    # Run training
    CUDA_VISIBLE_DEVICES="1" python ../linear_probe_with_fisher.py \
        --model $MODEL \
        --train_dataset $TRAIN_DATASET \
        --val_dataset $VAL_DATASET \
        --train_category $CATEGORY \
        --train_fake_type $TRAIN_FAKE_TYPE \
        --test_base_dir $TEST_BASE_DIR \
        --test_category $CATEGORY \
        --test_mode $TEST_MODE \
        --token_strategy $TOKEN_STRATEGY \
        --top_k $TOP_K \
        --max_train_samples 1000 \
        --max_test_samples 500 \
        --num_epochs 50 \
        --lr 0.01 \
        --batch_size 16 \
        --img_size 224 \
        --output_dir $OUTPUT_DIR

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Strategy $strategy completed successfully!"
        echo "Results saved to: $OUTPUT_DIR"
    else
        echo ""
        echo "✗ Strategy $strategy failed!"
    fi

    echo ""
    echo "========================================"
    echo ""

    # Small delay between runs
    sleep 5
done

echo ""
echo "========================================"
echo "All strategies tested!"
echo "========================================"
echo ""
echo "Results comparison:"
echo ""

# Compare results
for strategy in "${STRATEGIES[@]}"; do
    REPORT="../results/${MODEL}_${TOKEN_STRATEGY}_${TEST_MODE}_${strategy}/report.txt"
    if [ -f "$REPORT" ]; then
        echo "=== Strategy: $strategy ==="
        grep "平均值" "$REPORT" || grep "Average" "$REPORT" || echo "Report not found"
        echo ""
    fi
done

echo "========================================"
echo "Next steps:"
echo "1. Check the results above"
echo "2. Pick the best strategy"
echo "3. Use it for your final experiments"
echo "========================================"
