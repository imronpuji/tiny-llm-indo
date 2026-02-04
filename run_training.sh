#!/bin/bash

echo "======================================================"
echo "🚀 Fine-tuning Indonesian QA Model"
echo "======================================================"

# Step 1: Augmentasi Data (Merge + Augment)
echo ""
echo "Step 1: Augmenting datasets..."
echo "------------------------------------------------------"
python augment_indonesian_qa.py \
    --datasets ernaamaliaw/Indonesian-General-QA Azzindani/Indonesian_Regulation_QA \
    --output ./indonesian_qa_augmented \
    --augment-ratio 0.3

if [ $? -ne 0 ]; then
    echo "❌ Data augmentation failed!"
    exit 1
fi

echo ""
echo "✅ Data augmentation completed!"
echo ""

# Step 2: Fine-tuning Model
echo "Step 2: Fine-tuning model..."
echo "------------------------------------------------------"
python finetune_indonesian_qa.py \
    --base-model yasmeenimron/masa-ai \
    --output-dir ./masa-ai-qa-indonesian-v2 \
    --batch-size 8 \
    --epochs 10 \
    --learning-rate 2e-5

if [ $? -ne 0 ]; then
    echo "❌ Fine-tuning failed!"
    exit 1
fi

echo ""
echo "✅ Fine-tuning completed!"
echo ""

# Step 3: Test Model
echo "Step 3: Testing model..."
echo "------------------------------------------------------"
python finetune_indonesian_qa.py --test-only

echo ""
echo "======================================================"
echo "✅ All steps completed!"
echo "======================================================"
echo ""
echo "Model saved to: ./masa-ai-qa-indonesian-v2"
echo ""
echo "To test interactively:"
echo "  python test_masa_ai.py"
