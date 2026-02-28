#!/bin/bash

# ============================================================
# Setup dan Training Script untuk Tiny LLM Indo
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "🚀 SETUP & TRAINING TINY LLM INDO"
echo "============================================================"
echo ""

# ============================================================
# 1. CREATE VIRTUAL ENVIRONMENT
# ============================================================

echo "📦 Step 1: Creating virtual environment..."
python3 -m venv venv

echo "✓ Virtual environment created"
echo ""

# ============================================================
# 2. ACTIVATE VIRTUAL ENVIRONMENT
# ============================================================

echo "🔌 Step 2: Activating virtual environment..."
source venv/bin/activate

echo "✓ Virtual environment activated"
echo ""

# ============================================================
# 3. UPGRADE PIP
# ============================================================

echo "⬆️  Step 3: Upgrading pip..."
pip install --upgrade pip

echo "✓ Pip upgraded"
echo ""

# ============================================================
# 4. INSTALL DEPENDENCIES
# ============================================================

echo "📥 Step 4: Installing dependencies..."
pip install torch torchvision torchaudio
pip install transformers
pip install datasets
pip install tqdm
pip install accelerate

echo "✓ Dependencies installed"
echo ""

# ============================================================
# 5. VERIFY INSTALLATION
# ============================================================

echo "🔍 Step 5: Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import datasets; print(f'Datasets: {datasets.__version__}')"

if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    echo "✓ CUDA available"
    python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
else
    echo "⚠️  CUDA not available, will use CPU"
fi

echo ""

# ============================================================
# 6. PREPARE Q&A DATASET
# ============================================================

echo "============================================================"
echo "📚 Step 6: Preparing Q&A Dataset from dataset_topics/"
echo "============================================================"
echo ""

python3 prepare_qa_from_topics.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Dataset prepared successfully"
else
    echo ""
    echo "❌ Dataset preparation failed"
    exit 1
fi

echo ""

# ============================================================
# 7. START FINE-TUNING
# ============================================================

echo "============================================================"
echo "🏋️  Step 7: Starting Fine-tuning"
echo "============================================================"
echo ""

python3 finetune_qa.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ TRAINING COMPLETE!"
    echo "============================================================"
    echo ""
    echo "Model saved to: ./tiny-llm-indo-qa/"
    echo ""
    echo "Next steps:"
    echo "  1. Test model: python3 test_model.py"
    echo "  2. Deactivate venv: deactivate"
else
    echo ""
    echo "❌ Fine-tuning failed"
    exit 1
fi
