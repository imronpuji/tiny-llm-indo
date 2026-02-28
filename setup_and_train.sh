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
# 0. CHECK AND INSTALL PYTHON3-VENV (Linux/Docker only)
# ============================================================

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🔍 Step 0: Checking python3-venv..."
    
    # Detect Python version
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo "   Detected Python ${PYTHON_VERSION}"
    
    # Check if venv module is available
    if ! python3 -m venv --help &> /dev/null; then
        echo "   ⚠️  python3-venv not found, attempting to install..."
        
        # Try to install based on Python version
        if command -v apt &> /dev/null; then
            apt update
            apt install -y python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev || \
            apt install -y python3-venv python3-dev
        elif command -v yum &> /dev/null; then
            yum install -y python3-devel
        fi
        
        echo "   ✓ python3-venv installed"
    else
        echo "   ✓ python3-venv already available"
    fi
    echo ""
fi

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

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found, installing manually..."
    pip install torch torchvision torchaudio
    pip install transformers
    pip install datasets
    pip install tqdm
    pip install accelerate
fi

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
