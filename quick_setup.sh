#!/bin/bash

# ============================================================
# Quick Setup untuk Training Q&A
# ============================================================

set -e

echo "🚀 Quick Setup & Training"
echo ""

# Check dataset_topics
echo "🔍 Checking dataset_topics..."
if [ ! -d "dataset_topics" ]; then
    echo "❌ ERROR: Folder dataset_topics/ tidak ditemukan!"
    echo ""
    echo "Clone repository dulu:"
    echo "   git clone https://github.com/imronpuji/tiny-llm-indo.git"
    echo "   cd tiny-llm-indo"
    echo ""
    exit 1
fi

JSON_COUNT=$(ls -1 dataset_topics/*.json 2>/dev/null | wc -l)
if [ "$JSON_COUNT" -lt 40 ]; then
    echo "⚠️  Found $JSON_COUNT JSON files (expected ~42)"
    echo "Dataset mungkin tidak lengkap"
    exit 1
else
    echo "✓ Found $JSON_COUNT topic files"
fi
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "📋 Detected: Linux"
    
    # Check if running as root or with sudo
    if [ "$EUID" -ne 0 ]; then 
        echo "⚠️  Untuk install system packages, run dengan sudo:"
        echo "   sudo $0"
        echo ""
        echo "Atau install manual dulu:"
        echo "   sudo apt update"
        echo "   sudo apt install -y python3-venv python3-dev"
        echo ""
        exit 1
    fi
    
    # Install python3-venv
    echo "📦 Installing python3-venv..."
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo "   Python version: ${PYTHON_VERSION}"
    
    apt update
    apt install -y python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev 2>/dev/null || \
    apt install -y python3-venv python3-dev
    
    echo "✓ System packages installed"
    echo ""
fi

# Create venv
echo "🔧 Creating virtual environment..."
python3 -m venv venv
echo "✓ venv created"
echo ""

# Activate
echo "🔌 Activating venv..."
source venv/bin/activate
echo "✓ venv activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip -q
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Verify
echo "🔍 Verifying installation..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"
python -c "import datasets; print(f'✓ Datasets {datasets.__version__}')"
echo ""

# Prepare dataset
echo "📚 Preparing Q&A dataset..."
python prepare_qa_from_topics.py
echo ""

# Start training
echo "🏋️  Starting fine-tuning..."
echo "This will take 20-30 minutes with GPU, 2-3 hours with CPU"
echo ""
read -p "Continue with training? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python finetune_qa.py
    echo ""
    echo "✅ Training complete!"
else
    echo "Training cancelled. Run manually:"
    echo "  source venv/bin/activate"
    echo "  python finetune_qa.py"
fi
