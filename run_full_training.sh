#!/bin/bash

# ============================================================
# FULL TRAINING PIPELINE - NO TRUNCATION
# ============================================================
# Script ini menjalankan semua training dari awal sampai akhir
# dengan optimasi full dataset (tidak dipotong)
#
# Estimasi waktu total: 3-6 jam
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "🚀 FULL TRAINING PIPELINE - OPTIMIZED FOR 150M MODEL"
echo "============================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============================================================
# STEP 1: PREPARE DATASETS
# ============================================================
echo -e "${YELLOW}[1/4] Preparing QA Datasets...${NC}"
echo "-----------------------------------------------------------"

if [ ! -f "./dataset/train_qa.json" ]; then
    echo "  → Generating base QA dataset..."
    python add_qa_data.py
    echo -e "${GREEN}  ✓ Base QA dataset created${NC}"
else
    echo -e "${GREEN}  ✓ Base QA dataset already exists${NC}"
fi

if [ ! -f "./dataset/train_general_qa.json" ]; then
    echo "  → Downloading general QA dataset..."
    python add_general_qa.py
    echo -e "${GREEN}  ✓ General QA dataset created${NC}"
else
    echo -e "${GREEN}  ✓ General QA dataset already exists${NC}"
fi

if [ ! -f "./dataset/train_alpaca_qa.json" ]; then
    echo "  → Downloading Alpaca Indonesia dataset..."
    python add_alpaca_qa.py
    echo -e "${GREEN}  ✓ Alpaca QA dataset created${NC}"
else
    echo -e "${GREEN}  ✓ Alpaca QA dataset already exists${NC}"
fi

echo ""
echo "-----------------------------------------------------------"
echo -e "${GREEN}✅ All QA datasets prepared!${NC}"
echo "-----------------------------------------------------------"
echo ""
sleep 2

# ============================================================
# STEP 2: PREPARE PREFERENCE PAIRS
# ============================================================
echo -e "${YELLOW}[2/4] Preparing Preference Pairs for DPO...${NC}"
echo "-----------------------------------------------------------"

if [ ! -f "./dataset/train_preference.json" ]; then
    echo "  → Generating preference pairs (good vs bad answers)..."
    python add_preference_data.py
    echo -e "${GREEN}  ✓ Preference dataset created${NC}"
else
    echo -e "${GREEN}  ✓ Preference dataset already exists${NC}"
fi

echo ""
echo "-----------------------------------------------------------"
echo -e "${GREEN}✅ Preference pairs ready!${NC}"
echo "-----------------------------------------------------------"
echo ""
sleep 2

# ============================================================
# STEP 3: SUPERVISED FINE-TUNING (SFT)
# ============================================================
echo -e "${YELLOW}[3/4] Supervised Fine-Tuning (SFT)...${NC}"
echo "-----------------------------------------------------------"
echo "  Config:"
echo "    - Max Length: 512 tokens (full context)"
echo "    - Epochs: 5"
echo "    - Learning Rate: 3e-5"
echo "    - Batch Size: 8 (effective 32 with grad accum)"
echo "    - Gradient Checkpointing: Enabled"
echo ""
echo "  Estimasi waktu: 2-4 jam"
echo "-----------------------------------------------------------"
echo ""

# Check if base model exists
if [ ! -d "./masa-ai-qa-v2" ] && [ ! -d "./tiny-llm-indo-final" ]; then
    echo -e "${RED}❌ Base model tidak ditemukan!${NC}"
    echo "   Opsi:"
    echo "   1. Training from scratch: python train_tiny_llm.py"
    echo "   2. Download pre-trained dari HuggingFace"
    exit 1
fi

# Run fine-tuning
echo "  → Starting SFT..."
python finetune_qa.py

if [ $? -eq 0 ]; then
    echo ""
    echo "-----------------------------------------------------------"
    echo -e "${GREEN}✅ SFT Complete!${NC}"
    echo "-----------------------------------------------------------"
    echo ""
else
    echo -e "${RED}❌ SFT Failed! Check error above.${NC}"
    exit 1
fi

sleep 2

# ============================================================
# STEP 4: DPO ALIGNMENT
# ============================================================
echo -e "${YELLOW}[4/4] DPO Training (Preference Alignment)...${NC}"
echo "-----------------------------------------------------------"
echo "  Config:"
echo "    - Beta: 0.1"
echo "    - Epochs: 3"
echo "    - Learning Rate: 5e-7 (very small)"
echo ""
echo "  DPO akan mengajarkan model:"
echo "    ✅ Memilih jawaban yang benar"
echo "    ❌ Menghindari jawaban salah/halusinasi"
echo "    📊 Meningkatkan relevansi dan akurasi"
echo ""
echo "  Estimasi waktu: 30-60 menit"
echo "-----------------------------------------------------------"
echo ""

# Check if TRL is installed
python -c "import trl" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  TRL not installed. Installing...${NC}"
    pip install trl
fi

# Run DPO training
echo "  → Starting DPO alignment..."
python train_dpo.py

if [ $? -eq 0 ]; then
    echo ""
    echo "-----------------------------------------------------------"
    echo -e "${GREEN}✅ DPO Alignment Complete!${NC}"
    echo "-----------------------------------------------------------"
    echo ""
else
    echo -e "${RED}❌ DPO Failed! Check error above.${NC}"
    exit 1
fi

# ============================================================
# FINAL SUMMARY
# ============================================================
echo ""
echo "============================================================"
echo -e "${GREEN}🎉 FULL TRAINING PIPELINE COMPLETE!${NC}"
echo "============================================================"
echo ""
echo "📊 Training Summary:"
echo "  ✅ Dataset prepared (QA + Preference pairs)"
echo "  ✅ Supervised Fine-Tuning (SFT) - Full 512 tokens"
echo "  ✅ DPO Alignment - Preference optimization"
echo ""
echo "📁 Model locations:"
echo "  - SFT Model: ./masa-ai-qa-v3"
echo "  - DPO Model: ./masa-ai-dpo-aligned"
echo ""
echo "🧪 Testing:"
echo "  python test_model.py ./masa-ai-dpo-aligned --qa"
echo ""
echo "📤 Upload to HuggingFace:"
echo "  python upload_to_hf.py"
echo ""
echo "============================================================"
echo ""

# Optional: Auto-test
read -p "Test model sekarang? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🧪 Testing model..."
    python test_model.py ./masa-ai-dpo-aligned --qa
fi
