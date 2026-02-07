#!/bin/bash

# ============================================================
# FULL TRAINING PIPELINE — FROM SCRATCH TO DEPLOYED
# ============================================================
# Pipeline lengkap: Dataset → Pre-training → SFT → DPO → Test
#
# Estimasi waktu total:
#   - Pre-training: 8-12 jam (150M params)
#   - SFT: 2-4 jam
#   - DPO: 30-60 menit
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "  FULL TRAINING PIPELINE — MASA AI 150M"
echo "============================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Track start time
START_TIME=$(date +%s)

# ============================================================
# STEP 0: CHECK DEPENDENCIES
# ============================================================
echo -e "${BLUE}[0/6] Checking dependencies...${NC}"
echo "-----------------------------------------------------------"

python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'  Datasets: {datasets.__version__}')"
python -c "import trl; print(f'  TRL: {trl.__version__}')" 2>/dev/null || {
    echo -e "${YELLOW}  TRL not installed. Installing...${NC}"
    pip install trl
}

if python -c "import torch; assert torch.cuda.is_available()"; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')")
    echo -e "${GREEN}  GPU: ${GPU_NAME} (${GPU_MEM})${NC}"
else
    echo -e "${YELLOW}  WARNING: No GPU detected. Training will be VERY slow.${NC}"
fi

echo ""
sleep 1

# ============================================================
# STEP 1: PREPARE ALL DATASETS
# ============================================================
echo -e "${YELLOW}[1/6] Preparing all datasets...${NC}"
echo "-----------------------------------------------------------"

# QA Dataset (manual — most important for coherence)
if [ ! -f "./dataset/train_qa.json" ]; then
    echo "  -> Generating base QA dataset (manual)..."
    python add_qa_data.py
    echo -e "${GREEN}  Done: Base QA dataset${NC}"
else
    echo -e "${GREEN}  Skip: Base QA dataset exists${NC}"
fi

# General QA from HuggingFace
if [ ! -f "./dataset/train_general_qa.json" ]; then
    echo "  -> Downloading General QA dataset..."
    python add_general_qa.py
    echo -e "${GREEN}  Done: General QA dataset${NC}"
else
    echo -e "${GREEN}  Skip: General QA dataset exists${NC}"
fi

# Alpaca Indonesia
if [ ! -f "./dataset/train_alpaca_qa.json" ]; then
    echo "  -> Downloading Alpaca Indonesia..."
    python add_alpaca_qa.py
    echo -e "${GREEN}  Done: Alpaca QA dataset${NC}"
else
    echo -e "${GREEN}  Skip: Alpaca QA dataset exists${NC}"
fi

# Regulation QA
if [ ! -f "./dataset/train_regulation_qa.json" ]; then
    echo "  -> Downloading Regulation QA..."
    python add_regulation_qa.py
    echo -e "${GREEN}  Done: Regulation QA dataset${NC}"
else
    echo -e "${GREEN}  Skip: Regulation QA dataset exists${NC}"
fi

# Preference pairs for DPO
if [ ! -f "./dataset/train_preference.json" ]; then
    echo "  -> Generating preference pairs for DPO..."
    python add_preference_data.py
    echo -e "${GREEN}  Done: Preference dataset${NC}"
else
    echo -e "${GREEN}  Skip: Preference dataset exists${NC}"
fi

echo ""
echo -e "${GREEN}All datasets ready!${NC}"
echo ""
sleep 1

# ============================================================
# STEP 2: PREPARE LARGE PRE-TRAINING DATASET
# ============================================================
echo -e "${YELLOW}[2/6] Preparing large pre-training dataset (Wiki + CC100)...${NC}"
echo "-----------------------------------------------------------"

if [ ! -f "./dataset/train_large.json" ]; then
    echo "  -> Downloading Wikipedia ID + CC100..."
    echo "  -> This may take 30-60 minutes..."
    python prepare_large_dataset.py
    echo -e "${GREEN}  Done: Large dataset${NC}"
else
    echo -e "${GREEN}  Skip: Large dataset exists${NC}"
fi

echo ""
sleep 1

# ============================================================
# STEP 3: PRE-TRAINING FROM SCRATCH
# ============================================================
echo -e "${YELLOW}[3/6] Pre-training 150M model from scratch...${NC}"
echo "-----------------------------------------------------------"
echo "  Config:"
echo "    - Parameters: ~150M"
echo "    - Epochs: 3"
echo "    - Learning Rate: 6e-4 (Chinchilla-optimal)"
echo "    - Effective Batch: 64"
echo "    - Warmup: 10%"
echo "    - Weight Init: GPT-2 scaled initialization"
echo ""
echo "  Estimasi: 8-12 jam (GPU dependent)"
echo "-----------------------------------------------------------"
echo ""

if [ ! -d "./tiny-llm-indo-final" ]; then
    echo "  -> Starting pre-training..."
    python train_tiny_llm.py
    echo -e "${GREEN}  Done: Pre-training complete${NC}"
else
    echo -e "${GREEN}  Skip: Pre-trained model found${NC}"
fi

echo ""
sleep 1

# ============================================================
# STEP 4: SUPERVISED FINE-TUNING (SFT)
# ============================================================
echo -e "${YELLOW}[4/6] Supervised Fine-Tuning (SFT)...${NC}"
echo "-----------------------------------------------------------"
echo "  Config:"
echo "    - Epochs: 3"
echo "    - Learning Rate: 2e-5 (gentle SFT)"
echo "    - Effective Batch: 64"
echo ""
echo "  Estimasi: 2-4 jam"
echo "-----------------------------------------------------------"
echo ""

# Check base model
if [ ! -d "./masa-ai-qa-v2" ] && [ ! -d "./tiny-llm-indo-final" ]; then
    echo -e "${RED}  No base model found! Run pre-training first.${NC}"
    exit 1
fi

echo "  -> Starting SFT..."
python finetune_qa.py || { echo -e "${RED}  SFT failed!${NC}"; exit 1; }
echo -e "${GREEN}  Done: SFT complete${NC}"
echo ""
sleep 1

# ============================================================
# STEP 5: DPO ALIGNMENT
# ============================================================
echo -e "${YELLOW}[5/6] DPO Alignment (Coherence & Anti-Hallucination)...${NC}"
echo "-----------------------------------------------------------"
echo "  Config:"
echo "    - Beta: 0.2 (conservative)"
echo "    - Epochs: 2"
echo "    - Learning Rate: 1e-6"
echo ""
echo "  Estimasi: 30-60 menit"
echo "-----------------------------------------------------------"
echo ""

echo "  -> Starting DPO alignment..."
python train_dpo.py || { echo -e "${RED}  DPO failed!${NC}"; exit 1; }
echo -e "${GREEN}  Done: DPO complete${NC}"
echo ""
sleep 1

# ============================================================
# STEP 6: EVALUATION
# ============================================================
echo -e "${YELLOW}[6/6] Evaluating model...${NC}"
echo "-----------------------------------------------------------"

echo "  -> Running QA evaluation..."
python test_model.py ./masa-ai-dpo-aligned --qa-batch

echo ""

# ============================================================
# FINAL SUMMARY
# ============================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "============================================================"
echo -e "${GREEN}  TRAINING PIPELINE COMPLETE!${NC}"
echo "============================================================"
echo ""
echo "  Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "  Model locations:"
echo "    Pre-trained:  ./tiny-llm-indo-final"
echo "    SFT:          ./masa-ai-qa-v3"
echo "    DPO (final):  ./masa-ai-dpo-aligned"
echo ""
echo "  Test model:"
echo "    python test_model.py ./masa-ai-dpo-aligned --qa"
echo ""
echo "  Upload to HuggingFace:"
echo "    python upload_to_hf.py"
echo ""
echo "============================================================"
