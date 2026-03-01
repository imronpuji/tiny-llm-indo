#!/bin/bash
# ============================================================
# Setup & Train Qwen2.5-1.5B untuk Bahasa Indonesia
# ============================================================
# 
# Jalankan di Docker container dengan GPU:
#   docker run --gpus all -v $(pwd):/workspace -it pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel bash
#   cd /workspace/tiny-llm-indo
#   bash setup_qwen.sh
#
# Minimum requirements:
#   - GPU with 8GB+ VRAM (recommended 16GB+)
#   - 10GB disk space
#   - Internet connection (download model ~3GB)
# ============================================================

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🚀 SETUP QWEN2.5-1.5B FINE-TUNING                       ║"
echo "║  Upgrade dari GPT2-small (124M) → Qwen2.5 (1.5B)         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Install dependencies ───────────────────────────
echo "📦 [1/5] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet torch>=2.2.0 transformers>=4.38.0 datasets accelerate
pip install --quiet peft bitsandbytes trl
pip install --quiet tqdm huggingface-hub

# Verify
python3 -c "
import torch, transformers, peft, trl
print(f'  ✓ PyTorch {torch.__version__}')
print(f'  ✓ Transformers {transformers.__version__}')
print(f'  ✓ PEFT {peft.__version__}')
print(f'  ✓ TRL {trl.__version__}')
print(f'  ✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'  ✓ VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
"
echo ""

# ── Step 2: Prepare dataset ────────────────────────────────
echo "📂 [2/5] Preparing dataset..."
python3 prepare_qa_from_topics.py
echo ""

# ── Step 3: Verify dataset ─────────────────────────────────
echo "✅ [3/5] Verifying dataset..."
python3 -c "
import json
train = json.load(open('dataset/train_qa.json'))
eval_d = json.load(open('dataset/eval_qa.json'))
print(f'  Train: {len(train)} samples')
print(f'  Eval:  {len(eval_d)} samples')
print(f'  Total: {len(train) + len(eval_d)} samples')
"
echo ""

# ── Step 4: Fine-tune ──────────────────────────────────────
echo "🏋️ [4/5] Starting fine-tuning..."
echo "  Model: Qwen/Qwen2.5-1.5B"
echo "  Method: QLoRA (4-bit quantization + LoRA)"
echo "  This will take 30-60 minutes on a single GPU"
echo ""
python3 finetune_qwen.py
echo ""

# ── Step 5: Test ────────────────────────────────────────────
echo "🧪 [5/5] Testing model..."
if [ -d "./masa-ai-qwen-merged" ]; then
    python3 test_model_qwen.py ./masa-ai-qwen-merged --qa-batch
elif [ -d "./masa-ai-qwen" ]; then
    python3 test_model_qwen.py ./masa-ai-qwen --qa-batch
else
    echo "⚠️  Model not found, skipping test"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ SETUP COMPLETE!                                        ║"
echo "║                                                            ║"
echo "║  Test interaktif:                                          ║"
echo "║    python3 test_model_qwen.py ./masa-ai-qwen-merged       ║"
echo "║                                                            ║"
echo "║  Upload ke HuggingFace:                                    ║"
echo "║    huggingface-cli login                                   ║"
echo "║    python3 upload_model.py                                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
