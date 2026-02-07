"""
DPO (Direct Preference Optimization) Training
==============================================
Script ini melakukan alignment menggunakan DPO untuk meningkatkan kualitas jawaban.

DPO mengajarkan model untuk:
- Memilih jawaban yang benar vs salah
- Menghindari halusinasi
- Memberikan jawaban yang relevan

Penggunaan:
    python train_dpo.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

# Install TRL untuk DPO
try:
    from trl import DPOTrainer, DPOConfig
    TRL_AVAILABLE = True
except ImportError:
    print("❌ TRL tidak terinstall!")
    print("   Install dengan: pip install trl")
    TRL_AVAILABLE = False
    exit(1)

# ============================================================
# KONFIGURASI
# ============================================================

# Model yang sudah di-finetune dengan QA
BASE_MODEL_PATH = "./masa-ai-qa-v3"  # Atau "./tiny-llm-indo-final"

# Output model setelah DPO
OUTPUT_PATH = "./masa-ai-dpo-aligned"

# Dataset preference
TRAIN_PREFERENCE = "./dataset/train_preference.json"
EVAL_PREFERENCE = "./dataset/eval_preference.json"

# DPO Training Config — OPTIMIZED FOR ~64GB VRAM GPU
DPO_CONFIG = {
    "output_dir": "./dpo-checkpoints",
    "num_train_epochs": 2,                     # 2 epoch cukup untuk DPO (lebih = overfitting)
    "per_device_train_batch_size": 16,          # DPO butuh 2x memori (model + ref), 64GB → batch 16 aman
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 8,           # Effective batch = 128
    "learning_rate": 1e-6,                     # LR sangat kecil untuk DPO (hanya fine-adjust)
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "bf16": True,
    "bf16_full_eval": True,
    "fp16": False,
    "gradient_checkpointing": False,           # MATIKAN — 64GB VRAM cukup
    "dataloader_num_workers": 40,              # Half of 80 CPU cores
    "dataloader_pin_memory": True,
    "dataloader_prefetch_factor": 4,
    "seed": 42,
    "report_to": "none",
    "remove_unused_columns": False,
    "max_grad_norm": 0.5,                      # Gradient clip lebih ketat untuk DPO
    "torch_compile": True,                     # torch.compile() speedup
    "optim": "adamw_torch_fused",              # Fused AdamW
    
    # DPO Specific
    "beta": 0.2,                               # Beta 0.2 lebih konservatif
    "max_prompt_length": 1024,                 # Extended prompt
    "max_length": 2048,                        # Extended total length
}


# ============================================================
# FUNCTIONS
# ============================================================

def load_preference_dataset(file_path):
    """Load preference dataset"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return Dataset.from_list(data)


def main():
    print("=" * 60)
    print("🎯 DPO TRAINING - DIRECT PREFERENCE OPTIMIZATION")
    print("=" * 60)
    
    # Check base model
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"❌ Model tidak ditemukan: {BASE_MODEL_PATH}")
        print("   Jalankan finetune_qa.py terlebih dahulu!")
        return
    
    # Check datasets
    if not os.path.exists(TRAIN_PREFERENCE):
        print(f"❌ Dataset tidak ditemukan: {TRAIN_PREFERENCE}")
        print("   Jalankan add_preference_data.py terlebih dahulu!")
        return
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 Device: {device}")
    
    # Load model and tokenizer
    print(f"\n📦 Loading model: {BASE_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    print(f"\n📂 Loading preference datasets...")
    train_dataset = load_preference_dataset(TRAIN_PREFERENCE)
    eval_dataset = load_preference_dataset(EVAL_PREFERENCE)
    
    print(f"   - Train: {len(train_dataset)} pairs")
    print(f"   - Eval: {len(eval_dataset)} pairs")
    
    # DPO Training Arguments
    training_args = DPOConfig(**DPO_CONFIG)
    
    # DPO Trainer
    print("\n🏗️  Initializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Reference model akan dibuat otomatis
        args=training_args,
        processing_class=tokenizer,  # TRL 0.27+ menggunakan processing_class
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Training
    print("\n" + "=" * 60)
    print("🏋️  STARTING DPO TRAINING")
    print("=" * 60)
    print("\nDPO akan mengajarkan model untuk:")
    print("  ✅ Memilih jawaban yang benar")
    print("  ❌ Menghindari jawaban yang salah/halusinasi")
    print("  📊 Meningkatkan kualitas dan relevansi jawaban")
    print()
    
    trainer.train()
    
    # Save model
    print(f"\n💾 Saving DPO-aligned model ke: {OUTPUT_PATH}")
    trainer.save_model(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    
    print("\n" + "=" * 60)
    print("✅ DPO TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel telah di-align dengan DPO dan disimpan di:")
    print(f"   {OUTPUT_PATH}")
    print("\nUntuk test model:")
    print(f"   python test_model.py {OUTPUT_PATH} --qa")
    print("\nAtau upload ke HuggingFace:")
    print(f"   python upload_to_hf.py")


if __name__ == "__main__":
    if not TRL_AVAILABLE:
        print("❌ Install TRL terlebih dahulu: pip install trl")
    else:
        main()
