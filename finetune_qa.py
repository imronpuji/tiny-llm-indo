"""
Fine-tune Script untuk Q&A
==========================
Jalankan SETELAH training general selesai!

Penggunaan:
    python finetune_qa.py
"""

import os
# JANGAN set CUDA_VISIBLE_DEVICES — gunakan semua 4 GPU

import json
import torch
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)

# ============================================================
# KONFIGURASI
# ============================================================

# Belajar dari V2 yang sudah lumayan "waras"
BASE_MODEL_PATH = "./masa-ai-qa-v2"

# Output ke V3 sebagai versi Final/Pintar
OUTPUT_PATH = "./masa-ai-qa-v3"

# Datasets - Sekarang kita bisa ambil dari banyak sumber!
TRAIN_DATA_FILES = [
    "./dataset/train_qa.json",            # Data Sejarah, Lokal (Manual)
    "./dataset/train_general_qa.json",    # Data General dari Hugging Face
    "./dataset/train_alpaca_qa.json"      # Alpaca Indonesia (cahya) - Instruction Following
]

EVAL_DATA_FILES = [
    "./dataset/eval_qa.json",
    "./dataset/eval_general_qa.json",
    "./dataset/eval_alpaca_qa.json"
]

# Training config untuk fine-tuning model 200M - OPTIMIZED FOR 4x RTX PRO 6000 S (382GB VRAM)
FINETUNE_CONFIG = {
    "output_dir": "./tiny-llm-indo-qa-checkpoints",
    "num_train_epochs": 3,                     # 3 epoch cukup, >3 overfitting
    "per_device_train_batch_size": 64,          # 95GB per GPU — batch 64 aman
    "per_device_eval_batch_size": 64,
    "gradient_accumulation_steps": 2,           # Effective batch = 64*4*2 = 512
    "learning_rate": 2e-5,                     # LR lebih kecil untuk SFT agar tidak rusak pretraining
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,                      # 3% warmup — batch besar
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "bf16": True,                              # Native bf16
    "bf16_full_eval": True,
    "gradient_checkpointing": False,           # MATIKAN — VRAM >>>
    "dataloader_num_workers": 16,              # Per GPU, total 64 workers
    "dataloader_pin_memory": True,
    "dataloader_prefetch_factor": 4,
    "seed": 42,
    "report_to": "none",
    "max_grad_norm": 1.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,                        # Lebih stabil
    "torch_compile": True,                     # torch.compile() speedup
    "optim": "adamw_torch_fused",              # Fused AdamW
    "ddp_backend": "nccl",                     # NCCL untuk multi-GPU
}


# ============================================================
# FUNCTIONS
# ============================================================

def load_combined_datasets(file_paths):
    """Load dan gabungkan beberapa dataset JSON"""
    combined_data = []
    for path in file_paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                combined_data.extend(data)
                print(f"   ✓ Loaded: {path} ({len(data)} samples)")
        else:
            print(f"   ⚠ Warning: File not found {path}")
            
    if not combined_data:
        raise ValueError("No datasets found! Harap jalankan script add_*.py dulu.")
        
    return Dataset.from_list(combined_data)


def tokenize_function(examples, tokenizer, max_length=1024):
    """Tokenize texts dengan EOS token agar model belajar kapan berhenti"""
    texts = [t + tokenizer.eos_token for t in examples["text"]]
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # Dynamic padding via DataCollator
        return_special_tokens_mask=True,
    )


def main():
    global BASE_MODEL_PATH
    print("=" * 60)
    print("🎯 FINE-TUNING UNTUK Q&A (GENERAL)")
    print("=" * 60)
    
    # Check base model exists
    if not os.path.exists(BASE_MODEL_PATH):
        # Jika model utama tidak ada, coba pakai base model mentah
        print(f"⚠ Warning: {BASE_MODEL_PATH} tidak ditemukan.")
        print("   Mencoba mencari base model alternatif...")
        alternate = "./tiny-llm-indo-final"
        if os.path.exists(alternate):
            BASE_MODEL_PATH = alternate
            print(f"   ✓ Menggunakan: {BASE_MODEL_PATH}")
        else:
            print("❌ Tidak ada model untuk di-finetune!")
            return
            
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 Device: {device}")
    
    # Load base model
    print(f"\n📦 Loading model dari: {BASE_MODEL_PATH}")
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    print(f"\n📂 Loading & Merging datasets...")
    train_dataset = load_combined_datasets(TRAIN_DATA_FILES)
    eval_dataset = load_combined_datasets(EVAL_DATA_FILES)
    
    print(f"\n📊 Total Dataset:")
    print(f"   - Train: {len(train_dataset)} samples")
    print(f"   - Eval: {len(eval_dataset)} samples")
    
    # Tokenize
    print("\n🔤 Tokenizing (Max Length: 2048 - Extended Context)...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=2048),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train",
        num_proc=32,  # Parallel tokenization — 64 cores
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=2048),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing eval",
        num_proc=32,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(**FINETUNE_CONFIG)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Fine-tune!
    print("\n" + "=" * 60)
    print("🏋️ STARTING FINE-TUNING")
    print("=" * 60)
    
    trainer.train()
    
    # Save final model
    print(f"\n💾 Saving fine-tuned model ke: {OUTPUT_PATH}")
    trainer.save_model(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    
    print(f"✓ Model saved!")
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"   {key}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ FINE-TUNING COMPLETE!")
    print("=" * 60)
    print(f"\nUntuk test model Q&A:")
    print(f"   python test_model.py {OUTPUT_PATH} --qa")


if __name__ == "__main__":
    main()
