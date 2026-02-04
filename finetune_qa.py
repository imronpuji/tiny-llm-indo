"""
Fine-tune Script untuk Q&A
==========================
Jalankan SETELAH training general selesai!

Penggunaan:
    python finetune_qa.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    "./dataset/train_qa.json",         # Data Hukum, Sejarah, Lokal (Manual)
    "./dataset/train_general_qa.json"  # Data General dari Hugging Face
]

EVAL_DATA_FILES = [
    "./dataset/eval_qa.json",
    "./dataset/eval_general_qa.json"
]

# Training config untuk fine-tuning model 150M yang LEBIH CEPAT
FINETUNE_CONFIG = {
    "output_dir": "./tiny-llm-indo-qa-checkpoints",
    "num_train_epochs": 3,                     # Cukup 3 epoch jika dataset sudah besar (10rb+ samples)
    "per_device_train_batch_size": 16,         # Naikkan lagi batch size agar lebih cepat
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 2,          # Total batch tetap 32 (16 * 2)
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "eval_strategy": "no",
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "fp16": torch.cuda.is_available(),
    "dataloader_num_workers": 4,
    "seed": 42,
    "report_to": "none",
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


def tokenize_function(examples, tokenizer, max_length=256):
    """Tokenize texts - 256 sudah cukup untuk instruksi pendek"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_special_tokens_mask=True,
    )


def main():
    print("=" * 60)
    print("🎯 FINE-TUNING UNTUK Q&A (LAW + GENERAL)")
    print("=" * 60)
    
    # Check base model exists
    if not os.path.exists(BASE_MODEL_PATH):
        # Jika masa-ai-qa-fresh belum ada, coba pakai base model mentah
        print(f"⚠ Warning: {BASE_MODEL_PATH} tidak ditemukan.")
        print("   Mencoba mencari base model alternatif...")
        alternate = "./tiny-llm-indo-final"
        if os.path.exists(alternate):
            global BASE_MODEL_PATH
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
    print("\n🔤 Tokenizing (Max Length: 256)...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train"
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing eval"
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
