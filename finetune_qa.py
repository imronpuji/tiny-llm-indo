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
    "./dataset/train_qa.json",            # Data Hukum, Sejarah, Lokal (Manual)
    "./dataset/train_general_qa.json",    # Data General dari Hugging Face
    "./dataset/train_regulation_qa.json", # Data Regulasi Indonesia (Azzindani)
    "./dataset/train_alpaca_qa.json"      # Alpaca Indonesia (cahya) - Instruction Following
]

EVAL_DATA_FILES = [
    "./dataset/eval_qa.json",
    "./dataset/eval_general_qa.json",
    "./dataset/eval_regulation_qa.json",
    "./dataset/eval_alpaca_qa.json"
]

# Training config untuk fine-tuning model 150M - FULL DATASET OPTIMIZATION
FINETUNE_CONFIG = {
    "output_dir": "./tiny-llm-indo-qa-checkpoints",
    "num_train_epochs": 5,                     # 5 epoch untuk memorisasi lebih baik
    "per_device_train_batch_size": 8,          # Turunkan batch agar muat context panjang
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,          # Total batch = 32 (8 * 4)
    "learning_rate": 3e-5,                     # Naikkan learning rate untuk konvergensi cepat
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 300,                         # Lebih sering evaluasi
    "save_strategy": "steps",
    "save_steps": 300,
    "save_total_limit": 2,                     # Simpan 2 checkpoint terbaik
    "load_best_model_at_end": True,           # Load model terbaik di akhir
    "metric_for_best_model": "eval_loss",
    "fp16": torch.cuda.is_available(),
    "gradient_checkpointing": True,            # Aktifkan untuk memory efficiency
    "dataloader_num_workers": 4,
    "seed": 42,
    "report_to": "none",
    "max_grad_norm": 1.0,                      # Gradient clipping untuk stabilitas
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


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize texts - 512 token untuk QA yang lebih lengkap tanpa dipotong"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_special_tokens_mask=True,
    )


def main():
    global BASE_MODEL_PATH
    print("=" * 60)
    print("🎯 FINE-TUNING UNTUK Q&A (LAW + GENERAL)")
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
    print("\n🔤 Tokenizing (Max Length: 512 - Full Context)...")
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
