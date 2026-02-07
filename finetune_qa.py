"""
Fine-tune Script untuk Q&A
==========================
Jalankan SETELAH training general selesai!

Penggunaan:
    python finetune_qa.py
"""

import os
# ============================================================
# SINGLE GPU MODE
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only GPU 0
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# Training config untuk fine-tuning model 200M - SINGLE A100 GPU
FINETUNE_CONFIG = {
    "output_dir": "./tiny-llm-indo-qa-checkpoints",
    "num_train_epochs": 3,                     # 3 epoch cukup, >3 overfitting
    "per_device_train_batch_size": 32,          # Single A100 40GB
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 16,          # Effective batch = 32*16 = 512
    "learning_rate": 2e-5,                     # LR lebih kecil untuk SFT
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "bf16": True,                              # A100 native bf16
    "bf16_full_eval": True,
    "gradient_checkpointing": False,
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
    "seed": 42,
    "report_to": "none",
    "max_grad_norm": 1.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "torch_compile": False,
    "optim": "adamw_torch",
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
    result = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # Dynamic padding via DataCollator
        return_special_tokens_mask=True,
    )
    return result


def filter_valid_samples(examples, vocab_size):
    """Filter out samples with token IDs >= vocab_size"""
    valid = []
    for ids in examples["input_ids"]:
        max_id = max(ids) if ids else 0
        valid.append(max_id < vocab_size)
    return valid


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
    
    # Get vocab size dari model
    model_vocab_size = model.config.vocab_size
    tokenizer_vocab_size = len(tokenizer)
    print(f"   Model vocab size: {model_vocab_size}")
    print(f"   Tokenizer vocab size: {tokenizer_vocab_size}")
    
    # IMPORTANT: Resize embeddings to match tokenizer vocab size
    if model_vocab_size != tokenizer_vocab_size:
        model.resize_token_embeddings(tokenizer_vocab_size)
        print(f"   ✓ Embeddings resized from {model_vocab_size} to {tokenizer_vocab_size}")
    
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
        num_proc=8,  # Reduced for stability
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=2048),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing eval",
        num_proc=8,
    )
    
    # Filter out samples with invalid token IDs
    vocab_size = model.config.vocab_size
    print(f"\n🔍 Filtering samples with token IDs >= {vocab_size}...")
    
    train_before = len(train_dataset)
    train_dataset = train_dataset.filter(
        lambda x: filter_valid_samples(x, vocab_size),
        batched=True,
        desc="Filtering train",
    )
    print(f"   Train: {train_before} -> {len(train_dataset)} (removed {train_before - len(train_dataset)})")
    
    eval_before = len(eval_dataset)
    eval_dataset = eval_dataset.filter(
        lambda x: filter_valid_samples(x, vocab_size),
        batched=True,
        desc="Filtering eval",
    )
    print(f"   Eval: {eval_before} -> {len(eval_dataset)} (removed {eval_before - len(eval_dataset)})")
    
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
