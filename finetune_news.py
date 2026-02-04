"""
Fine-tuning dengan Dataset Berita Indonesia
============================================
Causal Language Modeling (CLM) untuk bahasa natural

Penggunaan:
    python finetune_news.py
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI
# ============================================================

# Model - melanjutkan dari v3 ke v4 (berita)
BASE_MODEL_PATH = "./masa-ai-qa-v3"
OUTPUT_PATH = "./masa-ai-qa-v4"

# Dataset
TRAIN_DATA_FILE = "./dataset/train_news.json"
EVAL_DATA_FILE = "./dataset/eval_news.json"

# Training config
FINETUNE_CONFIG = {
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,  # Effective batch = 16
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "logging_steps": 100,
    "save_steps": 500,
    "eval_steps": 500,
    "save_total_limit": 3,
    "fp16": torch.cuda.is_available(),
    "dataloader_num_workers": 0,
    "max_length": 512,  # Panjang konteks untuk artikel
}


def load_dataset_from_json(filepath: str) -> Dataset:
    """Load dataset dari file JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize teks untuk CLM"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_special_tokens_mask=True,
    )


def main():
    print("=" * 60)
    print("📰 FINE-TUNING DENGAN DATASET BERITA")
    print("=" * 60)
    
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n🖥️  Device: {device}")
    
    # Load tokenizer dan model
    print(f"\n📦 Loading model from: {BASE_MODEL_PATH}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH)
    except:
        print("   Model tidak ditemukan, menggunakan GPT-2 base...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Set pad token jika tidak ada
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    
    # Load datasets
    print(f"\n📂 Loading datasets...")
    
    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"   ❌ File tidak ditemukan: {TRAIN_DATA_FILE}")
        print("   Jalankan dulu: python add_news_data.py")
        return
    
    train_dataset = load_dataset_from_json(TRAIN_DATA_FILE)
    eval_dataset = load_dataset_from_json(EVAL_DATA_FILE) if os.path.exists(EVAL_DATA_FILE) else None
    
    print(f"   Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"   Eval samples: {len(eval_dataset)}")
    
    # Tokenize
    print(f"\n🔤 Tokenizing...")
    max_length = FINETUNE_CONFIG["max_length"]
    
    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train"
    )
    
    eval_tokenized = None
    if eval_dataset:
        eval_tokenized = eval_dataset.map(
            lambda x: tokenize_function(x, tokenizer, max_length),
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval"
        )
    
    # Data collator untuk CLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, bukan Masked LM
    )
    
    # Training arguments
    print(f"\n⚙️  Setting up training...")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_PATH,
        overwrite_output_dir=True,
        
        # Training
        num_train_epochs=FINETUNE_CONFIG["num_train_epochs"],
        per_device_train_batch_size=FINETUNE_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=FINETUNE_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=FINETUNE_CONFIG["gradient_accumulation_steps"],
        
        # Optimizer
        learning_rate=FINETUNE_CONFIG["learning_rate"],
        weight_decay=FINETUNE_CONFIG["weight_decay"],
        warmup_ratio=FINETUNE_CONFIG["warmup_ratio"],
        lr_scheduler_type="cosine",
        
        # Logging & Saving
        logging_dir=f"{OUTPUT_PATH}/logs",
        logging_steps=FINETUNE_CONFIG["logging_steps"],
        save_steps=FINETUNE_CONFIG["save_steps"],
        save_total_limit=FINETUNE_CONFIG["save_total_limit"],
        
        # Evaluation
        eval_strategy="steps" if eval_tokenized else "no",
        eval_steps=FINETUNE_CONFIG["eval_steps"] if eval_tokenized else None,
        
        # Performance
        fp16=FINETUNE_CONFIG["fp16"],
        dataloader_num_workers=FINETUNE_CONFIG["dataloader_num_workers"],
        
        # Other
        load_best_model_at_end=True if eval_tokenized else False,
        metric_for_best_model="eval_loss" if eval_tokenized else None,
        greater_is_better=False if eval_tokenized else None,
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train!
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {FINETUNE_CONFIG['num_train_epochs']}")
    print(f"   Batch size: {FINETUNE_CONFIG['per_device_train_batch_size']} x {FINETUNE_CONFIG['gradient_accumulation_steps']} = {FINETUNE_CONFIG['per_device_train_batch_size'] * FINETUNE_CONFIG['gradient_accumulation_steps']}")
    print(f"   Learning rate: {FINETUNE_CONFIG['learning_rate']}")
    print(f"   Max length: {max_length}")
    print("-" * 60)
    
    trainer.train()
    
    # Save final model
    print(f"\n💾 Saving model to: {OUTPUT_PATH}")
    trainer.save_model(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING SELESAI!")
    print("=" * 60)
    print(f"\nModel tersimpan di: {OUTPUT_PATH}")
    print("\nUntuk test model, jalankan:")
    print("  python test_news_model.py")


if __name__ == "__main__":
    main()
