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

# Base model - bisa dari HuggingFace atau local path
# HuggingFace: "yasmeenimron/masa-ai-qa"
# Local: "./tiny-llm-indo-final"
BASE_MODEL_PATH = "yasmeenimron/masa-ai-qa"

# Output fine-tuned model
OUTPUT_PATH = "./tiny-llm-indo-qa"

# Dataset Q&A
TRAIN_DATA_PATH = "./dataset/train_qa.json"
EVAL_DATA_PATH = "./dataset/eval_qa.json"

# Training config untuk fine-tuning Q&A
FINETUNE_CONFIG = {
    "output_dir": "./tiny-llm-indo-qa-checkpoints",
    "num_train_epochs": 10,                    # 10 epoch cukup untuk Q&A
    "per_device_train_batch_size": 8,          # Batch size yang seimbang
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 2,          # Effective batch = 16
    "learning_rate": 3e-5,                     # Learning rate moderat
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,                       # Warmup 10% dari training
    "lr_scheduler_type": "cosine",
    "logging_steps": 50,
    "eval_strategy": "steps",                  # Eval per steps, bukan epoch
    "eval_steps": 200,                         # Eval setiap 200 steps
    "save_strategy": "steps",
    "save_steps": 200,                         # Save setiap 200 steps
    "save_total_limit": 3,                     # Simpan max 3 checkpoints
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "fp16": torch.cuda.is_available(),
    "dataloader_num_workers": 2,
    "seed": 42,
    "report_to": "none",
    "max_grad_norm": 1.0,                      # Gradient clipping
}


# ============================================================
# FUNCTIONS
# ============================================================

def load_dataset_from_json(path):
    """Load dataset dari JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize texts"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_special_tokens_mask=True,
    )


def main():
    print("=" * 60)
    print("🎯 FINE-TUNING UNTUK Q&A")
    print("=" * 60)
    print()
    
    # Check Q&A dataset exists
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"❌ Dataset Q&A tidak ditemukan: {TRAIN_DATA_PATH}")
        print("   Jalankan dulu:")
        print("   python prepare_qa_from_topics.py")
        return
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load base model
    print(f"\n📦 Loading base model: {BASE_MODEL_PATH}")
    is_huggingface = "/" in BASE_MODEL_PATH and not BASE_MODEL_PATH.startswith(".")
    if is_huggingface:
        print(f"   Source: HuggingFace Hub")
    else:
        print(f"   Source: Local path")
        if not os.path.exists(BASE_MODEL_PATH):
            print(f"\n❌ Model tidak ditemukan: {BASE_MODEL_PATH}")
            print("   Opsi:")
            print("   1. Gunakan model dari HuggingFace: BASE_MODEL_PATH = 'yasmeenimron/masa-ai-qa'")
            print("   2. Train model lokal: python train_tiny_llm.py")
            return
    
    try:
        model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("   Pastikan model path/name benar dan internet tersambung (untuk HuggingFace)")
        return
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {total_params/1e6:.1f}M parameters")
    
    # Load datasets
    print(f"\n📂 Loading Q&A datasets...")
    train_dataset = load_dataset_from_json(TRAIN_DATA_PATH)
    eval_dataset = load_dataset_from_json(EVAL_DATA_PATH)
    
    print(f"✓ Train: {len(train_dataset)} samples")
    print(f"✓ Eval: {len(eval_dataset)} samples")
    
    # Tokenize
    print("\n🔤 Tokenizing...")
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
    
    # Callbacks - Early stopping lebih patient untuk Q&A
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=3,         # Stop jika 3 eval berturut tidak improve
            early_stopping_threshold=0.001     # Threshold minimal improvement
        ),
    ]
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Fine-tune!
    print("\n" + "=" * 60)
    print("🏋️ STARTING FINE-TUNING")
    print("=" * 60)
    
    trainer.train()
    
    # Save final model
    print(f"\n💾 Saving fine-tuned model ke: {OUTPUT_PATH}")
    trainer.save_model(OUTPUT_PATH)
    
    # Save tokenizer dengan legacy format untuk compatibility
    try:
        tokenizer.save_pretrained(OUTPUT_PATH, legacy_format=True)
        print(f"✓ Model dan tokenizer saved!")
    except TypeError:
        # Fallback jika legacy_format tidak didukung
        tokenizer.save_pretrained(OUTPUT_PATH)
        print(f"✓ Model dan tokenizer saved (standard format)!")
    
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
