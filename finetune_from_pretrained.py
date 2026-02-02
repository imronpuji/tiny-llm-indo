"""
Fine-tune dari Model Pretrained Indonesia
==========================================
Lebih baik untuk dataset kecil (< 100K samples)

Model base: cahya/gpt2-small-indonesian-522M
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

# Model pretrained Indonesia (sudah dilatih dengan banyak data)
BASE_MODEL = "cahya/gpt2-small-indonesian-522M"

# Output
OUTPUT_PATH = "./tiny-llm-indo-qa-finetuned"

# Dataset
TRAIN_DATA_PATH = "./dataset/train_qa.json"
EVAL_DATA_PATH = "./dataset/eval_qa.json"

# Training config - optimized untuk A100 20GB
FINETUNE_CONFIG = {
    "output_dir": "./checkpoints-finetune",
    "num_train_epochs": 5,                     # Cukup untuk fine-tuning
    "per_device_train_batch_size": 8,          # A100 bisa handle
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,          # Effective batch = 32
    "learning_rate": 5e-5,                     # LR kecil untuk fine-tuning
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "logging_steps": 20,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "bf16": True,                              # A100 supports bf16
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
    "seed": 42,
    "report_to": "none",
}


def load_dataset_from_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)


def tokenize_function(examples, tokenizer, max_length=512):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_special_tokens_mask=True,
    )


def main():
    print("=" * 60)
    print("🚀 FINE-TUNE DARI MODEL PRETRAINED INDONESIA")
    print(f"   Base model: {BASE_MODEL}")
    print("=" * 60)
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load pretrained model & tokenizer
    print(f"\n📥 Loading pretrained model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {total_params:,} ({total_params/1e6:.1f}M) parameters")
    
    # Load datasets
    print("\n📂 Loading datasets...")
    
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"❌ Dataset tidak ditemukan: {TRAIN_DATA_PATH}")
        print("   Jalankan: python add_qa_data.py")
        return
    
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
    
    # Callbacks
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=3)
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
    
    # Train
    print("\n" + "=" * 60)
    print("🏋️ STARTING FINE-TUNING")
    print("=" * 60)
    
    trainer.train()
    
    # Save
    print(f"\n💾 Saving model to: {OUTPUT_PATH}")
    trainer.save_model(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    
    # Eval
    print("\n📊 Final Evaluation:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"   {key}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ FINE-TUNING COMPLETE!")
    print(f"   Model saved to: {OUTPUT_PATH}")
    print("\nUntuk test, jalankan:")
    print(f"   python test_model.py --model {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
