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

# Path model dari tahap 1 (training general)
BASE_MODEL_PATH = "./tiny-llm-indo-final"

# Output fine-tuned model
OUTPUT_PATH = "./tiny-llm-indo-qa"

# Dataset Q&A
TRAIN_DATA_PATH = "./dataset/train_qa.json"
EVAL_DATA_PATH = "./dataset/eval_qa.json"

# Training config untuk fine-tuning model 150M
FINETUNE_CONFIG = {
    "output_dir": "./tiny-llm-indo-qa-checkpoints",
    "num_train_epochs": 10,                    # Reduced dari 20 → 10 (2x lebih cepat!)
    "per_device_train_batch_size": 4,          # Naikkan dari 2 → 4
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 8,          # Turunkan dari 16 → 8
    "learning_rate": 3e-5,                     # Sedikit lebih tinggi untuk converge cepat
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "logging_steps": 20,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "fp16": torch.cuda.is_available(),
    "dataloader_num_workers": 2,
    "seed": 42,
    "report_to": "none",
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
    
    # Check base model exists
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"❌ Base model tidak ditemukan: {BASE_MODEL_PATH}")
        print("   Jalankan training general dulu:")
        print("   python train_tiny_llm.py")
        return
    
    # Check Q&A dataset exists
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"❌ Dataset Q&A tidak ditemukan: {TRAIN_DATA_PATH}")
        print("   Jalankan dulu:")
        print("   python add_qa_data.py")
        return
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load base model
    print(f"\n📦 Loading base model dari: {BASE_MODEL_PATH}")
    model = GPT2LMHeadModel.from_pretrained(BASE_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    
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
    
    # Callbacks
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.01
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
