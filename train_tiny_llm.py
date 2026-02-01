"""
Training Script untuk Tiny Indonesian LLM (13M params)
======================================================
Dengan early stopping dan optimasi untuk menghindari overfitting
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from transformers.trainer_callback import TrainerCallback
import math

# ============================================================
# KONFIGURASI MODEL 13M
# ============================================================

MODEL_CONFIG = {
    "vocab_size": 32000,
    "n_positions": 512,
    "n_embd": 384,        # Hidden size
    "n_layer": 6,         # Layers
    "n_head": 6,          # Attention heads
    "n_inner": 1536,      # FFN hidden (4x n_embd)
    "activation_function": "gelu_new",
    "resid_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "layer_norm_epsilon": 1e-5,
    "bos_token_id": 1,
    "eos_token_id": 2,
}

TRAINING_CONFIG = {
    "output_dir": "./tiny-llm-indo",
    "num_train_epochs": 15,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "logging_steps": 50,
    "eval_strategy": "steps",
    "eval_steps": 200,
    "save_strategy": "steps",
    "save_steps": 200,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "fp16": torch.cuda.is_available(),
    "dataloader_num_workers": 4,
    "seed": 42,
    "report_to": "none",
}

# ============================================================
# CUSTOM CALLBACK UNTUK MONITORING
# ============================================================

class LossMonitorCallback(TrainerCallback):
    """Monitor loss dan detect overfitting"""
    
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_eval_loss = float('inf')
        self.wait = 0
        self.train_losses = []
        self.eval_losses = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
                
                # Check overfitting
                if len(self.eval_losses) >= 3:
                    recent = self.eval_losses[-3:]
                    if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                        print("\n⚠️  WARNING: Eval loss increasing for 3 consecutive evals!")
                        print("    Consider stopping training to prevent overfitting.")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            current_loss = metrics['eval_loss']
            
            # Calculate gap
            if self.train_losses:
                train_loss = self.train_losses[-1]
                gap = current_loss - train_loss
                print(f"\n📊 Loss Gap (eval - train): {gap:.3f}")
                
                if gap > 1.0:
                    print("⚠️  Large gap detected - possible overfitting!")


# ============================================================
# DATA LOADING & TOKENIZATION
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


# ============================================================
# MODEL CREATION
# ============================================================

def create_model_and_tokenizer():
    """Buat model dan tokenizer baru"""
    print("\n🔧 Creating model and tokenizer...")
    
    # Load tokenizer Indonesia
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "cahya/gpt2-small-indonesian-522M",
            trust_remote_code=True
        )
    except:
        # Fallback ke GPT2 tokenizer
        print("⚠ Indonesian tokenizer not found, using GPT2 tokenizer")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Set special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Update vocab size
    MODEL_CONFIG["vocab_size"] = len(tokenizer)
    
    # Create model config
    config = GPT2Config(**MODEL_CONFIG)
    
    # Create model from scratch
    model = GPT2LMHeadModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model, tokenizer


# ============================================================
# MAIN TRAINING
# ============================================================

def main():
    print("=" * 60)
    print("🚀 TINY INDONESIAN LLM TRAINING")
    print("=" * 60)
    
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer()
    
    # Load datasets
    print("\n📂 Loading datasets...")
    
    train_path = "./dataset/train.json"
    eval_path = "./dataset/eval.json"
    
    if not os.path.exists(train_path):
        print("❌ Dataset not found! Run prepare_dataset.py first.")
        return
    
    train_dataset = load_dataset_from_json(train_path)
    eval_dataset = load_dataset_from_json(eval_path)
    
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
        mlm=False,  # Causal LM
    )
    
    # Training arguments
    training_args = TrainingArguments(**TRAINING_CONFIG)
    
    # Callbacks
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.01
        ),
        LossMonitorCallback(patience=5)
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
    
    # Train!
    print("\n" + "=" * 60)
    print("🏋️ STARTING TRAINING")
    print("=" * 60)
    
    trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    final_path = "./tiny-llm-indo-final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"✓ Model saved to: {final_path}")
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"   {key}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
