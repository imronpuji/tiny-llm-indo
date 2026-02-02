"""
Training Script untuk Tiny Indonesian LLM (13M params)
======================================================
Dengan early stopping dan optimasi untuk menghindari overfitting
Support PEFT (Parameter-Efficient Fine-Tuning) dan LoRA
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

# PEFT & LoRA imports
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
        PeftModel,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  PEFT not installed. Install with: pip install peft")

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
    "resid_pdrop": 0.05,
    "embd_pdrop": 0.05,
    "attn_pdrop": 0.05,
    "layer_norm_epsilon": 1e-5,
    "bos_token_id": 1,
    "eos_token_id": 2,
}

TRAINING_CONFIG = {
    "output_dir": "./tiny-llm-indo",
    "num_train_epochs": 5,                     # 5 epoch = ~1 jam
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
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
    "fp16": False,
    "dataloader_num_workers": 2,
    "dataloader_pin_memory": False,
    "seed": 42,
    "report_to": "none",
    "max_grad_norm": 1.0,
}

# ============================================================
# LORA CONFIGURATION
# ============================================================

USE_LORA = True  # Set False untuk full fine-tuning

LORA_CONFIG = {
    "r": 8,                          # LoRA rank (dimensi adaptasi)
    "lora_alpha": 16,                # LoRA scaling factor
    "lora_dropout": 0.05,            # Dropout untuk LoRA layers
    "bias": "none",                  # Bias training: "none", "all", "lora_only"
    "target_modules": [              # Modules yang akan di-adapt
        "c_attn",                    # Attention weights (Q, K, V)
        "c_proj",                    # Attention output projection
        "c_fc",                      # FFN first layer
    ],
    "task_type": "CAUSAL_LM",        # Task type untuk language modeling
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

def create_model_and_tokenizer(use_lora=False, from_pretrained=None):
    """
    Buat model dan tokenizer baru
    
    Args:
        use_lora: Boolean, apakah menggunakan LoRA
        from_pretrained: Path ke pretrained model (untuk fine-tuning)
    """
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
    
    # Create or load model
    if from_pretrained:
        print(f"📥 Loading pretrained model from: {from_pretrained}")
        model = GPT2LMHeadModel.from_pretrained(from_pretrained)
    else:
        # Create model config
        config = GPT2Config(**MODEL_CONFIG)
        # Create model from scratch
        model = GPT2LMHeadModel(config)
    
    # Apply LoRA if enabled
    if use_lora and PEFT_AVAILABLE:
        model = apply_lora(model)
    elif use_lora and not PEFT_AVAILABLE:
        print("⚠️  LoRA requested but PEFT not available. Using full fine-tuning.")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    return model, tokenizer


def apply_lora(model):
    """
    Apply LoRA adapter ke model
    
    LoRA (Low-Rank Adaptation) menambahkan trainable low-rank matrices
    ke attention layers, memungkinkan fine-tuning efisien dengan
    parameter yang jauh lebih sedikit.
    """
    print("\n🔗 Applying LoRA adapter...")
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        target_modules=LORA_CONFIG["target_modules"],
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply PEFT
    model = get_peft_model(model, lora_config)
    
    # Print LoRA info
    print(f"  LoRA rank (r): {LORA_CONFIG['r']}")
    print(f"  LoRA alpha: {LORA_CONFIG['lora_alpha']}")
    print(f"  Target modules: {LORA_CONFIG['target_modules']}")
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def merge_lora_weights(model_path, output_path):
    """
    Merge LoRA weights ke base model untuk inference yang lebih cepat
    
    Args:
        model_path: Path ke model dengan LoRA adapter
        output_path: Path untuk menyimpan merged model
    """
    print(f"\n🔀 Merging LoRA weights...")
    print(f"   Input: {model_path}")
    print(f"   Output: {output_path}")
    
    # Load base model
    base_model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Merge and unload
    model = model.merge_and_unload()
    
    # Save merged model
    model.save_pretrained(output_path)
    
    print(f"✓ Merged model saved to: {output_path}")
    
    return model


def load_lora_model(base_model_path, lora_adapter_path):
    """
    Load model dengan LoRA adapter terpisah
    
    Args:
        base_model_path: Path ke base model
        lora_adapter_path: Path ke LoRA adapter
    """
    print(f"\n📥 Loading LoRA model...")
    
    # Load base model
    base_model = GPT2LMHeadModel.from_pretrained(base_model_path)
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    print(f"✓ Model loaded with LoRA adapter")
    
    return model


# ============================================================
# MAIN TRAINING
# ============================================================

def main():
    print("=" * 60)
    print("🚀 TINY INDONESIAN LLM TRAINING")
    if USE_LORA and PEFT_AVAILABLE:
        print("   Mode: PEFT with LoRA")
    else:
        print("   Mode: Full Fine-tuning")
    print("=" * 60)
    
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model and tokenizer (dengan LoRA jika diaktifkan)
    model, tokenizer = create_model_and_tokenizer(use_lora=USE_LORA)
    
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
    
    # If using LoRA, optionally merge weights
    if USE_LORA and PEFT_AVAILABLE:
        print("\n💡 LoRA adapter saved. To merge weights for faster inference:")
        print(f"   merge_lora_weights('{final_path}', './tiny-llm-indo-merged')")
        
        # Uncomment to auto-merge:
        # merged_path = "./tiny-llm-indo-merged"
        # merge_lora_weights(final_path, merged_path)
    
    # Final evaluation
    print("\n📊 Final Evaluation:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"   {key}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)


def finetune_with_lora(base_model_path, dataset_path, output_path):
    """
    Fine-tune existing model dengan LoRA
    
    Contoh penggunaan:
        finetune_with_lora(
            base_model_path="./tiny-llm-indo-final",
            dataset_path="./dataset/finetune_data.json",
            output_path="./tiny-llm-indo-lora-finetuned"
        )
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT not installed. Run: pip install peft")
    
    print("=" * 60)
    print("🔧 FINE-TUNING WITH LoRA")
    print("=" * 60)
    
    # Load base model with LoRA
    model, tokenizer = create_model_and_tokenizer(
        use_lora=True,
        from_pretrained=base_model_path
    )
    
    # Load and prepare dataset
    dataset = load_dataset_from_json(dataset_path)
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"],
    )
    
    # Split if needed
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Adjusted training args for fine-tuning
    finetune_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,  # Lower LR for fine-tuning
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=finetune_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✓ Fine-tuned model saved to: {output_path}")
    
    return model, tokenizer


if __name__ == "__main__":
    main()
