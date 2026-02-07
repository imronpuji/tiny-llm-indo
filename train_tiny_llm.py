"""
Training Script untuk Tiny Indonesian LLM (150M params)
======================================================
Optimized for NVIDIA H200 NVL (140GB VRAM)
Dengan early stopping dan optimasi untuk menghindari overfitting
Support PEFT (Parameter-Efficient Fine-Tuning) dan LoRA
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# H200 NVL Optimizations
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"  # H200 = Hopper
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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
# KONFIGURASI MODEL - PILIH SALAH SATU!
# ============================================================

# === TINY: 13M params (~1 jam training) ===
# MODEL_CONFIG = {
#     "vocab_size": 32000,
#     "n_positions": 512,
#     "n_embd": 384,
#     "n_layer": 6,
#     "n_head": 6,
#     "n_inner": 1536,
#     "activation_function": "gelu_new",
#     "resid_pdrop": 0.05,
#     "embd_pdrop": 0.05,
#     "attn_pdrop": 0.05,
#     "layer_norm_epsilon": 1e-5,
#     "bos_token_id": 1,
#     "eos_token_id": 2,
# }

# === SMALL: 30M params (~2-3 jam training) ===
# MODEL_CONFIG = {
#     "vocab_size": 32000,
#     "n_positions": 512,
#     "n_embd": 512,
#     "n_layer": 8,
#     "n_head": 8,
#     "n_inner": 2048,
#     "activation_function": "gelu_new",
#     "resid_pdrop": 0.1,
#     "embd_pdrop": 0.1,
#     "attn_pdrop": 0.1,
#     "layer_norm_epsilon": 1e-5,
#     "bos_token_id": 1,
#     "eos_token_id": 2,
# }

# === MEDIUM: 85M params (~4-6 jam training) ===
# MODEL_CONFIG = {
#     "vocab_size": 32000,
#     "n_positions": 1024,
#     "n_embd": 768,
#     "n_layer": 12,
#     "n_head": 12,
#     "n_inner": 3072,
#     "activation_function": "gelu_new",
#     "resid_pdrop": 0.1,
#     "embd_pdrop": 0.1,
#     "attn_pdrop": 0.1,
#     "layer_norm_epsilon": 1e-5,
#     "bos_token_id": 1,
#     "eos_token_id": 2,
# }

# === LARGE: 150M params - OPTIMIZED FOR H200 NVL ===
MODEL_CONFIG = {
    "vocab_size": 32000,
    "n_positions": 2048,   # Extended context — H200 punya VRAM lebih dari cukup
    "n_embd": 1024,        # Hidden size besar
    "n_layer": 12,         # 12 layers
    "n_head": 16,          # 16 attention heads
    "n_inner": 4096,       # 4x n_embd
    "activation_function": "gelu_new",
    "resid_pdrop": 0.05,   # Lower dropout agar model lebih stabil & coherent
    "embd_pdrop": 0.05,
    "attn_pdrop": 0.05,
    "layer_norm_epsilon": 1e-5,
    "bos_token_id": 1,
    "eos_token_id": 2,
}

# ============================================================
# H200 NVL SPECS:
#   - 140.4 GB VRAM (HBM3e)
#   - 48.3 TFLOPS (FP32), ~989 TFLOPS (FP8)
#   - 3862.3 GB/s memory bandwidth
#   - PCIe 5.0 x16
#   - 24 CPU cores (AMD EPYC 9255)
# ============================================================

TRAINING_CONFIG = {
    "output_dir": "./tiny-llm-indo",
    "num_train_epochs": 3,                     # 3 epoch cukup, lebih dari itu overfitting
    "per_device_train_batch_size": 128,        # H200 140GB bisa handle batch besar untuk 150M model
    "per_device_eval_batch_size": 128,
    "gradient_accumulation_steps": 2,          # Effective batch = 128*2 = 256 (sangat stabil convergence)
    "learning_rate": 6e-4,                     # Chinchilla-optimal LR untuk 150M
    "weight_decay": 0.1,                       # Weight decay lebih tinggi untuk regularisasi
    "warmup_ratio": 0.06,                      # 6% warmup — batch besar butuh less warmup
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 200,
    "save_strategy": "steps",
    "save_steps": 500,
    "save_total_limit": 5,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "bf16": True,                              # H200 native bf16 support (Hopper arch)
    "bf16_full_eval": True,                    # bf16 juga saat eval
    "dataloader_num_workers": 24,              # Match 24 CPU cores
    "dataloader_pin_memory": True,
    "dataloader_prefetch_factor": 4,           # Prefetch lebih banyak data
    "ddp_find_unused_parameters": False,
    "gradient_checkpointing": False,           # MATIKAN — H200 140GB cukup VRAM, lebih cepat tanpa ini
    "seed": 42,
    "report_to": "none",
    "max_grad_norm": 1.0,
    "adam_beta1": 0.9,                         # Standard untuk LLM training
    "adam_beta2": 0.95,                        # 0.95 lebih stabil dari default 0.999
    "adam_epsilon": 1e-8,
    "torch_compile": True,                     # torch.compile() — fuse kernels untuk speedup 20-40%
    "optim": "adamw_torch_fused",              # Fused AdamW — lebih cepat di H200
}

# ============================================================
# LORA CONFIGURATION
# ============================================================

USE_LORA = False  # Set False untuk full fine-tuning (lebih simpel)

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


def tokenize_function(examples, tokenizer, max_length=2048):
    """Tokenize texts dengan EOS token di akhir setiap sample"""
    # Tambahkan EOS token di akhir setiap teks agar model belajar kapan berhenti
    texts = [t + tokenizer.eos_token for t in examples["text"]]
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # Dynamic padding via DataCollator — lebih efisien
        return_special_tokens_mask=True,
    )


# ============================================================
# MODEL CREATION
# ============================================================

def _init_weights(model, config):
    """Inisialisasi weight yang lebih baik untuk training dari awal.
    Mengikuti GPT-2 paper: scaled initialization berdasarkan depth."""
    import math
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'ln' in name or 'layernorm' in name:
                # LayerNorm weights diinisialisasi ke 1
                torch.nn.init.ones_(param)
            elif 'wte' in name or 'wpe' in name:
                # Token & position embeddings
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'c_proj' in name:
                # Output projection: scaled by 1/sqrt(2*n_layer) per GPT-2
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
            else:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
        elif 'bias' in name:
            torch.nn.init.zeros_(param)
    print("  ✓ Weights initialized (GPT-2 scaled init)")


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
        # Better weight initialization untuk coherence
        _init_weights(model, config)
    
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
# DATASET MODE: Pilih salah satu
# ============================================================
# "qa"      = QA data kecil (dari add_qa_data.py)
# "general" = Teks biasa (dari prepare_dataset.py)  
# "large"   = Dataset besar Wikipedia + CC100 (dari prepare_large_dataset.py)
DATASET_MODE = "large"  # Recommended untuk knowledge yang baik


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
    print(f"   Dataset: {DATASET_MODE}")
    print("=" * 60)
    
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model and tokenizer (dengan LoRA jika diaktifkan)
    model, tokenizer = create_model_and_tokenizer(use_lora=USE_LORA)
    
    # Load datasets (pilih berdasarkan mode)
    print("\n📂 Loading datasets...")
    
    if DATASET_MODE == "qa":
        train_path = "./dataset/train_qa.json"
        eval_path = "./dataset/eval_qa.json"
    elif DATASET_MODE == "large":
        train_path = "./dataset/train_large.json"
        eval_path = "./dataset/eval_large.json"
    else:
        train_path = "./dataset/train.json"
        eval_path = "./dataset/eval.json"
    
    if not os.path.exists(train_path):
        print(f"❌ Dataset not found: {train_path}")
        if DATASET_MODE == "qa":
            print("   Run: python add_qa_data.py")
        elif DATASET_MODE == "large":
            print("   Run: python prepare_large_dataset.py")
        else:
            print("   Run: python prepare_dataset.py")
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
    
    # Save model
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Jika pakai LoRA, simpan juga config base model untuk loading nanti
    if USE_LORA and PEFT_AVAILABLE:
        # Save the base model config
        config = GPT2Config(**MODEL_CONFIG)
        config.save_pretrained(final_path)
        print(f"✓ LoRA adapter + config saved to: {final_path}")
    else:
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
