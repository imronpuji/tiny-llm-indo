"""
Fine-tune Model Sendiri (yasmeenimron/masa-ai) untuk Q&A Bahasa Indonesia
===========================================================================
Melanjutkan training dari model sendiri dengan dataset Q&A tambahan
Menggunakan LoRA untuk efisiensi memory GPU

Penggunaan:
    python finetune_qwen.py

Requirements:
    pip install torch transformers datasets accelerate peft bitsandbytes trl
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ============================================================
# KONFIGURASI
# ============================================================

# Base model dari HuggingFace (model sendiri)
BASE_MODEL = "yasmeenimron/masa-ai"

# Output
OUTPUT_PATH = "./masa-ai-continued"
CHECKPOINT_DIR = "./masa-ai-continued-checkpoints"

# Dataset
TRAIN_DATA_PATH = "./dataset/train_qa.json"
EVAL_DATA_PATH = "./dataset/eval_qa.json"

# LoRA config - efisien memory, fine-tune hanya sebagian kecil parameter
LORA_CONFIG = {
    "r": 64,                        # LoRA rank (lebih tinggi = lebih ekspresif)
    "lora_alpha": 128,              # Scaling factor
    "lora_dropout": 0.05,           # Dropout untuk regularisasi
    "target_modules": [             # Layer yang di-finetune
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "task_type": TaskType.CAUSAL_LM,
    "bias": "none",
}

# Quantization - 4bit untuk hemat VRAM
USE_4BIT = True  # Set False jika GPU VRAM >= 24GB

# Training config
TRAINING_CONFIG = {
    "output_dir": CHECKPOINT_DIR,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,       # Effective batch = 16
    "learning_rate": 2e-4,                  # Lebih tinggi untuk LoRA
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "logging_steps": 25,
    "eval_strategy": "steps",
    "eval_steps": 100,
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "fp16": False,
    "bf16": torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    "dataloader_num_workers": 2,
    "seed": 42,
    "report_to": "none",
    "max_grad_norm": 0.3,
    "optim": "paged_adamw_8bit",            # Memory-efficient optimizer
    "gradient_checkpointing": True,         # Trade compute for memory
    "group_by_length": True,                # Batch similar lengths
    "max_seq_length": 512,
}


# ============================================================
# CHAT TEMPLATE
# ============================================================

SYSTEM_PROMPT = (
    "Kamu adalah Masa AI, asisten AI berbahasa Indonesia yang cerdas dan membantu. "
    "Jawab pertanyaan dengan akurat, jelas, dan ringkas dalam bahasa Indonesia."
)

def format_chat(q, a, cot=None):
    """Format QA pair ke Qwen chat template"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
    ]
    
    if cot:
        # Chain of Thought: tambah reasoning sebelum jawaban
        answer = f"Mari kita pikirkan langkah demi langkah.\n{cot}\n\nJadi, {a}"
    else:
        answer = a
    
    messages.append({"role": "assistant", "content": answer})
    return messages


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("🚀 FINE-TUNE MODEL SENDIRI UNTUK Q&A INDONESIA")
    print("=" * 60)
    print()
    
    # Check dataset
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"❌ Dataset tidak ditemukan: {TRAIN_DATA_PATH}")
        print("   Jalankan dulu: python prepare_qa_from_topics.py")
        return
    
    # Device info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {gpu_mem:.1f} GB")
        print(f"   4-bit quantization: {'ON' if USE_4BIT else 'OFF'}")
    else:
        print("⚠️  WARNING: Training on CPU will be very slow!")
        print("   Sangat disarankan menggunakan GPU.")
    
    # ── Load tokenizer ──────────────────────────────────────
    print(f"\n📦 Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"   Vocab size: {tokenizer.vocab_size}")
    
    # ── Load model ──────────────────────────────────────────
    print(f"\n📦 Loading model: {BASE_MODEL}")
    
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    
    if USE_4BIT and torch.cuda.is_available():
        print("   Loading with 4-bit quantization (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    
    # Prepare for training
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)
    
    model.config.use_cache = False  # Required for gradient checkpointing
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded: {total_params/1e9:.2f}B parameters")
    
    # ── Apply LoRA ──────────────────────────────────────────
    print(f"\n🔧 Applying LoRA adapter...")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable_params/1e6:.1f}M / {all_params/1e9:.2f}B ({100*trainable_params/all_params:.2f}%)")
    
    # ── Load & format dataset ───────────────────────────────
    print(f"\n📂 Loading datasets...")
    
    with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
        train_raw = json.load(f)
    with open(EVAL_DATA_PATH, 'r', encoding='utf-8') as f:
        eval_raw = json.load(f)
    
    print(f"   Raw train: {len(train_raw)} samples")
    print(f"   Raw eval: {len(eval_raw)} samples")
    
    # Format ke chat template
    print(f"\n🔄 Formatting ke chat template...")
    
    def extract_qa_from_text(item):
        """Extract q/a from pre-formatted text or dict"""
        text = item.get("text", "")
        q = item.get("q", "")
        a = item.get("a", "")
        cot = item.get("cot", "")
        
        if q and a:
            return q, a, cot
        
        # Parse dari format instruction
        if "### Instruksi:" in text and "### Jawaban:" in text:
            parts = text.split("### Jawaban:")
            q_part = parts[0].replace("### Instruksi:", "").strip()
            a_part = parts[1].strip() if len(parts) > 1 else ""
            
            # Check for COT
            cot_part = ""
            if "### Pemikiran:" in q_part:
                q_parts = q_part.split("### Pemikiran:")
                q_part = q_parts[0].strip()
                cot_part = q_parts[1].strip() if len(q_parts) > 1 else ""
            
            return q_part, a_part, cot_part
        
        # Parse dari format simple
        if "Pertanyaan:" in text and "Jawaban:" in text:
            parts = text.split("Jawaban:")
            q_part = parts[0].replace("Pertanyaan:", "").strip()
            a_part = parts[1].strip() if len(parts) > 1 else ""
            return q_part, a_part, ""
        
        return "", "", ""
    
    def format_dataset(raw_data):
        formatted = []
        skipped = 0
        for item in raw_data:
            q, a, cot = extract_qa_from_text(item)
            if q and a:
                messages = format_chat(q, a, cot if cot else None)
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                formatted.append({"text": text})
            else:
                skipped += 1
        if skipped:
            print(f"   ⚠️  Skipped {skipped} invalid entries")
        return formatted
    
    train_formatted = format_dataset(train_raw)
    eval_formatted = format_dataset(eval_raw)
    
    print(f"   ✓ Train: {len(train_formatted)} samples")
    print(f"   ✓ Eval: {len(eval_formatted)} samples")
    
    if train_formatted:
        print(f"\n📝 Sample formatted data:")
        print("-" * 60)
        sample = train_formatted[0]["text"]
        print(sample[:500] + ("..." if len(sample) > 500 else ""))
        print("-" * 60)
    
    train_dataset = Dataset.from_list(train_formatted)
    eval_dataset = Dataset.from_list(eval_formatted)
    
    # ── Training ────────────────────────────────────────────
    print(f"\n🏋️ STARTING TRAINING")
    print("=" * 60)
    
    # Extract max_seq_length from config
    max_seq_length = TRAINING_CONFIG.pop("max_seq_length", 512)
    
    training_args = TrainingArguments(**TRAINING_CONFIG)
    
    # Use SFTTrainer from trl for cleaner supervised fine-tuning
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False,
    )
    
    # Train!
    trainer.train()
    
    # ── Save model ──────────────────────────────────────────
    print(f"\n💾 Saving model to: {OUTPUT_PATH}")
    
    # Save LoRA adapter
    trainer.save_model(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    
    # Also save merged model (LoRA + base = standalone model)
    merged_path = OUTPUT_PATH + "-merged"
    print(f"💾 Saving merged model to: {merged_path}")
    
    try:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"✓ Merged model saved!")
    except Exception as e:
        print(f"⚠️  Could not save merged model: {e}")
        print("   You can merge later with: python merge_model.py")
    
    # ── Final eval ──────────────────────────────────────────
    print(f"\n📊 Final Evaluation:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"   {key}: {value:.4f}")
    
    # ── Done ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ FINE-TUNING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved:")
    print(f"  LoRA adapter: {OUTPUT_PATH}")
    print(f"  Merged model: {merged_path}")
    print(f"\nTest model:")
    print(f"  python test_model_qwen.py {merged_path}")
    print(f"  python test_model_qwen.py {OUTPUT_PATH}  # (needs base model)")


if __name__ == "__main__":
    main()
