"""
Fine-tuning Script untuk Indonesian General QA
Dataset: ernaamaliaw/Indonesian-General-QA
Model: yasmeenimron/masa-ai (atau base model lainnya)

Menggunakan LAB-style hyperparameters untuk meningkatkan pemahaman
"""

import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    GPT2LMHeadModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import evaluate
import numpy as np
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    # Model
    BASE_MODEL = "yasmeenimron/masa-ai"  # atau "gpt2" atau model lain
    OUTPUT_DIR = "./masa-ai-qa-indonesian"
    
    # Dataset
    DATASET_NAME = "ernaamaliaw/Indonesian-General-QA"
    MAX_LENGTH = 256
    
    # LAB-style Hyperparameters
    BATCH_SIZE = 8  # Per device
    GRADIENT_ACCUMULATION_STEPS = 120  # Effective batch size = 8 * 120 * num_gpus = 960 (adjust based on resources)
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 25
    NUM_EPOCHS = 10
    WEIGHT_DECAY = 0.01
    LR_SCHEDULER = "constant_with_warmup"  # No decay after warmup
    
    # LoRA Configuration
    USE_LORA = True
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["c_attn", "c_proj"]  # For GPT2
    
    # Training
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    LOGGING_STEPS = 100
    SAVE_TOTAL_LIMIT = 3
    SEED = 42


# ============================================================
# DATASET PREPARATION
# ============================================================

def format_qa_prompt(example):
    """
    Format dataset menjadi prompt QA yang konsisten
    Format: Pertanyaan: {question}\nJawaban: {answer}
    """
    question = example["Pertanyaan"].strip()
    answer = example["Jawaban"].strip()
    
    # Pastikan pertanyaan diakhiri dengan tanda tanya
    if not question.endswith("?"):
        question += "?"
    
    # Format prompt
    text = f"Pertanyaan: {question}\nJawaban: {answer}"
    
    return {"text": text}


def tokenize_function(examples, tokenizer, max_length=256):
    """Tokenize text dan tambahkan labels"""
    
    # Tokenize
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    
    # Untuk causal LM, labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def prepare_dataset(config):
    """Load dan prepare dataset"""
    
    print("📥 Loading dataset...")
    
    # Check jika dataset adalah local directory
    if os.path.exists(config.DATASET_NAME):
        print(f"   Loading from local: {config.DATASET_NAME}")
        from datasets import load_from_disk
        dataset_dict = load_from_disk(config.DATASET_NAME)
        
        # Convert ke DatasetDict jika belum
        if isinstance(dataset_dict, Dataset):
            dataset = {"train": dataset_dict}
        else:
            dataset = dataset_dict
    else:
        print(f"   Loading from HuggingFace: {config.DATASET_NAME}")
        dataset = load_dataset(config.DATASET_NAME)
    
    # Dataset info
    print(f"\n📊 Dataset Info:")
    print(f"   - Total samples: {len(dataset['train']):,}")
    print(f"   - Columns: {dataset['train'].column_names}")
    
    # Sample data
    print(f"\n📝 Sample data:")
    sample = dataset['train'][0]
    print(f"   Pertanyaan: {sample['Pertanyaan'][:100]}...")
    print(f"   Jawaban: {sample['Jawaban'][:100]}...")
    
    # Format ke QA prompt
    print("\n🔄 Formatting dataset...")
    formatted_dataset = dataset.map(
        format_qa_prompt,
        remove_columns=dataset['train'].column_names,
        desc="Formatting QA pairs"
    )
    
    # Split train/validation jika belum ada
    if "validation" not in formatted_dataset:
        print("\n✂️  Creating train/validation split (90/10)...")
        split_dataset = formatted_dataset['train'].train_test_split(
            test_size=0.1, 
            seed=config.SEED
        )
        formatted_dataset['train'] = split_dataset['train']
        formatted_dataset['validation'] = split_dataset['test']
    
    print(f"\n✅ Dataset prepared:")
    print(f"   - Train: {len(formatted_dataset['train']):,} samples")
    print(f"   - Validation: {len(formatted_dataset['validation']):,} samples")
    
    return formatted_dataset


# ============================================================
# MODEL SETUP
# ============================================================

def setup_model_and_tokenizer(config):
    """Setup model dan tokenizer"""
    
    print(f"\n🤖 Loading model: {config.BASE_MODEL}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    
    # Set pad token jika belum ada
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.float32,  # Atau torch.float16 untuk training lebih cepat
    )
    
    # Resize token embeddings jika ada token baru
    model.resize_token_embeddings(len(tokenizer))
    
    # Setup LoRA jika diaktifkan
    if config.USE_LORA:
        print("\n🔧 Setting up LoRA...")
        
        # Enable gradient checkpointing untuk model base
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # Fix untuk gradient checkpointing
        
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            target_modules=config.LORA_TARGET_MODULES,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # Full fine-tuning - enable gradients for all parameters
        print("\n🔧 Full fine-tuning mode (no LoRA)")
        for param in model.parameters():
            param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer


# ============================================================
# METRICS & EVALUATION
# ============================================================

def compute_metrics(eval_pred):
    """Compute metrics untuk evaluation"""
    
    logits, labels = eval_pred
    
    # Shift untuk causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate perplexity
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels_flat = shift_labels.view(-1)
    
    # Filter out padding
    mask = shift_labels_flat != -100
    shift_logits_filtered = shift_logits_flat[mask]
    shift_labels_filtered = shift_labels_flat[mask]
    
    loss = loss_fct(shift_logits_filtered, shift_labels_filtered)
    perplexity = torch.exp(loss).item()
    
    return {
        "perplexity": perplexity,
        "loss": loss.item(),
    }


# ============================================================
# TRAINING
# ============================================================

def train(config):
    """Main training function"""
    
    # Seed untuk reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    # Prepare dataset
    dataset = prepare_dataset(config)
    
    # Setup model dan tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Tokenize dataset
    print("\n🔤 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, config.MAX_LENGTH),
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )
    
    # Calculate effective batch size
    effective_batch_size = (
        config.BATCH_SIZE * 
        config.GRADIENT_ACCUMULATION_STEPS * 
        torch.cuda.device_count() if torch.cuda.is_available() else 1
    )
    
    print(f"\n📊 Training Configuration:")
    print(f"   - Base model: {config.BASE_MODEL}")
    print(f"   - Per-device batch size: {config.BATCH_SIZE}")
    print(f"   - Gradient accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"   - Effective batch size: {effective_batch_size}")
    print(f"   - Learning rate: {config.LEARNING_RATE}")
    print(f"   - Warmup steps: {config.WARMUP_STEPS}")
    print(f"   - Scheduler: {config.LR_SCHEDULER}")
    print(f"   - Epochs: {config.NUM_EPOCHS}")
    print(f"   - Max length: {config.MAX_LENGTH}")
    
    # Training arguments (LAB-style)
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        
        # Batch size & accumulation
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        
        # Learning rate & scheduler
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type=config.LR_SCHEDULER,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        
        # Epochs
        num_train_epochs=config.NUM_EPOCHS,
        
        # Evaluation & Saving
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        
        # Logging
        logging_steps=config.LOGGING_STEPS,
        logging_dir=f"{config.OUTPUT_DIR}/logs",
        report_to=["tensorboard"],
        
        # Optimization
        fp16=torch.cuda.is_available(),  # Mixed precision jika ada GPU
        gradient_checkpointing=False if config.USE_LORA else True,  # Disable for LoRA (handled separately)
        
        # Misc
        seed=config.SEED,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train!
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    train_result = trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    
    # Training metrics
    print("\n✅ Training completed!")
    print(f"\n📊 Training Metrics:")
    print(f"   - Final loss: {train_result.training_loss:.4f}")
    print(f"   - Training time: {train_result.metrics['train_runtime']:.2f}s")
    print(f"   - Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")
    
    # Final evaluation
    print("\n📈 Running final evaluation...")
    eval_results = trainer.evaluate()
    
    print(f"\n📊 Evaluation Metrics:")
    for key, value in eval_results.items():
        print(f"   - {key}: {value:.4f}")
    
    print(f"\n✅ Model saved to: {config.OUTPUT_DIR}")
    print("\nUntuk test model, jalankan:")
    print(f"   python test_masa_ai.py")


# ============================================================
# TESTING
# ============================================================

def test_model(config, test_questions=None):
    """Test model dengan beberapa pertanyaan"""
    
    if test_questions is None:
        test_questions = [
            "Apa itu semester antara?",
            "Siapa yang dapat mengikuti semester antara?",
            "Apa ibu kota Indonesia?",
            "Kapan Indonesia merdeka?",
        ]
    
    print("\n🧪 Testing model...")
    print("=" * 60)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(config.OUTPUT_DIR)
    model = GPT2LMHeadModel.from_pretrained(config.OUTPUT_DIR)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Test each question
    for i, question in enumerate(test_questions, 1):
        prompt = f"Pertanyaan: {question}\nJawaban:"
        
        print(f"\n[{i}] {question}")
        print("-" * 40)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.3,
                top_p=0.85,
                top_k=40,
                do_sample=True,
                repetition_penalty=1.5,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode hanya jawaban
        answer_tokens = outputs[0][prompt_length:]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # Clean up
        if "Pertanyaan:" in answer:
            answer = answer.split("Pertanyaan:")[0].strip()
        
        print(f"💬 {answer}")
    
    print("\n" + "=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Indonesian QA Model")
    parser.add_argument("--test-only", action="store_true", help="Only test the model")
    parser.add_argument("--base-model", type=str, default=None, help="Base model to use")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    
    # Override config dari args
    if args.base_model:
        config.BASE_MODEL = args.base_model
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
    
    # Test only atau train
    if args.test_only:
        test_model(config)
    else:
        train(config)
        
        # Test setelah training
        print("\n" + "=" * 60)
        answer = input("Test model sekarang? (y/n): ").strip().lower()
        if answer == 'y':
            test_model(config)


if __name__ == "__main__":
    main()
