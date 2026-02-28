"""
Test Base Model dari HuggingFace
=================================
Script untuk verify bahwa base model bisa diload
"""

from transformers import GPT2LMHeadModel, AutoTokenizer
import sys

model_name = 'yasmeenimron/masa-ai-qa'

print("=" * 60)
print(f"🔍 CHECKING MODEL: {model_name}")
print("=" * 60)
print()

try:
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"✓ Tokenizer loaded")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    print("\n📥 Loading model...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded")
    print(f"  Total parameters: {total_params/1e6:.1f}M")
    print(f"  Model type: {model.config.model_type}")
    print(f"  Hidden size: {model.config.n_embd}")
    print(f"  Layers: {model.config.n_layer}")
    print(f"  Attention heads: {model.config.n_head}")
    
    # Test generation
    print("\n🧪 Testing generation...")
    import torch
    
    prompt = "Halo, apa kabar?"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Prompt: {prompt}")
    print(f"  Generated: {generated}")
    
    print("\n" + "=" * 60)
    print("✅ MODEL READY FOR FINE-TUNING!")
    print("=" * 60)
    print("\nNext step:")
    print("  python finetune_qa.py")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nPastikan:")
    print("1. Koneksi internet aktif")
    print("2. Model name benar")
    print("3. transformers installed: pip install transformers")
    sys.exit(1)
