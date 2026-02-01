"""
Test Script untuk Tiny Indonesian LLM
=====================================
"""

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

def load_model(model_path="./tiny-llm-indo-final"):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    return model, tokenizer, device


def generate_text(model, tokenizer, prompt, device, 
                  max_length=100, 
                  temperature=0.7,
                  top_k=50,
                  top_p=0.9,
                  num_return=1):
    """Generate text dari prompt"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_return,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    
    results = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        results.append(text)
    
    return results


def interactive_test(model, tokenizer, device):
    """Interactive testing mode"""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE - ketik 'quit' untuk keluar")
    print("=" * 60)
    
    while True:
        prompt = input("\n📝 Prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Bye!")
            break
        
        if not prompt:
            continue
        
        print("\n⏳ Generating...")
        
        # Generate with different temperatures
        results = generate_text(model, tokenizer, prompt, device, 
                               temperature=0.7, num_return=3)
        
        print(f"\n✨ Hasil:")
        for i, text in enumerate(results, 1):
            print(f"\n[{i}] {text}")
        print("-" * 50)


def batch_test(model, tokenizer, device):
    """Test dengan berbagai prompt"""
    
    test_prompts = [
        "Halo, apa kabar?",
        "Indonesia adalah",
        "Jakarta terletak",
        "Bahasa Indonesia",
        "Nasi goreng adalah",
        "Pulau Bali terkenal",
        "Presiden pertama Indonesia",
        "Gunung tertinggi di Jawa",
    ]
    
    print("\n" + "=" * 60)
    print("BATCH TEST")
    print("=" * 60)
    
    for prompt in test_prompts:
        print(f"\n📝 {prompt}")
        print("-" * 40)
        
        results = generate_text(model, tokenizer, prompt, device,
                               temperature=0.7, max_length=80)
        
        for text in results:
            print(f"✨ {text}")


def main():
    # Load model
    model, tokenizer, device = load_model()
    
    # Batch test
    batch_test(model, tokenizer, device)
    
    # Interactive
    interactive_test(model, tokenizer, device)


if __name__ == "__main__":
    main()
