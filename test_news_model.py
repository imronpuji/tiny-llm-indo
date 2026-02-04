"""
Test Model Berita Indonesia
===========================
Test kemampuan model untuk melanjutkan teks/artikel

Penggunaan:
    python test_news_model.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./masa-ai-qa-v4"


def load_model(model_path: str):
    """Load model dan tokenizer"""
    print(f"📦 Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"   Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer, device


def generate_text(
    model, 
    tokenizer, 
    device,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    num_return_sequences: int = 1,
) -> str:
    """Generate text dari prompt"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


def main():
    print("=" * 60)
    print("📰 TEST MODEL BERITA INDONESIA")
    print("=" * 60)
    
    # Load model
    model, tokenizer, device = load_model(MODEL_PATH)
    
    # Test prompts
    test_prompts = [
        "Indonesia adalah negara",
        "Pariwisata di Bali terus berkembang",
        "Pemerintah mengumumkan kebijakan baru",
        "Dalam pertandingan sepak bola hari ini,",
        "Menurut data dari Badan Pusat Statistik,",
        "Cuaca di Jakarta hari ini",
        "Kuliner Indonesia yang terkenal di dunia",
        "Teknologi artificial intelligence semakin",
        "Perekonomian Indonesia pada kuartal ini",
        "Presiden dalam sambutannya menyatakan bahwa",
    ]
    
    print("\n📝 Generating text completions...\n")
    print("=" * 60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}] Prompt: {prompt}")
        print("-" * 40)
        
        generated = generate_text(
            model, tokenizer, device,
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7,
        )
        
        # Tampilkan hasil (highlight bagian yang di-generate)
        continuation = generated[len(prompt):]
        print(f"Generated: {prompt}\033[92m{continuation}\033[0m")
        print("-" * 60)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("💬 INTERACTIVE MODE")
    print("=" * 60)
    print("Ketik awal kalimat, model akan melanjutkan.")
    print("Ketik 'quit' untuk keluar.\n")
    
    while True:
        try:
            prompt = input("📝 Prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nTerima kasih! Sampai jumpa.")
                break
            
            if not prompt:
                continue
            
            generated = generate_text(
                model, tokenizer, device,
                prompt=prompt,
                max_new_tokens=150,
                temperature=0.7,
            )
            
            continuation = generated[len(prompt):]
            print(f"\n📄 Hasil: {prompt}\033[92m{continuation}\033[0m\n")
            
        except KeyboardInterrupt:
            print("\n\nTerima kasih! Sampai jumpa.")
            break


if __name__ == "__main__":
    main()
