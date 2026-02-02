"""
Test Script untuk Pretrained Indonesian Model
==============================================
Test model cahya/gpt2-small-indonesian-522M langsung

Penggunaan:
    python test_pretrained.py
    python test_pretrained.py --qa  # QA mode
"""

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import sys


def load_pretrained_model():
    """Load model pretrained Indonesia dari HuggingFace"""
    
    model_name = "cahya/gpt2-small-indonesian-522M"
    
    print("=" * 60)
    print(f"📥 DOWNLOADING & LOADING: {model_name}")
    print("=" * 60)
    print("\n⏳ Downloading model (first time: ~2GB)...")
    
    # Load tokenizer dan model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n✅ Model loaded successfully!")
    print(f"   Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    return model, tokenizer, device


def generate_text(model, tokenizer, prompt, device, max_length=100):
    """Generate text dari prompt"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


def ask_question(model, tokenizer, question, device):
    """Test QA dengan format instruction"""
    
    prompt = f"### Instruksi:\n{question}\n\n### Jawaban:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
        )
    
    # Decode only answer part
    answer_tokens = outputs[0][prompt_length:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    
    # Clean answer - ambil 1-2 kalimat
    if '.' in answer:
        sentences = answer.split('.')[:2]
        answer = '.'.join(sentences) + '.'
    
    # Remove markers
    for marker in ["###", "Instruksi:", "Pertanyaan:"]:
        if marker in answer:
            answer = answer.split(marker)[0].strip()
    
    return answer


def test_text_generation(model, tokenizer, device):
    """Test text generation biasa"""
    
    print("\n" + "=" * 60)
    print("📝 TEXT GENERATION TEST")
    print("=" * 60)
    
    test_prompts = [
        "Indonesia adalah negara",
        "Jakarta merupakan",
        "Bahasa Indonesia adalah",
        "Teknologi kecerdasan buatan",
    ]
    
    for prompt in test_prompts:
        print(f"\n💬 Prompt: {prompt}")
        print("-" * 40)
        
        result = generate_text(model, tokenizer, prompt, device, max_length=80)
        print(f"✨ Output: {result}")


def test_qa_mode(model, tokenizer, device):
    """Test QA mode"""
    
    print("\n" + "=" * 60)
    print("🤖 Q&A TEST MODE")
    print("=" * 60)
    
    test_questions = [
        "Halo, apa kabar?",
        "Siapa kamu?",
        "Apa itu Indonesia?",
        "Apa ibu kota Indonesia?",
        "Berapa 1 + 1?",
        "Dimana Semarang?",
        "Apa itu Malaysia?",
    ]
    
    for question in test_questions:
        print(f"\n❓ {question}")
        print("-" * 40)
        
        answer = ask_question(model, tokenizer, question, device)
        print(f"💬 {answer}")


def interactive_qa(model, tokenizer, device):
    """Interactive Q&A"""
    
    print("\n" + "=" * 60)
    print("🤖 INTERACTIVE Q&A MODE")
    print("   Ketik 'quit' untuk keluar")
    print("=" * 60)
    
    while True:
        question = input("\n❓ Pertanyaan: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Sampai jumpa!")
            break
        
        if not question:
            continue
        
        print("\n⏳ Berpikir...")
        answer = ask_question(model, tokenizer, question, device)
        
        print(f"\n💬 Jawaban: {answer}")
        print("-" * 50)


def main():
    # Load model
    model, tokenizer, device = load_pretrained_model()
    
    # Check mode
    if "--qa" in sys.argv:
        # QA mode
        test_qa_mode(model, tokenizer, device)
        interactive_qa(model, tokenizer, device)
    elif "--interactive" in sys.argv:
        # Interactive only
        interactive_qa(model, tokenizer, device)
    elif "--generation" in sys.argv:
        # Text generation only
        test_text_generation(model, tokenizer, device)
    else:
        # Default: test both
        test_text_generation(model, tokenizer, device)
        test_qa_mode(model, tokenizer, device)
        
        # Ask if want interactive
        print("\n" + "=" * 60)
        choice = input("Mau coba interactive Q&A? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            interactive_qa(model, tokenizer, device)


if __name__ == "__main__":
    main()
