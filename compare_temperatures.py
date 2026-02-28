"""
Compare Model Output dengan Temperature Berbeda
================================================
Script untuk membandingkan kualitas jawaban dengan temperature berbeda
"""

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer


def load_model(model_path):
    """Load model dan tokenizer"""
    print(f"Loading model from {model_path}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        print("⚠️  Using fallback tokenizer: cahya/gpt2-small-indonesian-522M")
        tokenizer = AutoTokenizer.from_pretrained("cahya/gpt2-small-indonesian-522M")
    
    model = GPT2LMHeadModel.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}\n")
    return model, tokenizer, device


def generate_with_temp(model, tokenizer, prompt, device, temperature):
    """Generate dengan temperature tertentu"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=120,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.92,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
    
    # Decode hanya jawaban (skip prompt)
    answer_tokens = outputs[0][prompt_length:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    
    # Clean up - ambil max 2 kalimat
    sentences = []
    current = ""
    for char in answer:
        current += char
        if char in '.!?':
            sentences.append(current.strip())
            current = ""
            if len(sentences) >= 2:
                break
    
    if sentences:
        return ' '.join(sentences)
    return answer


def compare_temperatures(model, tokenizer, device):
    """Compare berbagai temperature"""
    
    questions = [
        "Siapa presiden pertama Indonesia?",
        "Apa itu fotosintesis?",
        "Mengapa langit berwarna biru?",
        "Bagaimana cara membuat nasi goreng?",
        "Apa manfaat olahraga?",
    ]
    
    temperatures = [0.3, 0.4, 0.5, 0.7, 0.9]
    
    print("=" * 80)
    print("🔥 TEMPERATURE COMPARISON")
    print("=" * 80)
    print("Legend:")
    print("  • 0.3 = Sangat faktual, konsisten, kadang repetitif")
    print("  • 0.4 = Balance faktual + natural (RECOMMENDED)")
    print("  • 0.5 = Sedikit lebih kreatif")
    print("  • 0.7 = Kreatif, kadang kurang tepat")
    print("  • 0.9 = Sangat kreatif, sering tidak akurat")
    print("=" * 80)
    
    for question in questions:
        print(f"\n\n❓ PERTANYAAN: {question}")
        print("-" * 80)
        
        prompt = f"### Instruksi:\n{question}\n\n### Jawaban:"
        
        for temp in temperatures:
            answer = generate_with_temp(model, tokenizer, prompt, device, temp)
            print(f"\n🌡️  Temperature {temp}:")
            print(f"   {answer}")
    
    print("\n" + "=" * 80)
    print("📊 RECOMMENDATION:")
    print("   • Untuk Q&A faktual: gunakan temperature 0.3-0.4")
    print("   • Untuk conversational: gunakan temperature 0.5-0.6")
    print("   • Untuk creative writing: gunakan temperature 0.7-0.9")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./tiny-llm-indo-qa"
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  🔥 TEMPERATURE COMPARISON TOOL                            ║
║  Test model dengan berbagai temperature settings           ║
╚════════════════════════════════════════════════════════════╝
""")
    
    model, tokenizer, device = load_model(model_path)
    compare_temperatures(model, tokenizer, device)
