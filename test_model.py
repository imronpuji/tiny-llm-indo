"""
Test Script untuk Tiny Indonesian LLM
=====================================
Support text generation dan Q&A mode
"""

import os
# Force single GPU (device 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

# PEFT support untuk LoRA models
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# ============================================================
# Q&A TEMPLATES (harus sama dengan training)
# ============================================================

QA_TEMPLATES = {
    "simple": {
        "format": "Pertanyaan: {question}\nJawaban:",
        "stop": "\nPertanyaan:",
    },
    "instruction": {
        "format": "### Instruksi:\n{question}\n\n### Jawaban:",
        "stop": "\n### Instruksi:",
    },
    "chat": {
        "format": "<|user|>\n{question}\n<|assistant|>",
        "stop": "<|user|>",
    },
}


# ============================================================
# HALLUCINATION DETECTION
# ============================================================

def detect_hallucination(answer: str, question: str) -> bool:
    """
    Deteksi jawaban yang kemungkinan hallucination
    
    Returns:
        True jika dicurigai hallucination
    """
    answer_lower = answer.lower()
    
    # Pattern hallucination umum
    hallucination_patterns = [
        # Mixing unrelated concepts
        ("komputer", "cantik"),
        ("komputer", "besar"),
        ("teknologi", "pintar"),
        ("olahraga", "cantik"),
        # Random numbers in weird context
        ("17. 000", "september"),
        ("081", "pulau"),
        # Weird combinations
        ("minum", "olahraga", "daging"),
        ("padi", "banteng", "bintang"),
    ]
    
    for pattern in hallucination_patterns:
        if all(word in answer_lower for word in pattern):
            return True
    
    # Terlalu banyak topik berbeda dalam 1 jawaban
    topic_keywords = {
        "tech": ["komputer", "teknologi", "software", "internet"],
        "food": ["makanan", "minum", "makan", "kuliner"],
        "geo": ["pulau", "kota", "provinsi", "negara"],
        "politics": ["presiden", "partai", "politik", "pemilu"],
    }
    
    topic_count = sum(1 for keywords in topic_keywords.values() 
                     if any(k in answer_lower for k in keywords))
    
    if topic_count >= 3:  # Lebih dari 3 topik = suspicious
        return True
    
    # Jawaban terlalu panjang untuk pertanyaan sederhana
    if len(question.split()) <= 3 and len(answer.split()) > 50:
        return True
    
    return False


def load_model(model_path="./tiny-llm-indo-final", use_lora=True, base_model_path=None):
    """
    Load trained model
    
    Args:
        model_path: Path ke model atau LoRA adapter
        use_lora: Apakah model menggunakan LoRA
        base_model_path: Path ke base model (jika LoRA terpisah)
    """
    import os
    
    # Pastikan path tidak kosong
    if not model_path:
        model_path = "./tiny-llm-indo-final"
    
    print(f"Loading model from {model_path}...")
    
    # Check if path exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model path '{model_path}' not found!")
        print("   Pastikan training sudah selesai dan model tersimpan.")
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Check if this is a PEFT/LoRA model
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_peft_model = os.path.exists(adapter_config_path)
    
    if is_peft_model and PEFT_AVAILABLE:
        print("📦 Detected PEFT/LoRA model, loading with adapter...")
        
        # For PEFT model, we need to load the base model first
        # Check if there's a base model config
        config_path = os.path.join(model_path, "config.json")
        
        if os.path.exists(config_path):
            # Load config to create base model
            from transformers import GPT2Config
            import json
            
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            
            base_model_name = adapter_config.get("base_model_name_or_path", "")
            
            if base_model_name and os.path.exists(base_model_name):
                # Load from local base model
                base_model = GPT2LMHeadModel.from_pretrained(base_model_name)
            else:
                # Create base model from config in the saved path
                config = GPT2Config.from_pretrained(model_path)
                base_model = GPT2LMHeadModel(config)
            
            # Load PEFT adapter
            model = PeftModel.from_pretrained(base_model, model_path)
            print("✓ LoRA adapter loaded")
        else:
            # Fallback: try loading directly
            model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        # Standard model loading
        model = GPT2LMHeadModel.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    return model, tokenizer, device


def generate_text(model, tokenizer, prompt, device, 
                  max_length=150, 
                  temperature=0.7,
                  top_k=40,
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
            repetition_penalty=1.3,
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


# ============================================================
# Q&A / CHAT FUNCTIONS
# ============================================================

def ask_question(model, tokenizer, question, device, 
                 qa_format="instruction",
                 max_length=100,
                 temperature=0.3,
                 top_p=0.85,
                 do_sample=True):
    """
    Ajukan pertanyaan ke model — OPTIMIZED FOR COHERENCE
    
    Strategi:
    - Gunakan temperature rendah (0.3) agar jawaban fokus
    - top_p 0.85 untuk memfilter token noise  
    - repetition_penalty 1.3 (tidak terlalu tinggi agar tidak merusak output)
    - Minimal post-processing agar jawaban tetap natural
    """
    template = QA_TEMPLATES.get(qa_format, QA_TEMPLATES["instruction"])
    prompt = template["format"].format(question=question)
    stop_token = template["stop"]
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs.input_ids.shape[1]
    
    attention_mask = torch.ones_like(inputs.input_ids)
    
    curr_do_sample = do_sample if temperature > 0 else False
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,
            min_new_tokens=5,
            do_sample=curr_do_sample,
            temperature=temperature if curr_do_sample else None,
            top_p=top_p if curr_do_sample else None,
            top_k=40,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,         # Cukup untuk anti-repetisi tanpa merusak coherence
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    
    # Decode hanya bagian jawaban
    answer_tokens = outputs[0][prompt_length:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    
    # === MINIMAL CLEANING — jangan over-process ===
    import re
    
    # 1. Stop di marker format berikutnya
    if stop_token in answer:
        answer = answer.split(stop_token)[0].strip()
    
    for marker in ["### Instruksi:", "Pertanyaan:", "<|user|>", "<|assistant|>"]:
        if marker in answer:
            answer = answer.split(marker)[0].strip()
    
    # 2. Bersihkan whitespace berlebihan
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    # 3. Remove trailing incomplete thought
    if answer and not answer[-1] in '.!?':
        # Cari kalimat terakhir yang lengkap
        last_period = max(answer.rfind('.'), answer.rfind('!'), answer.rfind('?'))
        if last_period > len(answer) * 0.3:  # Hanya potong jika >30% teks sudah lengkap
            answer = answer[:last_period + 1]
        else:
            answer += "."
    
    # 4. Fallback jika kosong
    if not answer or len(answer.strip()) < 3:
        answer = "Maaf, saya tidak yakin dengan jawaban untuk pertanyaan ini."
    
    return answer


def qa_interactive(model, tokenizer, device, qa_format="instruction", temperature=0.3, top_p=0.85):
    """Interactive Q&A mode dengan hallucination detection"""
    print("\n" + "=" * 60)
    print("🤖 MODE TANYA JAWAB")
    print(f"   Format: {qa_format}")
    print(f"   Config: temp={temperature}, top_p={top_p}")
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
        answer = ask_question(model, tokenizer, question, device, 
                             qa_format=qa_format,
                             temperature=temperature,
                             top_p=top_p)
        
        # Check hallucination
        if detect_hallucination(answer, question):
            print(f"\n💬 Jawaban: {answer}")
            print("⚠️  [Warning: Jawaban mungkin kurang akurat]")
        else:
            print(f"\n💬 Jawaban: {answer}")
        
        print("-" * 50)


def qa_batch_test(model, tokenizer, device, qa_format="instruction", temperature=0.3, top_p=0.85):
    """Test Q&A dengan berbagai pertanyaan"""
    
    test_questions = [
        "Apa itu komputer?",
        "Di mana Jakarta berada?",
        "Jelaskan tentang Indonesia!",
        "Apa itu kecerdasan buatan?",
        "Bagaimana cara belajar programming?",
        "Siapa presiden pertama Indonesia?",
        "Mengapa pendidikan penting?",
        "Apa manfaat dari teknologi?",
    ]
    
    print("\n" + "=" * 60)
    print("🧪 Q&A BATCH TEST")
    print(f"   Format: {qa_format}")
    print(f"   Config: temp={temperature}, top_p={top_p}")
    print("=" * 60)
    
    for question in test_questions:
        print(f"\n❓ {question}")
        print("-" * 40)
        
        answer = ask_question(model, tokenizer, question, device,
                             qa_format=qa_format,
                             temperature=temperature,
                             top_p=top_p)
        
        print(f"💬 {answer}")


class ChatBot:
    """Simple chatbot wrapper"""
    
    def __init__(self, model_path="./tiny-llm-indo-final", qa_format="instruction"):
        self.model, self.tokenizer, self.device = load_model(model_path)
        self.qa_format = qa_format
        self.history = []
    
    def ask(self, question):
        """Ajukan pertanyaan"""
        answer = ask_question(
            self.model, self.tokenizer, question, self.device,
            qa_format=self.qa_format
        )
        self.history.append({"question": question, "answer": answer})
        return answer
    
    def get_history(self):
        """Lihat riwayat percakapan"""
        return self.history
    
    def clear_history(self):
        """Hapus riwayat"""
        self.history = []


def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Tiny Indonesian LLM')
    parser.add_argument('model_path', type=str, nargs='?', default="./tiny-llm-indo-final", help='Path to model')
    parser.add_argument('--qa', action='store_true', help='Q&A mode (batch + interactive)')
    parser.add_argument('--qa-interactive', action='store_true', help='Interactive Q&A mode')
    parser.add_argument('--qa-batch', action='store_true', help='Batch Q&A test')
    parser.add_argument('--format', type=str, default="instruction", choices=["simple", "instruction", "chat"], help='QA Template format')
    parser.add_argument('--temperature', type=float, default=0.3, help='Generation temperature')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    
    args = parser.parse_args()
    
    model_path = args.model_path
    qa_format = args.format
    temp = args.temperature
    top_p = args.top_p
    
    # Load model
    model, tokenizer, device = load_model(model_path)
    
    # Check mode
    if args.qa:
        # Q&A batch test lalu interactive
        qa_batch_test(model, tokenizer, device, qa_format, temp, top_p)
        qa_interactive(model, tokenizer, device, qa_format, temp, top_p)
    elif args.qa_interactive:
        # Langsung ke interactive Q&A
        qa_interactive(model, tokenizer, device, qa_format, temp, top_p)
    elif args.qa_batch:
        # Hanya batch test
        qa_batch_test(model, tokenizer, device, qa_format, temp, top_p)
    else:
        # Default: text generation mode
        batch_test(model, tokenizer, device)
        interactive_test(model, tokenizer, device)


if __name__ == "__main__":
    main()
