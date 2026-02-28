"""
Test Script untuk Tiny Indonesian LLM
=====================================
Support text generation dan Q&A mode
"""

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
    
    # Load tokenizer - with fallback for corrupted tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"⚠️  Warning: Could not load tokenizer from {model_path}")
        print(f"   Error: {str(e)[:100]}")
        print("   Trying fallback: loading from cahya/gpt2-small-indonesian-522M...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("cahya/gpt2-small-indonesian-522M")
            print("✓ Fallback tokenizer loaded successfully")
        except:
            print("   Trying another fallback: gpt2...")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            print("✓ Generic GPT2 tokenizer loaded")
    
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
                  temperature=0.75,
                  top_k=50,
                  top_p=0.92,
                  num_return=1):
    """Generate text dari prompt"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_return,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
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
                 max_length=150,
                 temperature=0.5,
                 use_beam_search=False):
    """
    Ajukan pertanyaan ke model
    
    Args:
        question: Pertanyaan dalam bahasa Indonesia
        qa_format: Format template ("simple", "instruction", "chat")
        temperature: Kontrol kreativitas (0.3=faktual, 0.7=kreatif)
        use_beam_search: Gunakan beam search untuk jawaban lebih konsisten
    
    Returns:
        Jawaban dari model
    """
    template = QA_TEMPLATES.get(qa_format, QA_TEMPLATES["instruction"])
    prompt = template["format"].format(question=question)
    stop_token = template["stop"]
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs.input_ids.shape[1]  # Panjang prompt dalam TOKENS
    
    # Buat attention mask
    attention_mask = torch.ones_like(inputs.input_ids)
    
    # Generation parameters bergantung mode
    gen_params = {
        "input_ids": inputs.input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": 120,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.3,
        "no_repeat_ngram_size": 3,
        "length_penalty": 1.0,
        "early_stopping": True,
    }
    
    if use_beam_search:
        # Beam search - lebih konsisten, tapi kurang natural
        gen_params.update({
            "num_beams": 4,
            "do_sample": False,
        })
    else:
        # Sampling - lebih natural
        gen_params.update({
            "do_sample": True,
            "temperature": temperature,
            "top_k": 50,
            "top_p": 0.92,
        })
    
    with torch.no_grad():
        outputs = model.generate(**gen_params)
    
    # Decode HANYA bagian jawaban (skip prompt tokens)
    answer_tokens = outputs[0][prompt_length:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    
    # Stop at next question marker if exists
    if stop_token in answer:
        answer = answer.split(stop_token)[0].strip()
    
    # Stop di kalimat pertama atau kedua
    sentences = []
    current = ""
    for char in answer:
        current += char
        if char in '.!?':
            sentences.append(current.strip())
            current = ""
            if len(sentences) >= 2:  # Max 2 kalimat
                break
    
    if sentences:
        answer = ' '.join(sentences)
    elif current.strip():
        answer = current.strip()
    
    return answer


def qa_interactive(model, tokenizer, device, qa_format="instruction"):
    """Interactive Q&A mode"""
    print("\n" + "=" * 60)
    print("🤖 MODE TANYA JAWAB")
    print(f"   Format: {qa_format}")
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
                             temperature=0.4)
        
        print(f"\n💬 Jawaban: {answer}")
        print("-" * 50)


def qa_batch_test(model, tokenizer, device, qa_format="instruction"):
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
    print("=" * 60)
    
    for question in test_questions:
        print(f"\n❓ {question}")
        print("-" * 40)
        
        answer = ask_question(model, tokenizer, question, device,
                             qa_format=qa_format,
                             temperature=0.4)
        
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
    
    # Parse arguments
    model_path = "./tiny-llm-indo-final"
    mode = None
    qa_format = "instruction"
    
    # Show usage if --help
    if "--help" in sys.argv or "-h" in sys.argv:
        print("""
═══════════════════════════════════════════════════════════
📖 USAGE: python test_model.py [model_path] [mode] [format]
═══════════════════════════════════════════════════════════

MODES:
  --qa              Q&A batch test + interactive
  --qa-interactive  Interactive Q&A only
  --qa-batch        Q&A batch test only
  (default)         Text generation mode
  
Q&A FORMATS:
  instruction       (default) ### Instruksi format
  simple            Pertanyaan: / Jawaban: format
  chat              <|user|> / <|assistant|> format

EXAMPLES:
  python test_model.py ./tiny-llm-indo-qa --qa
  python test_model.py ./tiny-llm-indo-qa --qa-interactive simple
  python test_model.py ./my-model --qa-batch instruction

GENERATION PARAMETERS (optimized):
  • Temperature: 0.4 (factual Q&A) / 0.75 (creative text)
  • Max tokens: 120 (Q&A) / 100 (text)
  • Top-p: 0.92, Top-k: 50
  • Repetition penalty: 1.15-1.3
  • No-repeat n-gram: 3
═══════════════════════════════════════════════════════════
        """)
        return
    
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg.startswith("--"):
            mode = arg
            # Check if next arg is format (not a flag or path)
            if i + 1 < len(args):
                next_arg = args[i + 1]
                if not next_arg.startswith("--") and not next_arg.startswith("./") and not next_arg.startswith("/"):
                    qa_format = next_arg
        elif arg.startswith("./") or arg.startswith("/"):
            model_path = arg
        elif not arg.startswith("-"):
            # Could be a relative path without ./
            import os
            if os.path.exists(arg):
                model_path = arg
    
    # Load model
    model, tokenizer, device = load_model(model_path)
    
    # Check mode
    if mode == "--qa":
        # Q&A batch test lalu interactive
        qa_batch_test(model, tokenizer, device, qa_format)
        qa_interactive(model, tokenizer, device, qa_format)
    elif mode == "--qa-interactive":
        # Langsung ke interactive Q&A
        qa_interactive(model, tokenizer, device, qa_format)
    elif mode == "--qa-batch":
        # Hanya batch test
        qa_batch_test(model, tokenizer, device, qa_format)
    elif mode is None:
        # Default: text generation mode
        batch_test(model, tokenizer, device)
        interactive_test(model, tokenizer, device)
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: --qa, --qa-interactive, --qa-batch")


if __name__ == "__main__":
    main()
