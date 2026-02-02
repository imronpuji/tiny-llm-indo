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


# ============================================================
# Q&A / CHAT FUNCTIONS
# ============================================================

def ask_question(model, tokenizer, question, device, 
                 qa_format="instruction",
                 max_length=150,
                 temperature=0.7):
    """
    Ajukan pertanyaan ke model
    
    Args:
        question: Pertanyaan dalam bahasa Indonesia
        qa_format: Format template ("simple", "instruction", "chat")
    
    Returns:
        Jawaban dari model
    """
    template = QA_TEMPLATES.get(qa_format, QA_TEMPLATES["instruction"])
    prompt = template["format"].format(question=question)
    stop_token = template["stop"]
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    
    # Decode output
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer (setelah prompt)
    answer = full_text[len(prompt):].strip()
    
    # Stop at next question marker if exists
    if stop_token in answer:
        answer = answer.split(stop_token)[0].strip()
    
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
                             qa_format=qa_format)
        
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
                             qa_format=qa_format)
        
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
    
    # Load model
    model, tokenizer, device = load_model()
    
    # Check mode
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        qa_format = sys.argv[2] if len(sys.argv) > 2 else "instruction"
        
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
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: --qa, --qa-interactive, --qa-batch")
    else:
        # Default: text generation mode
        batch_test(model, tokenizer, device)
        interactive_test(model, tokenizer, device)


if __name__ == "__main__":
    main()
