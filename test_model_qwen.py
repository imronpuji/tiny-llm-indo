"""
Test Script untuk Qwen2.5-1.5B Fine-tuned Model
=================================================
Support LoRA adapter dan merged model

Penggunaan:
    python test_model_qwen.py ./masa-ai-qwen-merged --qa
    python test_model_qwen.py ./masa-ai-qwen-merged --qa-interactive
    python test_model_qwen.py --help
"""

import os
import sys
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress warnings
warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI
# ============================================================

SYSTEM_PROMPT = (
    "Kamu adalah Masa AI, asisten AI berbahasa Indonesia yang cerdas dan membantu. "
    "Jawab pertanyaan dengan akurat, jelas, dan ringkas dalam bahasa Indonesia."
)

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
DEFAULT_MODEL_PATH = "./masa-ai-qwen-merged"


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_path=None):
    """Load model - supports merged model or LoRA adapter"""
    if not model_path:
        model_path = DEFAULT_MODEL_PATH
    
    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"❌ Model path '{model_path}' not found!")
        print(f"   Available options:")
        for p in ["./masa-ai-qwen-merged", "./masa-ai-qwen", "./tiny-llm-indo-qa"]:
            if os.path.exists(p):
                print(f"     ✓ {p}")
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Check if LoRA adapter
    adapter_config = os.path.join(model_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if is_lora:
        print("📦 Detected LoRA adapter, loading base model + adapter...")
        try:
            from peft import PeftModel
        except ImportError:
            print("❌ peft not installed! Run: pip install peft")
            sys.exit(1)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        print("✓ LoRA adapter merged")
    else:
        # Load merged model directly
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model.eval()
    device = next(model.parameters()).device
    print(f"✓ Model loaded on {device}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params/1e9:.2f}B")
    
    return model, tokenizer, device


# ============================================================
# GENERATION
# ============================================================

def ask_question(model, tokenizer, question, device,
                 temperature=0.4, max_new_tokens=256,
                 system_prompt=None):
    """Tanya jawab dengan chat template"""
    
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_length = inputs.input_ids.shape[1]
    
    gen_params = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
    }
    
    if temperature <= 0.1:
        # Greedy - paling konsisten
        gen_params["do_sample"] = False
    else:
        gen_params.update({
            "do_sample": True,
            "temperature": temperature,
            "top_k": 40,
            "top_p": 0.90,
        })
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_params)
    
    # Decode hanya jawaban
    answer_tokens = outputs[0][prompt_length:]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    
    return answer


def generate_text(model, tokenizer, prompt, device,
                  temperature=0.7, max_new_tokens=200):
    """Generate text bebas"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.92,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text


# ============================================================
# INTERACTIVE MODES
# ============================================================

def qa_interactive(model, tokenizer, device):
    """Interactive Q&A mode"""
    print("\n" + "=" * 60)
    print("🤖 MASA AI - MODE TANYA JAWAB")
    print("   Model: Qwen2.5-1.5B Fine-tuned")
    print("   Ketik 'quit' untuk keluar")
    print("   Ketik 'temp 0.7' untuk ubah temperature")
    print("=" * 60)
    
    temp = 0.4
    
    while True:
        question = input("\n❓ Pertanyaan: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Sampai jumpa! 👋")
            break
        
        if not question:
            continue
        
        # Change temperature
        if question.lower().startswith("temp "):
            try:
                temp = float(question.split()[1])
                print(f"🌡️  Temperature diubah ke: {temp}")
            except:
                print("❌ Format: temp 0.7")
            continue
        
        print("\n⏳ Berpikir...")
        answer = ask_question(model, tokenizer, question, device,
                             temperature=temp)
        
        print(f"\n💬 Jawaban: {answer}")
        print("-" * 50)


def qa_batch_test(model, tokenizer, device):
    """Test Q&A dengan berbagai pertanyaan"""
    
    test_questions = [
        "Halo!",
        "Siapa kamu?",
        "Apa itu Indonesia?",
        "Siapa presiden pertama Indonesia?",
        "Apa itu komputer?",
        "Apa itu mobil?",
        "Apa itu sepeda?",
        "1+1 berapa?",
        "Jelaskan tentang fotosintesis!",
        "Apa manfaat olahraga?",
        "Bagaimana cara membuat nasi goreng?",
        "Apa itu kecerdasan buatan?",
        "Mengapa langit berwarna biru?",
        "Apa perbedaan DNA dan RNA?",
    ]
    
    print("\n" + "=" * 60)
    print("🧪 Q&A BATCH TEST - Qwen2.5-1.5B")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}/{len(test_questions)}] ❓ {question}")
        print("-" * 40)
        
        answer = ask_question(model, tokenizer, question, device,
                             temperature=0.3)  # Low temp for consistency
        
        print(f"💬 {answer}")
    
    print("\n" + "=" * 60)
    print("✅ Batch test selesai!")
    print("=" * 60)


# ============================================================
# CHATBOT CLASS
# ============================================================

class MasaAI:
    """Chatbot wrapper untuk Masa AI"""
    
    def __init__(self, model_path=None):
        self.model, self.tokenizer, self.device = load_model(model_path)
        self.history = []
    
    def ask(self, question, temperature=0.4):
        """Ajukan pertanyaan"""
        answer = ask_question(
            self.model, self.tokenizer, question, self.device,
            temperature=temperature
        )
        self.history.append({"q": question, "a": answer})
        return answer
    
    def get_history(self):
        return self.history
    
    def clear_history(self):
        self.history = []


# ============================================================
# MAIN
# ============================================================

def main():
    # Parse arguments
    model_path = DEFAULT_MODEL_PATH
    mode = None
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print("""
═══════════════════════════════════════════════════════════
📖 MASA AI - Qwen2.5-1.5B Test Script
═══════════════════════════════════════════════════════════

USAGE:
  python test_model_qwen.py [model_path] [mode]

MODES:
  --qa              Batch test + interactive Q&A
  --qa-interactive  Interactive Q&A only
  --qa-batch        Batch test only
  (default)         Interactive Q&A

MODEL PATHS:
  ./masa-ai-qwen-merged   Merged model (recommended)
  ./masa-ai-qwen           LoRA adapter (needs base model)

EXAMPLES:
  python test_model_qwen.py ./masa-ai-qwen-merged --qa
  python test_model_qwen.py --qa-interactive
═══════════════════════════════════════════════════════════
        """)
        return
    
    args = sys.argv[1:]
    for arg in args:
        if arg.startswith("--"):
            mode = arg
        elif arg.startswith("./") or arg.startswith("/") or os.path.exists(arg):
            model_path = arg
    
    # Load model
    model, tokenizer, device = load_model(model_path)
    
    # Run mode
    if mode == "--qa":
        qa_batch_test(model, tokenizer, device)
        qa_interactive(model, tokenizer, device)
    elif mode == "--qa-batch":
        qa_batch_test(model, tokenizer, device)
    else:
        # Default: interactive
        qa_interactive(model, tokenizer, device)


if __name__ == "__main__":
    main()
