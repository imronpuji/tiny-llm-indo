"""
Upload Model ke Hugging Face Hub
================================
Script untuk mempublikasikan model ke Hugging Face

Penggunaan:
    python upload_to_hf.py

Sebelum menjalankan:
    1. pip install huggingface_hub
    2. huggingface-cli login
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from huggingface_hub import HfApi, create_repo
from transformers import GPT2LMHeadModel, AutoTokenizer

# ============================================================
# KONFIGURASI - SESUAIKAN DENGAN KEBUTUHANMU
# ============================================================

# Username Hugging Face kamu
HF_USERNAME = "yasmeenimron"

# Nama model di Hugging Face
MODEL_NAME = "masa-ai"

# Path ke model lokal
MODEL_PATHS = {
    "base": "./tiny-llm-indo-final",      # Model base (pretrained)
    "qa": "./tiny-llm-indo-qa-finetuned", # Model fine-tuned untuk Q&A
}

# Model mana yang mau diupload? Pilih: "base" atau "qa"
UPLOAD_MODEL = "qa"  # Ganti ke "qa" untuk upload model fine-tuned

# ============================================================
# MAIN SCRIPT
# ============================================================

def create_model_card(model_type: str) -> str:
    """Generate model card (README.md) untuk Hugging Face"""
    
    if model_type == "base":
        return """---
language:
  - id
license: mit
tags:
  - indonesian
  - gpt2
  - causal-lm
  - text-generation
datasets:
  - wikipedia
  - cc100
library_name: transformers
pipeline_tag: text-generation
---

# Masa AI - Indonesian Language Model (150M Parameters)

Model bahasa Indonesia berbasis arsitektur GPT-2, dilatih dari nol menggunakan data Wikipedia Indonesia, CC100, dan OSCAR.

## Model Details

| Component | Value |
|-----------|-------|
| Parameters | ~150M |
| Hidden Size | 1024 |
| Layers | 12 |
| Attention Heads | 16 |
| FFN Size | 4096 |
| Max Length | 1024 |
| Vocab Size | 32K |

## Cara Penggunaan

```python
from transformers import GPT2LMHeadModel, AutoTokenizer

model = GPT2LMHeadModel.from_pretrained("yasmeenimron/masa-ai")
tokenizer = AutoTokenizer.from_pretrained("yasmeenimron/masa-ai")

prompt = "Indonesia adalah negara"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training

Model ini dilatih menggunakan:
- Wikipedia Bahasa Indonesia
- CC100 Indonesian subset
- OSCAR Indonesian subset

## Limitations

- Model ini memiliki ukuran kecil (150M params) sehingga kemampuannya terbatas
- Cocok untuk text generation sederhana, bukan untuk reasoning kompleks
- Dapat menghasilkan teks yang tidak akurat atau bias

## License

MIT License
"""
    else:  # qa model
        return """---
language:
  - id
license: mit
tags:
  - indonesian
  - gpt2
  - question-answering
  - text-generation
datasets:
  - wikipedia
library_name: transformers
pipeline_tag: text-generation
---

# Masa AI - Q&A (150M Parameters)

Model bahasa Indonesia yang di-fine-tune untuk menjawab pertanyaan sederhana.

## Cara Penggunaan

```python
from transformers import GPT2LMHeadModel, AutoTokenizer

model = GPT2LMHeadModel.from_pretrained("yasmeenimron/masa-ai-qa")
tokenizer = AutoTokenizer.from_pretrained("yasmeenimron/masa-ai-qa")

prompt = "Pertanyaan: Apa ibu kota Indonesia?\\nJawaban:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Base Model

Model ini di-fine-tune dari [yasmeenimron/masa-ai](https://huggingface.co/yasmeenimron/masa-ai)

## License

MIT License
"""


def main():
    api = HfApi()
    
    model_path = MODEL_PATHS[UPLOAD_MODEL]
    repo_name = MODEL_NAME if UPLOAD_MODEL == "base" else f"{MODEL_NAME}-qa"
    repo_id = f"{HF_USERNAME}/{repo_name}"
    
    print(f"=" * 60)
    print(f"📤 Upload Model ke Hugging Face")
    print(f"=" * 60)
    print(f"Model Path : {model_path}")
    print(f"Repo ID    : {repo_id}")
    print(f"=" * 60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model tidak ditemukan di {model_path}")
        print("   Pastikan model sudah di-train terlebih dahulu!")
        return
    
    # Load model untuk verifikasi
    print("\n🔍 Memverifikasi model...")
    try:
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   ✅ Model loaded: {param_count:,} parameters")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create repo
    print("\n📁 Membuat repository di Hugging Face...")
    try:
        create_repo(repo_id, exist_ok=True, private=False)
        print(f"   ✅ Repository siap: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"   ⚠️  {e}")
    
    # Upload model
    print("\n📤 Mengupload model...")
    try:
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print(f"   ✅ Model berhasil diupload!")
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        return
    
    # Upload model card
    print("\n📝 Membuat model card...")
    model_card = create_model_card(UPLOAD_MODEL)
    model_card_path = os.path.join(model_path, "README.md")
    with open(model_card_path, "w") as f:
        f.write(model_card)
    
    api.upload_file(
        path_or_fileobj=model_card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    print(f"   ✅ Model card berhasil diupload!")
    
    print("\n" + "=" * 60)
    print(f"🎉 SUKSES! Model tersedia di:")
    print(f"   https://huggingface.co/{repo_id}")
    print("=" * 60)
    print("\nCara pakai:")
    print(f'   from transformers import GPT2LMHeadModel, AutoTokenizer')
    print(f'   model = GPT2LMHeadModel.from_pretrained("{repo_id}")')
    print(f'   tokenizer = AutoTokenizer.from_pretrained("{repo_id}")')
    print("=" * 60)


if __name__ == "__main__":
    main()
