# Commands untuk Setup & Training

## Option 1: Automatic (Recommended) 🚀

Jalankan script otomatis yang akan setup semuanya:

```bash
chmod +x setup_and_train.sh
./setup_and_train.sh
```

Script ini akan:
1. ✅ Create virtual environment
2. ✅ Install semua dependencies
3. ✅ Prepare dataset Q&A
4. ✅ Fine-tune model
5. ✅ Save hasil ke `tiny-llm-indo-qa/`

**Estimasi waktu:** 30-40 menit (tergantung internet & hardware)

---

## Option 2: Manual Step-by-Step 📝

### **1. Create Virtual Environment**

```bash
# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows
```

### **2. Upgrade pip**

```bash
pip install --upgrade pip
```

### **3. Install Dependencies**

```bash
# Install PyTorch (pilih sesuai sistem)
# CPU only:
pip install torch torchvision torchaudio

# CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install libraries lainnya
pip install transformers datasets tqdm accelerate
```

### **4. Verify Installation**

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### **5. Prepare Q&A Dataset**

```bash
python3 prepare_qa_from_topics.py
```

**Output:**
- `dataset/train_qa.json` (3975 samples)
- `dataset/eval_qa.json` (994 samples)

### **6. Fine-tune Model**

```bash
python3 finetune_qa.py
```

**Pertama kali akan download base model dari HuggingFace:**
- Model: `yasmeenimron/masa-ai-qa`
- Size: ~500 MB
- Cache: `~/.cache/huggingface/`

**Estimasi:**
- Download: 5-10 menit
- Training: 20-30 menit (GPU) / 2-3 jam (CPU)

### **7. Test Model**

```bash
python3 test_model.py
```

### **8. Deactivate Virtual Environment**

```bash
deactivate
```

---

## Quick Commands (Copy-Paste)

### Setup Lengkap (satu baris):
```bash
python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip && pip install torch transformers datasets tqdm accelerate && python3 prepare_qa_from_topics.py && python3 finetune_qa.py
```

### Hanya Training (jika sudah setup):
```bash
source venv/bin/activate && python3 prepare_qa_from_topics.py && python3 finetune_qa.py
```

### Re-train (jika dataset sudah ready):
```bash
source venv/bin/activate && python3 finetune_qa.py
```

---

## Troubleshooting

### **Jika venv error:**
```bash
# Install venv module dulu
sudo apt-get install python3-venv  # Ubuntu/Debian
# atau
brew install python3  # macOS
```

### **Jika pip slow:**
```bash
# Gunakan mirror Tsinghua (China)
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch transformers datasets

# Atau mirror Aliyun
pip install -i https://mirrors.aliyun.com/pypi/simple/ torch transformers datasets
```

### **Jika CUDA out of memory:**
Edit `finetune_qa.py`:
```python
"per_device_train_batch_size": 4,  # kurangi dari 8
"gradient_accumulation_steps": 4,  # naikkan dari 2
```

### **Jika download HuggingFace lambat:**
Set proxy atau gunakan mirror:
```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 finetune_qa.py
```

---

## Requirements

**Minimum:**
- Python 3.8+
- RAM: 8 GB
- Storage: 10 GB free space
- CPU: 4 cores

**Recommended:**
- Python 3.9+
- RAM: 16 GB
- Storage: 20 GB free space
- GPU: NVIDIA dengan 6 GB+ VRAM
- CUDA 11.8 atau 12.1

---

## Next Steps

Setelah training selesai:

### **1. Test Interactive**
```bash
python3 test_model.py
```

### **2. Test dengan Custom Prompt**
```python
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("./tiny-llm-indo-qa")
tokenizer = AutoTokenizer.from_pretrained("./tiny-llm-indo-qa")

prompt = "### Instruksi:\nApa ibu kota Indonesia?\n\n### Jawaban:\n"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### **3. Upload ke HuggingFace (Optional)**
```bash
pip install huggingface_hub
huggingface-cli login
cd tiny-llm-indo-qa
huggingface-cli upload your-username/model-name .
```

---

## GPU Cloud Options

Jika tidak punya GPU lokal, gunakan cloud:

### **Google Colab (Free)**
```bash
!git clone https://github.com/imronpuji/tiny-llm-indo.git
%cd tiny-llm-indo
!pip install transformers datasets tqdm accelerate
!python prepare_qa_from_topics.py
!python finetune_qa.py
```

### **Kaggle (Free)**
- Runtime: GPU T4 x2 (30 jam/minggu)
- RAM: 16 GB
- Storage: 20 GB

### **Other Options:**
- AWS SageMaker
- Google Cloud Platform
- Azure ML
- Paperspace
- Lambda Labs

---

**Status:** ✅ Ready to train!
**Support:** Open issue di GitHub
