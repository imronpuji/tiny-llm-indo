# Quick Setup: Fine-tuning dengan Model HuggingFace

## Setup Cepat (5 Menit)

### **1. Install Dependencies**
```bash
pip install torch transformers datasets tqdm
```

### **2. Clone Repository**
```bash
git clone https://github.com/imronpuji/tiny-llm-indo.git
cd tiny-llm-indo
```

### **3. Prepare Q&A Dataset**
```bash
python prepare_qa_from_topics.py
```

Output:
- ✅ `dataset/train_qa.json` (1.6 MB, 3975 samples)
- ✅ `dataset/eval_qa.json` (420 KB, 994 samples)

### **4. Fine-tuning**
```bash
python finetune_qa.py
```

Pertama kali run akan download base model dari HuggingFace:
- Model: `yasmeenimron/masa-ai-qa`
- Size: ~500 MB
- Lokasi cache: `~/.cache/huggingface/`

**Estimasi waktu:**
- Download model: 5-10 menit (tergantung internet)
- Fine-tuning: 20-30 menit (dengan GPU)
- Total: ~30-40 menit

### **5. Testing**
```bash
python test_model.py
```

---

## Base Model Info

**Model:** `yasmeenimron/masa-ai-qa`
- Source: HuggingFace Hub
- Language: Indonesian (Bahasa Indonesia)
- Type: GPT-2 based model
- Pre-trained on Indonesian data

**Keuntungan menggunakan model ini:**
- ✅ Tidak perlu training dari scratch
- ✅ Sudah pre-trained dengan data Indonesia
- ✅ Lebih cepat untuk fine-tuning
- ✅ Hasil lebih baik untuk Q&A Indonesia

---

## Konfigurasi

### **Mengubah Base Model**

Edit `finetune_qa.py`:
```python
# Opsi 1: HuggingFace model (recommended)
BASE_MODEL_PATH = "yasmeenimron/masa-ai-qa"

# Opsi 2: Model lokal (jika sudah train sendiri)
BASE_MODEL_PATH = "./tiny-llm-indo-final"

# Opsi 3: Model HuggingFace lain
BASE_MODEL_PATH = "cahya/gpt2-small-indonesian-522M"
```

### **Mengubah Training Config**

Edit `finetune_qa.py`:
```python
FINETUNE_CONFIG = {
    "num_train_epochs": 10,              # Jumlah epoch
    "per_device_train_batch_size": 8,    # Batch size
    "learning_rate": 3e-5,               # Learning rate
    "eval_steps": 200,                   # Eval setiap N steps
    # ... dll
}
```

---

## Troubleshooting

### **Error: Internet Required**
Model `yasmeenimron/masa-ai-qa` perlu didownload dari HuggingFace pertama kali.

**Solusi:**
1. Pastikan koneksi internet aktif
2. Jika di balik proxy, set env:
   ```bash
   export HTTP_PROXY=http://proxy:port
   export HTTPS_PROXY=http://proxy:port
   ```

### **Error: CUDA Out of Memory**
Batch size terlalu besar untuk GPU Anda.

**Solusi:**
```python
"per_device_train_batch_size": 4,  # dari 8
"gradient_accumulation_steps": 4,  # dari 2
```

### **Error: Model Not Found**
Cek nama model benar: `yasmeenimron/masa-ai-qa`

**Solusi:**
1. Verify di https://huggingface.co/yasmeenimron/masa-ai-qa
2. Pastikan transformers updated: `pip install -U transformers`

### **Download Lambat**
HuggingFace download ~500MB model.

**Solusi (alternative):**
1. Download manual dari HuggingFace
2. Simpan di local folder
3. Update `BASE_MODEL_PATH = "./path/to/model"`

---

## Dataset Topics Structure

Folder `dataset_topics/` berisi 42 file JSON dengan format:
```json
[
  {
    "q": "Pertanyaan",
    "a": "Jawaban",
    "cot": "Chain of thought (optional)"
  }
]
```

**Menambah topic baru:**
1. Buat file baru: `dataset_topics/my_topic.json`
2. Ikuti format di atas
3. Run ulang: `python prepare_qa_from_topics.py`
4. Fine-tune: `python finetune_qa.py`

---

## Model Output

Setelah fine-tuning, model disimpan di:
```
tiny-llm-indo-qa/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── vocab.json
└── merges.txt
```

Bisa langsung dipakai:
```python
from transformers import GPT2LMHeadModel, AutoTokenizer

model = GPT2LMHeadModel.from_pretrained("./tiny-llm-indo-qa")
tokenizer = AutoTokenizer.from_pretrained("./tiny-llm-indo-qa")

# Generate
prompt = "Apa ibu kota Indonesia?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

---

## Upload ke HuggingFace (Optional)

Setelah fine-tuning berhasil, bisa upload ke HuggingFace:

```bash
# Install huggingface CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload
cd tiny-llm-indo-qa
huggingface-cli upload your-username/model-name .
```

---

## FAQ

**Q: Berapa lama training?**
A: ~20-30 menit dengan GPU, ~2-3 jam dengan CPU

**Q: Perlu GPU?**
A: Tidak wajib, tapi GPU jauh lebih cepat. CPU bisa, tapi lama.

**Q: Bisa pakai GPU cloud?**
A: Ya! Google Colab, Kaggle, atau cloud lain yang support PyTorch.

**Q: Hasil training disimpan otomatis?**
A: Ya, di `tiny-llm-indo-qa/` dan checkpoints di `tiny-llm-indo-qa-checkpoints/`

**Q: Bisa fine-tune ulang?**
A: Ya! Set `BASE_MODEL_PATH = "./tiny-llm-indo-qa"` dan run lagi.

---

**Status:** ✅ Ready to use!
**Support:** Open issue di GitHub untuk bantuan.
