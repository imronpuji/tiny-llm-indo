# Fine-tuning Indonesian QA Model

Fine-tuning GPT-2 based model untuk Indonesian Question Answering dengan fokus pada **pemahaman** bukan hafalan.

## 📋 Dataset

**Dataset:** [ernaamaliaw/Indonesian-General-QA](https://huggingface.co/datasets/ernaamaliaw/Indonesian-General-QA)

Format:
- `Pertanyaan`: Pertanyaan dalam bahasa Indonesia
- `Jawaban`: Jawaban yang sesuai

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Option A: Training Langsung (Tanpa Augmentasi)

```bash
python finetune_indonesian_qa.py
```

### 3. Option B: Training dengan Data Augmentation (Recommended)

**Step 1: Augmentasi Dataset**
```bash
python augment_indonesian_qa.py \
    --dataset ernaamaliaw/Indonesian-General-QA \
    --output ./indonesian_qa_augmented \
    --augment-ratio 0.3
```

**Step 2: Fine-tune dengan Dataset Augmented**
```bash
# Edit finetune_indonesian_qa.py, ubah:
# DATASET_NAME = "ernaamaliaw/Indonesian-General-QA"
# menjadi:
# DATASET_NAME = "./indonesian_qa_augmented"

python finetune_indonesian_qa.py
```

## ⚙️ Hyperparameters (LAB-style)

Menggunakan konfigurasi LAB untuk hasil terbaik:

| Parameter | Value | Keterangan |
|-----------|-------|------------|
| Effective Batch Size | 960 - 3840 | 8 × 120 × num_gpus |
| Learning Rate | 2e-5 | Constant setelah warmup |
| Warmup Steps | 25 | Linear warmup |
| Scheduler | constant_with_warmup | Tidak ada decay |
| Epochs | 10 | Lebih banyak epoch |
| Weight Decay | 0.01 | Regularization |

### Custom Arguments

```bash
# Ubah base model
python finetune_indonesian_qa.py --base-model gpt2

# Ubah batch size
python finetune_indonesian_qa.py --batch-size 4

# Ubah learning rate
python finetune_indonesian_qa.py --learning-rate 1e-5

# Ubah jumlah epochs
python finetune_indonesian_qa.py --epochs 5
```

## 🧪 Testing Model

### Test setelah training:
```bash
python finetune_indonesian_qa.py --test-only
```

### Test dengan test_masa_ai.py:
```bash
python test_masa_ai.py
```

## 📊 Monitoring Training

Gunakan TensorBoard untuk monitor training:

```bash
tensorboard --logdir ./masa-ai-qa-indonesian/logs
```

Buka browser: http://localhost:6006

## 🎯 Tips untuk Hasil Terbaik

### 1. **Data Quality > Quantity**
- Pastikan jawaban singkat dan fokus
- Hindari jawaban yang terlalu panjang
- Tambahkan negative examples (model bilang "tidak tahu")

### 2. **Augmentasi Data**
- Variasi pertanyaan untuk satu topik
- Paraphrase konteks
- Balance dataset dengan negative examples

### 3. **Monitoring**
Track metrics penting:
- **Loss**: Harus turun bertahap
- **Perplexity**: Semakin rendah semakin baik
- **Validation Loss**: Tidak boleh naik (overfitting)

### 4. **Early Stopping**
Script sudah include `load_best_model_at_end=True` untuk load model terbaik.

### 5. **Resource Management**

**Low Memory?** Kurangi batch size:
```bash
python finetune_indonesian_qa.py --batch-size 4
```

**Faster Training?** Aktifkan FP16:
```python
# Otomatis aktif jika ada CUDA
fp16=torch.cuda.is_available()
```

## 📈 Expected Results

Setelah training dengan dataset Indonesian-General-QA:

**Before Fine-tuning:**
```
Q: Apa itu semester antara?
A: Saya siap membantu, tapi tidak yakin... [hallucination]
```

**After Fine-tuning:**
```
Q: Apa itu semester antara?
A: Semester antara adalah program yang diselenggarakan untuk mahasiswa aktif 
   yang ingin memperbaiki nilai mata kuliah yang pernah diambil.
```

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python finetune_indonesian_qa.py --batch-size 2

# Atau increase gradient accumulation
# Edit Config class: GRADIENT_ACCUMULATION_STEPS = 240
```

### Training terlalu lambat
```bash
# Kurangi max_length
# Edit Config class: MAX_LENGTH = 128

# Atau reduce dataset size untuk testing
```

### Model overfitting
```bash
# Kurangi epochs
python finetune_indonesian_qa.py --epochs 5

# Atau increase weight_decay
# Edit Config class: WEIGHT_DECAY = 0.05
```

## 📁 Output Structure

```
masa-ai-qa-indonesian/
├── config.json                 # Model config
├── pytorch_model.bin           # Model weights
├── tokenizer_config.json       # Tokenizer config
├── vocab.json                  # Vocabulary
├── merges.txt                  # BPE merges
├── adapter_config.json         # LoRA config (jika pakai LoRA)
├── adapter_model.bin           # LoRA weights
└── logs/                       # TensorBoard logs
    └── events.out.tfevents.*
```

## 🎓 Advanced: Custom Dataset

Jika ingin pakai dataset sendiri:

```python
from datasets import Dataset

# Format data
data = [
    {"Pertanyaan": "Apa itu AI?", "Jawaban": "AI adalah..."},
    {"Pertanyaan": "...", "Jawaban": "..."},
]

# Convert ke Dataset
dataset = Dataset.from_list(data)
dataset.save_to_disk("./my_custom_dataset")

# Train
python finetune_indonesian_qa.py
# (Edit DATASET_NAME = "./my_custom_dataset")
```

## 📚 References

- Dataset: [ernaamaliaw/Indonesian-General-QA](https://huggingface.co/datasets/ernaamaliaw/Indonesian-General-QA)
- Base Model: [yasmeenimron/masa-ai](https://huggingface.co/yasmeenimron/masa-ai)
- LAB Paper: [Learning to Align Better](https://arxiv.org/abs/xxx)

## 📝 License

MIT License - See LICENSE file for details
