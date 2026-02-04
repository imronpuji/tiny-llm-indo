# 🎯 Langkah-langkah Fine-tuning

## Status Saat Ini
✅ Script augmentasi siap (`augment_indonesian_qa.py`)
✅ Script fine-tuning siap (`finetune_indonesian_qa.py`)
✅ Environment sudah setup

## 🚀 Langkah Selanjutnya

### Option 1: Auto (Recommended) - Semua Step Otomatis

```bash
chmod +x run_training.sh
./run_training.sh
```

Script ini akan otomatis:
1. ✅ Download & merge 2 datasets
2. ✅ Augmentasi data (paraphrase + negative examples)
3. ✅ Fine-tune model (10 epochs)
4. ✅ Test model

---

### Option 2: Manual - Step by Step

#### **Step 1: Augmentasi Data (20-30 menit)**

```bash
python augment_indonesian_qa.py
```

Output:
- Dataset augmented akan tersimpan di `./indonesian_qa_augmented`
- Akan muncul statistik: jumlah original, paraphrased, negative examples

#### **Step 2: Fine-tuning (2-4 jam, tergantung GPU)**

**Low Memory (4-8GB VRAM):**
```bash
python finetune_indonesian_qa.py \
    --base-model yasmeenimron/masa-ai \
    --batch-size 2 \
    --epochs 5
```

**Standard (16GB+ VRAM):**
```bash
python finetune_indonesian_qa.py \
    --base-model yasmeenimron/masa-ai \
    --batch-size 8 \
    --epochs 10
```

**High Performance (24GB+ VRAM):**
```bash
python finetune_indonesian_qa.py \
    --base-model yasmeenimron/masa-ai \
    --batch-size 16 \
    --epochs 10 \
    --learning-rate 2e-5
```

#### **Step 3: Monitor Training (Buka Terminal Baru)**

```bash
tensorboard --logdir ./masa-ai-qa-indonesian/logs
```

Buka browser: `http://localhost:6006`

#### **Step 4: Test Model**

```bash
# Quick test
python finetune_indonesian_qa.py --test-only

# Interactive test
python test_masa_ai.py
```

---

## 📊 Expected Timeline

| Step | Duration | Description |
|------|----------|-------------|
| Data Download | 5-10 min | Download datasets dari HuggingFace |
| Augmentation | 10-20 min | Paraphrase + merge datasets |
| Training | 2-4 hours | Fine-tuning model (depends on GPU) |
| Evaluation | 5 min | Test model performance |

---

## 🔍 Monitoring Training

**Track Metrics:**
1. **Loss** - Harus turun bertahap (target: <1.0)
2. **Perplexity** - Semakin rendah semakin baik (target: <10)
3. **Validation Loss** - Tidak boleh naik (overfitting indicator)

**Signs of Good Training:**
- ✅ Loss turun smooth
- ✅ Val loss ikut turun
- ✅ No sudden spikes

**Signs of Problems:**
- ❌ Loss stuck (learning rate terlalu kecil)
- ❌ Val loss naik (overfitting - stop early!)
- ❌ Loss explodes (learning rate terlalu besar)

---

## 🎓 After Training

### Upload ke HuggingFace (Optional)

```bash
python upload_to_hf.py \
    --model-path ./masa-ai-qa-indonesian \
    --repo-name your-username/masa-ai-qa-indonesian \
    --token your_hf_token
```

### Test dengan Context (RAG Simulation)

```bash
python test_masa_ai.py --context
```

---

## ⚙️ Hyperparameters Cheat Sheet

**Conservative (Safe, slower learning):**
```bash
--learning-rate 1e-5 --epochs 15
```

**Standard (Balanced):**
```bash
--learning-rate 2e-5 --epochs 10
```

**Aggressive (Fast learning, risky):**
```bash
--learning-rate 5e-5 --epochs 5
```

---

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch-size 2

# Or increase gradient accumulation (edit script)
GRADIENT_ACCUMULATION_STEPS = 240
```

### Training Too Slow
```bash
# Reduce max_length (edit script)
MAX_LENGTH = 128

# Or use smaller dataset for testing
```

### Model Not Improving
```bash
# Try higher learning rate
--learning-rate 5e-5

# Or train longer
--epochs 15
```

---

## 📁 Expected Output Files

```
masa-ai-qa-indonesian/
├── config.json              # Model config
├── pytorch_model.bin        # Full model weights
├── tokenizer_config.json    # Tokenizer config
├── vocab.json              # Vocabulary
├── adapter_config.json      # LoRA config
├── adapter_model.bin        # LoRA weights (smaller)
└── logs/                   # TensorBoard logs
```

---

## 🎯 Next Steps

Setelah training selesai, Anda bisa:

1. **Test model** dengan berbagai pertanyaan
2. **Compare** dengan model sebelum fine-tuning
3. **Upload** ke HuggingFace untuk sharing
4. **Deploy** untuk production use

---

**Ready to start? Jalankan:**

```bash
chmod +x run_training.sh
./run_training.sh
```

atau manual:

```bash
python augment_indonesian_qa.py
```
