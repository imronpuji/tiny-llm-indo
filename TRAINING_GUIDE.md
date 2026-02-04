# 🚀 Training Guide - Full Dataset Optimization

## 📋 Strategi Training untuk Model 150M yang Waras

Panduan ini menjelaskan cara training **full dataset tanpa dipotong** untuk menghasilkan model LLM 150M yang memberikan jawaban berkualitas tinggi.

---

## 🎯 Pipeline Training Lengkap

### **Step 1: Prepare Dataset QA (High Quality)**
```bash
# Generate semua QA data dengan variasi
python add_qa_data.py

# Tambah data general QA dari HuggingFace
python add_general_qa.py

# Tambah data Alpaca Indonesia (instruction following)
python add_alpaca_qa.py

# Tambah data regulasi Indonesia
python add_regulation_qa.py
```

**Output**: Dataset berkualitas tinggi ~10K-50K samples

---

### **Step 2: Pre-training (Opsional - Skip jika sudah ada base model)**
```bash
# Training from scratch (8-12 jam untuk 150M)
python train_tiny_llm.py

# Atau download pre-trained:
# - cahya/gpt2-small-indonesian-522M
# - flax-community/gpt2-small-indonesian
```

---

### **Step 3: Supervised Fine-Tuning (SFT)**
```bash
# Fine-tune dengan dataset QA
python finetune_qa.py
```

**Konfigurasi Optimal (sudah diupdate)**:
- ✅ **Max Length**: 512 tokens (full context)
- ✅ **Epochs**: 5 (memorization lebih baik)
- ✅ **Learning Rate**: 3e-5 (optimal untuk 150M)
- ✅ **Batch Size**: 8 per device (handle context panjang)
- ✅ **Gradient Accumulation**: 4 steps (effective batch=32)
- ✅ **Gradient Checkpointing**: Enabled (memory efficient)

**Waktu Training**: ~2-4 jam (tergantung GPU)

---

### **Step 4: Preference Alignment dengan DPO**
```bash
# 1. Generate preference pairs (good vs bad answers)
python add_preference_data.py

# 2. DPO Training untuk alignment
python train_dpo.py
```

**Apa yang dilakukan DPO?**
- ✅ Mengajarkan model membedakan jawaban **benar** vs **salah**
- ❌ Mengurangi **halusinasi** dan jawaban random
- 🎯 Meningkatkan **relevansi** dan **akurasi**

**Hasil**: Model yang jauh lebih "waras" dan konsisten!

---

## 🔧 Optimasi Training Parameters

### **Model 150M Config**
```python
MODEL_CONFIG = {
    "vocab_size": 32000,
    "n_positions": 1024,    # Context panjang
    "n_embd": 1024,         # Hidden size
    "n_layer": 12,          # Depth over width
    "n_head": 16,           # Multi-head attention
    "n_inner": 4096,        # FFN size (4x)
}
```

### **Training Config (Full Dataset)**
```python
TRAINING_CONFIG = {
    "num_train_epochs": 5,           # 5 epoch untuk memorization
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "learning_rate": 3e-4,           # Optimal untuk 150M
    "warmup_ratio": 0.05,
    "max_grad_norm": 1.0,            # Gradient clipping
    "gradient_checkpointing": True,   # Memory efficient
}
```

### **Fine-tuning Config**
```python
FINETUNE_CONFIG = {
    "num_train_epochs": 5,
    "learning_rate": 3e-5,           # Lower LR untuk fine-tune
    "max_length": 512,               # Full context (tidak dipotong!)
    "gradient_checkpointing": True,
}
```

---

## 📊 Evaluasi Kualitas Model

### **Test Model**
```bash
# Test interactive QA
python test_model.py ./masa-ai-dpo-aligned --qa

# Test dengan evaluasi metrics
python evaluate_qa.py
```

### **Metrik Penting**
1. **Perplexity**: < 20 (semakin rendah semakin baik)
2. **Eval Loss**: < 1.5
3. **Factual Accuracy**: Manual check untuk fakta Indonesia
4. **Relevance**: Jawaban harus relevan dengan pertanyaan
5. **No Hallucination**: Tidak boleh halusinasi/random

---

## 🎓 Tips Penting

### **1. Data Quality > Data Quantity**
- ✅ 5K QA berkualitas > 50K QA asal-asalan
- ✅ Pastikan fakta akurat (cek Wikipedia, sumber resmi)
- ❌ Hindari data kontradiktif atau ambigu

### **2. Gunakan Full Context (512+ tokens)**
- ✅ Max length 512-1024 untuk QA detail
- ❌ Jangan truncate di 128-256 token

### **3. Training Cukup (5+ epochs)**
- ✅ 5-10 epoch untuk memorization
- ⚠️ Monitor overfitting (eval loss naik)

### **4. DPO untuk Alignment**
- ✅ DPO jauh lebih efektif dari RLHF
- ✅ Mudah implementasi (tidak butuh reward model)
- ✅ Stable training (beta=0.1 recommended)

### **5. Learning Rate Strategy**
- Pre-training: 3e-4
- Fine-tuning: 3e-5 (10x lebih kecil)
- DPO: 5e-7 (100x lebih kecil)

---

## 🔍 Debugging Masalah Umum

### **Problem: Jawaban Random/Halusinasi**
**Solusi**:
1. Tambah data preference (add_preference_data.py)
2. Training DPO (train_dpo.py)
3. Tingkatkan epochs (5-10)

### **Problem: Jawaban Terlalu Pendek**
**Solusi**:
1. Tingkatkan max_length ke 512-1024
2. Tambah contoh QA dengan jawaban panjang
3. Adjust temperature saat inference (0.5-0.7)

### **Problem: Fakta Salah**
**Solusi**:
1. Audit dataset (cek manual)
2. Tambah ground truth di preference pairs
3. DPO training dengan fakta benar vs salah

### **Problem: Model Overfitting**
**Solusi**:
1. Early stopping (load_best_model_at_end=True)
2. Tingkatkan dropout (0.1-0.2)
3. Data augmentation (variasi pertanyaan)

---

## 📈 Expected Results

### **Before Optimization**
- ❌ "Ibu kota Indonesia adalah Kuala Lumpur"
- ❌ "10 - 5 = 8"
- ❌ Jawaban random tidak relevan

### **After Full Training + DPO**
- ✅ "Ibu kota Indonesia adalah Jakarta"
- ✅ "10 - 5 = 5"
- ✅ Jawaban akurat, relevan, dan konsisten

---

## 🚀 Quick Start

```bash
# Full pipeline (3-6 jam total)
python add_qa_data.py           # 5 menit
python add_preference_data.py   # 1 menit
python finetune_qa.py           # 2-4 jam
python train_dpo.py             # 30-60 menit

# Test hasil akhir
python test_model.py ./masa-ai-dpo-aligned --qa
```

---

## 📚 Referensi

1. **SmolLM (HuggingFace)**: https://huggingface.co/blog/smollm
2. **DPO Paper**: https://arxiv.org/abs/2305.18290
3. **TinyStories**: https://arxiv.org/abs/2305.07759
4. **Preference Tuning**: https://huggingface.co/blog/pref-tuning

---

## ✅ Checklist Training

- [ ] Dataset QA berkualitas (5K-50K samples)
- [ ] Preference pairs untuk DPO (50-200 pairs)
- [ ] Max length 512+ tokens
- [ ] Training 5+ epochs dengan early stopping
- [ ] DPO alignment
- [ ] Evaluasi manual (faktual accuracy)
- [ ] Test dengan berbagai pertanyaan edge case

---

**Happy Training! 🎉**

Jika ada masalah, cek error di log dan sesuaikan hyperparameters.
