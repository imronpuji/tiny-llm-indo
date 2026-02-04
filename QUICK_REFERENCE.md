# 🎯 QUICK REFERENCE - Training Full Dataset

## 🚀 One Command Training
```bash
./run_full_training.sh
```
Ini akan menjalankan SEMUA step training secara otomatis (3-6 jam).

---

## 📝 Manual Step-by-Step

### 1️⃣ Generate Dataset
```bash
python add_qa_data.py              # Base QA
python add_general_qa.py           # General QA  
python add_alpaca_qa.py            # Alpaca Indonesia
python add_preference_data.py      # DPO pairs
```

### 2️⃣ Fine-tune dengan QA
```bash
python finetune_qa.py              # 2-4 jam
```

### 3️⃣ DPO Alignment
```bash
pip install trl                    # Install jika belum
python train_dpo.py                # 30-60 menit
```

### 4️⃣ Test Model
```bash
python test_model.py ./masa-ai-dpo-aligned --qa
```

---

## 🔧 Key Changes (Sudah Dioptimasi)

| Parameter | Before | After | Why |
|-----------|--------|-------|-----|
| max_length | 256 | **512** | Full context, tidak dipotong |
| epochs | 3 | **5** | Memorization lebih baik |
| learning_rate (FT) | 2e-5 | **3e-5** | Konvergensi lebih cepat |
| batch_size | 16 | **8** | Handle context panjang |
| gradient_accum | 2 | **4** | Effective batch tetap 32 |
| gradient_checkpoint | ❌ | **✅** | Memory efficient |
| train_size | 50K | **100K** | More data |

---

## 📊 Expected Improvements

### Before
- ❌ Halusinasi parah (Jakarta → Kuala Lumpur)
- ❌ Matematika salah (10-5=8)
- ❌ Jawaban random tidak relevan

### After (Full Training + DPO)
- ✅ Fakta akurat
- ✅ Matematika benar
- ✅ Jawaban relevan & konsisten

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```python
# Di finetune_qa.py, turunkan batch:
"per_device_train_batch_size": 4,  # dari 8
"gradient_accumulation_steps": 8,  # dari 4
```

### Model masih halusinasi
```bash
# Tambah preference pairs di add_preference_data.py
# Lalu training ulang DPO:
python train_dpo.py
```

### Training terlalu lambat
```bash
# Pakai distributed training (multi-GPU):
torchrun --nproc_per_node=4 finetune_qa.py
```

---

## 📚 File Locations

| File | Purpose |
|------|---------|
| `add_qa_data.py` | Generate base QA dataset |
| `add_preference_data.py` | Generate DPO pairs (good vs bad) |
| `finetune_qa.py` | **Main fine-tuning script** |
| `train_dpo.py` | DPO alignment training |
| `run_full_training.sh` | **One-command full pipeline** |
| `TRAINING_GUIDE.md` | Detailed guide |

---

## ⚡ Performance Tips

1. **Use BF16** (RTX 3090+): Faster + stable
2. **Gradient Checkpointing**: Save VRAM
3. **Multi-GPU**: 4x faster with `torchrun`
4. **Dataset Quality**: 5K good > 50K bad
5. **DPO is Key**: Biggest improvement!

---

## 🎓 Training Time Estimates

| Step | Time (Single GPU) | Time (4x GPU) |
|------|-------------------|---------------|
| Dataset prep | 5-10 min | 5-10 min |
| SFT (5 epochs) | 2-4 hours | 30-60 min |
| DPO (3 epochs) | 30-60 min | 10-15 min |
| **Total** | **3-5 hours** | **45-90 min** |

GPU: RTX 4090/5090 32GB

---

## 📈 Metrics to Monitor

```python
# Good model:
eval_loss < 1.5
perplexity < 20
factual_accuracy > 90%

# Bad model:
eval_loss > 2.5
perplexity > 50
halusinasi++
```

---

## ✅ Checklist

- [ ] Dataset QA (5K-50K samples)
- [ ] Preference pairs (50-200 pairs)  
- [ ] Max length = 512+ tokens
- [ ] Epochs ≥ 5 dengan early stopping
- [ ] DPO training done
- [ ] Manual testing (fakta Indonesia, matematika)
- [ ] Model tidak halusinasi

---

**Remember**: Quality > Quantity! 🎯

5,000 high-quality QA pairs + DPO alignment = Model yang waras!
