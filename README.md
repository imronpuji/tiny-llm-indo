# Tiny Indonesian LLM (13M Parameters)

## Quick Start

### 1. Install Dependencies
```bash
pip install torch transformers datasets tqdm
```

### 2. Prepare Dataset
```bash
python prepare_dataset.py
```

Ini akan download dan proses data dari:
- Wikipedia Indonesia
- Indonesian News 2018
- OSCAR (Common Crawl)
- CC100

### 3. Train Base Model
```bash
python train_tiny_llm.py
```

### 4. Prepare Q&A Dataset (dari dataset_topics)
```bash
python prepare_qa_from_topics.py
```

Ini akan:
- Membaca semua file JSON dari `dataset_topics/` (42 file topik)
- Convert format `{q, a, cot}` ke format training
- Split 80% training, 20% evaluation
- Generate `dataset/train_qa.json` dan `dataset/eval_qa.json`

### 5. Fine-tune untuk Q&A
```bash
python finetune_qa.py
```

### 6. Test Model
```bash
python test_model.py
```

## File Structure
```
llm/
├── prepare_dataset.py          # Download & clean dataset umum
├── train_tiny_llm.py           # Training script dengan early stopping
├── prepare_qa_from_topics.py  # Prepare Q&A dari dataset_topics/
├── finetune_qa.py              # Fine-tune untuk Q&A
├── test_model.py               # Test trained model
├── dataset_topics/             # 42 topik Q&A untuk fine-tuning
│   ├── bahasa.json
│   ├── matematika.json
│   ├── programming.json
│   └── ... (39 topik lainnya)
├── dataset/
│   ├── train.json              # Dataset general training
│   ├── eval.json               # Dataset general evaluation
│   ├── train_qa.json           # Dataset Q&A training
│   └── eval_qa.json            # Dataset Q&A evaluation
├── tiny-llm-indo-final/        # Base model hasil training
└── tiny-llm-indo-qa/           # Fine-tuned Q&A model
```

## Model Architecture (13M params)

| Component | Value |
|----Training Bertahap**: 
   - Tahap 1: Base model dengan data general (Wikipedia, News, etc)
   - Tahap 2: Fine-tune dengan Q&A dari dataset_topics

2. **Dataset Q&A**: 
   - 42 topik berbeda di `dataset_topics/`
   - Format: `{"q": "pertanyaan", "a": "jawaban", "cot": "chain of thought"}`
   - Total ~1000+ Q&A pairs siap pakai

3. **Hindari Overfitting**: 
   - Monitor eval loss, stop jika naik
   - Early stopping sudah built-in
   - Fine-tuning: 10 epoch biasanya cukup

4. **Data Quality**: 
   - Lebih baik data sedikit tapi berkualitas
   - Review dan update dataset_topics sesuai kebutuhan

5. **Batch Size**: 
   - Sesuaikan dengan GPU memory
   - Default: batch=8, grad_accum=2 (effective batch=16)
| FFN Size | 1536 |
| Max Length | 512 |
| Vocab Size | ~32K |

## Training Tips

1. **Hindari Overfitting**: Monitor eval loss, stop jika naik
2. **Early Stopping**: Sudah built-in di script
3. **Data Quality**: Lebih baik data sedikit tapi berkualitas
4. **Batch Size**: Sesuaikan dengan GPU memory

## Expected Results

Untuk model 13M params, ekspektasi realistis:
- ✅ Bisa generate teks bahasa Indonesia
- ✅ Bisa melanjutkan kalimat sederhana
- ❌ Bukan untuk Q&A kompleks
- ❌ Bukan untuk reasoning

## Troubleshooting

### CUDA Out of Memory
Kurangi batch size di `train_tiny_llm.py`:
```python
"per_device_train_batch_size": 16,  # dari 32
```

### Dataset Download Failed
Coba satu-satu di `prepare_dataset.py` atau gunakan VPN.
# tiny-llm-indo
