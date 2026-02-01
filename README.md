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

### 3. Train Model
```bash
python train_tiny_llm.py
```

### 4. Test Model
```bash
python test_model.py
```

## File Structure
```
llm/
├── prepare_dataset.py  # Download & clean dataset
├── train_tiny_llm.py   # Training script dengan early stopping
├── test_model.py       # Test trained model
├── dataset/
│   ├── train.json
│   └── eval.json
└── tiny-llm-indo-final/  # Trained model
```

## Model Architecture (13M params)

| Component | Value |
|-----------|-------|
| Hidden Size | 384 |
| Layers | 6 |
| Attention Heads | 6 |
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
