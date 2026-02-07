# Masa AI — Tiny Indonesian LLM (150M Parameters)

Model bahasa Indonesia kecil yang dilatih dari awal, dioptimasi untuk **coherence** (jawaban nyambung) walaupun pengetahuan terbatas.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Jalankan full pipeline (dari nol)
bash run_full_training.sh

# 3. Atau jalankan manual per tahap:
python add_qa_data.py              # Siapkan dataset QA
python add_general_qa.py           # Download dataset tambahan
python add_alpaca_qa.py            # Download Alpaca Indonesia
python add_preference_data.py      # Buat DPO pairs
python prepare_large_dataset.py    # Siapkan dataset pre-training
python train_tiny_llm.py           # Pre-training dari awal
python finetune_qa.py              # SFT fine-tuning
python train_dpo.py                # DPO alignment
python test_model.py ./masa-ai-dpo-aligned --qa  # Test
```

## Architecture (150M params)

| Component | Value |
|-----------|-------|
| Hidden Size | 1024 |
| Layers | 12 |
| Attention Heads | 16 |
| FFN Size | 4096 |
| Max Context | 1024 tokens |
| Vocab Size | ~32K |
| Dropout | 0.05 |
| Tokenizer | cahya/gpt2-small-indonesian-522M |

## Training Pipeline

```
Wikipedia + CC100 ──→ Pre-Training (3 epoch) ──→ Base Model
                                                     │
QA + Alpaca + Regulation ──→ SFT (3 epoch) ──→ SFT Model
                                                     │
Preference Pairs ──→ DPO (2 epoch) ──→ Final Model
```

### Key Optimizations for Coherence

1. **GPT-2 Scaled Weight Init** — output projection scaled by `1/sqrt(2*n_layer)`
2. **Dynamic Padding** — tidak pakai `padding=max_length` (hemat compute)
3. **EOS Token Training** — model belajar kapan harus berhenti
4. **Conservative SFT** — LR 2e-5 agar tidak merusak knowledge pre-training
5. **DPO Anti-Incoherence** — preference pairs mengajarkan jawaban yang nyambung vs ngaco
6. **Minimal Post-Processing** — tidak over-filter output saat inference

## File Structure

```
├── train_tiny_llm.py          # Pre-training dari awal
├── finetune_qa.py             # SFT fine-tuning
├── train_dpo.py               # DPO alignment
├── prepare_large_dataset.py   # Dataset pre-training (Wiki + CC100)
├── add_qa_data.py             # 500+ QA pairs manual
├── add_general_qa.py          # QA dari HuggingFace
├── add_alpaca_qa.py           # Alpaca Indonesia
├── add_regulation_qa.py       # Regulasi Indonesia
├── add_preference_data.py     # DPO preference pairs
├── augment_indonesian_qa.py   # Data augmentation
├── evaluate_qa.py             # Evaluasi metrik
├── test_model.py              # Testing & interactive chat
├── test_masa_ai.py            # Test dari HuggingFace
├── upload_to_hf.py            # Upload ke HuggingFace Hub
├── run_full_training.sh       # Pipeline otomatis
├── requirements.txt           # Dependencies
└── TRAINING_GUIDE.md          # Panduan detail
```

## Testing

```bash
# Batch test Q&A
python test_model.py ./masa-ai-dpo-aligned --qa-batch

# Interactive chat
python test_model.py ./masa-ai-dpo-aligned --qa

# Text generation
python test_model.py ./masa-ai-dpo-aligned
```

## Upload ke HuggingFace

```bash
python upload_to_hf.py
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA OOM | Kurangi `per_device_train_batch_size` |
| Dataset download gagal | Pakai VPN atau `--wiki-only` |
| Model ngaco | Cek eval_loss, jangan train >3 epoch SFT |
| Jawaban repetitif | Naikkan `repetition_penalty` saat inference |
