# Quick Guide: Fine-tuning dengan Dataset Topics

## Workflow Lengkap

### **Tahap 1: Prepare Q&A Dataset** ✅ **MULAI DARI SINI**
```bash
python prepare_qa_from_topics.py
```

**Yang dilakukan:**
- ✅ Membaca 42 file JSON dari `dataset_topics/`
- ✅ Total: 5109 Q&A pairs
- ✅ Valid: 4969 pairs (140 di-skip karena invalid)
- ✅ Format: Instruction template dengan COT support
- ✅ Split: 3975 training, 994 evaluation

**Output:**
- `dataset/train_qa.json` (1.6 MB)
- `dataset/eval_qa.json` (420 KB)

**Format Training:**
```
### Instruksi:
{pertanyaan}

### Jawaban:
{jawaban}
```

Atau dengan Chain of Thought:
```
### Instruksi:
{pertanyaan}

### Pemikiran:
{chain_of_thought}

### Jawaban:
{jawaban}
```

### **Tahap 2: Fine-tuning** ✅ **MENGGUNAKAN BASE MODEL DARI HUGGINGFACE**
```bash
python finetune_qa.py
```

**Base Model:**
- Model: `yasmeenimron/masa-ai-qa` (HuggingFace)
- Otomatis download saat pertama kali run
- Tidak perlu training base model dari scratch!

**Konfigurasi Optimal:**
- Epochs: 10 (turun dari 50)
- Batch size: 8 (naik dari 4)
- Learning rate: 3e-5 (turun dari 5e-5)
- Eval strategy: per 200 steps
- Early stopping: 3 patience

**Estimasi:**
- Dataset: ~4000 samples
- Batch effective: 16
- Steps per epoch: ~250
- Total steps: ~2500
- Waktu: 20-30 menit (GPU)

**Output:** `tiny-llm-indo-qa/`

### **Tahap 3: Testing**
```bash
python test_model.py
```

---

## Training Base Model Sendiri (Optional)

Jika ingin training base model dari scratch (tidak wajib):
```bash
python prepare_dataset.py    # Download dataset umum
python train_tiny_llm.py      # Train base model
```

Kemudian update `BASE_MODEL_PATH` di `finetune_qa.py`:
```python
BASE_MODEL_PATH = "./tiny-llm-indo-final"  # Local model
```

---

## Dataset Topics (42 Topik)

### **Bahasa & Linguistik** (5 topik)
- bahasa.json
- bahasa_huruf.json
- bahasa_indonesia_linguistik_analitis.json

### **Matematika** (4 topik)
- angka_matematika_dasar.json
- matematika.json
- matematika_menengah_soal_cerita_penalaran_bertahap.json
- matematika_tingkat_lanjut.json

### **Teknologi & Programming** (3 topik)
- programming.json
- teknologi_komputer.json
- teknologi_ai_penjelasan_mendalam.json

### **Pengetahuan Indonesia** (8 topik)
- budaya_indonesia.json
- geografi_indonesia.json
- kota_kota_indonesia.json
- sejarah_indonesia.json
- pengetahuan_umum_indonesia.json
- makanan_indonesia.json
- pancasila_kewarganegaraan.json

### **Conversational** (3 topik)
- conversational_chatbot.json
- percakapan_lebih_natural.json
- respon_emosi_user.json

### **Sains & Kesehatan** (4 topik)
- sains.json
- alam_lingkungan.json
- kesehatan.json
- kesehatan_kedokteran_penjelasan_mekanisme.json

### **Seni & Hiburan** (5 topik)
- musik_seni.json
- seni_desain_kreatif.json
- seni_desain_comprehensive.json
- film_hiburan_entertainment.json
- film_hiburan_expanded.json

### **Lainnya** (10 topik)
- ekonomi.json
- pendidikan.json
- olahraga.json
- kehidupan_sehari_hari.json
- umkm_percakapan.json
- petualangan_travel_wisata.json
- waktu_hari.json
- pertanyaan_yes_no.json
- keterbatasan_kejujuran_ai.json
- jawaban_tidak_tahu_untuk_pertanyaan_random.json

---

## Customization

### **Mengubah Format Template**
Edit [prepare_qa_from_topics.py](prepare_qa_from_topics.py):
```python
QA_FORMAT = "instruction"  # Ubah ke: "simple", "chat", atau custom
```

### **Mengubah Train/Eval Split**
```python
TRAIN_SPLIT = 0.8  # 80% training, 20% eval
```

### **Menambah Topic Baru**
1. Buat file JSON baru di `dataset_topics/`
2. Format: `[{"q": "...", "a": "...", "cot": "..."}]`
3. Jalankan ulang `prepare_qa_from_topics.py`

### **Tuning Hyperparameter**
Edit [finetune_qa.py](finetune_qa.py):
```python
FINETUNE_CONFIG = {
    "num_train_epochs": 10,        # Ubah jumlah epoch
    "learning_rate": 3e-5,         # Ubah learning rate
    "per_device_train_batch_size": 8,  # Ubah batch size
    # ... dll
}
```

---

## Troubleshooting

### **CUDA Out of Memory**
Kurangi batch size:
```python
"per_device_train_batch_size": 4,  # dari 8
"gradient_accumulation_steps": 4,  # dari 2
```

### **Skipped Invalid Entries**
Jika banyak entries di-skip, check format JSON:
- Pastikan ada field `"q"` dan `"a"`
- Pastikan tidak kosong
- Pastikan format string valid

### **Overfitting**
Training loss turun tapi eval loss naik:
1. Early stopping akan otomatis stop
2. Atau kurangi `num_train_epochs`
3. Tambah `weight_decay`

---

## Monitoring

### **Selama Fine-tuning**
```
Step 200: loss=1.234 eval_loss=1.345
Step 400: loss=1.123 eval_loss=1.289
```
✅ **Bagus**: Eval loss turun
⚠️ **Warning**: Eval loss naik 3x berturut → early stopping

### **Setelah Fine-tuning**
Check folder checkpoints:
```
tiny-llm-indo-qa-checkpoints/
├── checkpoint-200/
├── checkpoint-400/
└── checkpoint-600/
```

Model terbaik otomatis disimpan ke `tiny-llm-indo-qa/`

---

## Tips

1. **Quality over Quantity**: 
   - 5000 Q&A berkualitas > 50000 random Q&A
   - Review dan improve dataset_topics secara berkala

2. **Balanced Topics**: 
   - Pastikan semua topik penting ter-representasi
   - Tambah lebih banyak contoh untuk topik yang penting

3. **Chain of Thought**: 
   - Gunakan field `"cot"` untuk pertanyaan complex
   - Bantu model belajar reasoning step-by-step

4. **Iterative Improvement**: 
   - Test model → identify weakness → tambah data → retrain
   - Dokumentasi hasil testing untuk tracking progress

---

**Status**: ✅ Ready to fine-tune!
**Next**: `python finetune_qa.py`
