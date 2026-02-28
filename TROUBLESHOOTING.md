# 🚨 Quick Fix: Error Dataset Topics Tidak Ditemukan

## Error Message
```
❌ Folder ./dataset_topics tidak ditemukan!
```

## Solusi

### **Option 1: Clone Repository Lengkap (Recommended)**

```bash
# Jika belum clone, clone dulu entire repository
git clone https://github.com/imronpuji/tiny-llm-indo.git
cd tiny-llm-indo

# Kemudian run setup
apt update && apt install -y python3.12-venv python3.12-dev
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python prepare_qa_from_topics.py
python finetune_qa.py
```

### **Option 2: Copy dataset_topics dari Local**

Jika Anda sudah punya folder dataset_topics di tempat lain:

```bash
# Copy folder dataset_topics ke workspace saat ini
cp -r /path/to/dataset_topics ./

# Verify
ls -la dataset_topics/*.json | wc -l
# Should show: 42

# Kemudian run prepare
python prepare_qa_from_topics.py
```

### **Option 3: Download Dataset Topics Manual**

```bash
# Create folder
mkdir -p dataset_topics

# Download each file from GitHub
# (Ganti URL sesuai repo Anda)
cd dataset_topics

# Cara 1: Download via wget (contoh)
wget https://raw.githubusercontent.com/imronpuji/tiny-llm-indo/main/dataset_topics/bahasa.json
wget https://raw.githubusercontent.com/imronpuji/tiny-llm-indo/main/dataset_topics/matematika.json
# ... dll untuk semua 42 file

# Atau cara 2: Clone hanya folder tertentu (git sparse-checkout)
cd ..
git clone --depth 1 --filter=blob:none --sparse https://github.com/imronpuji/tiny-llm-indo.git temp
cd temp
git sparse-checkout set dataset_topics
cp -r dataset_topics ../
cd ..
rm -rf temp
```

## Verification

Setelah copy/clone, verify dataset_topics ada:

```bash
# Check folder exists
ls -la dataset_topics/

# Count JSON files (should be 42)
ls -1 dataset_topics/*.json | wc -l

# Check total QA pairs
python3 -c "
import json
import glob

total = 0
for f in sorted(glob.glob('dataset_topics/*.json')):
    with open(f) as fp:
        data = json.load(fp)
        total += len(data)
        print(f'{f}: {len(data)} pairs')
print(f'\nTotal: {total} QA pairs')
"
```

Expected output:
- 42 JSON files
- ~5000+ total QA pairs

## Kemudian Run Training

```bash
source venv/bin/activate
python prepare_qa_from_topics.py
python finetune_qa.py
```

---

## Error: Dependency Conflict

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
spacy 3.7.2 requires typer<0.10.0,>=0.3.0, but you have typer 0.24.1
```

### Solusi

**Option A: Ignore (Recommended untuk training)**
Error ini warning saja, tidak mempengaruhi fine-tuning. Lanjut saja.

**Option B: Fix Dependencies**
```bash
# Uninstall conflicting packages
pip uninstall spacy weasel -y

# Install ulang requirements
pip install -r requirements.txt
```

**Option C: Fresh Virtual Environment**
```bash
# Hapus venv lama
deactivate
rm -rf venv

# Buat baru
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Complete Command (From Scratch)

```bash
# 1. Clone repository (pastikan ada dataset_topics/)
git clone https://github.com/imronpuji/tiny-llm-indo.git
cd tiny-llm-indo

# 2. Install system deps
apt update && apt install -y python3.12-venv python3.12-dev

# 3. Setup venv
python3 -m venv venv
source venv/bin/activate

# 4. Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# 5. Verify dataset_topics exists
ls dataset_topics/*.json | wc -l  # Should be 42

# 6. Prepare dataset
python prepare_qa_from_topics.py

# 7. Fine-tune
python finetune_qa.py
```

---

## Quick Check Before Training

```bash
# Check 1: Python version
python3 --version  # Should be 3.10+

# Check 2: dataset_topics exists
[ -d "dataset_topics" ] && echo "✓ dataset_topics found" || echo "❌ dataset_topics NOT found"

# Check 3: Count files
ls -1 dataset_topics/*.json 2>/dev/null | wc -l  # Should be 42

# Check 4: Virtual env active
[ -n "$VIRTUAL_ENV" ] && echo "✓ venv active" || echo "⚠️  venv not active"

# Check 5: Dependencies installed
python -c "import transformers; print('✓ transformers installed')" 2>/dev/null || echo "❌ transformers not installed"
```

All checks should pass ✓ before running training.
