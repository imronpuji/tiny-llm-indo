# Command Cheatsheet - Quick Reference

## 🚀 QUICK START (Copy-Paste Commands)

### **Docker/Linux (Python 3.12) - Root Access**

```bash
# Full setup + training (one command)
apt update && apt install -y python3.12-venv python3.12-dev && \
python3 -m venv venv && source venv/bin/activate && \
pip install --upgrade pip && pip install -r requirements.txt && \
python prepare_qa_from_topics.py && python finetune_qa.py
```

### **Docker/Linux - Manual Steps**

```bash
# 1. Install system dependencies
apt update
apt install -y python3.12-venv python3.12-dev

# 2. Create & activate venv
python3 -m venv venv
source venv/bin/activate

# 3. Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# 4. Prepare dataset
python prepare_qa_from_topics.py

# 5. Train
python finetune_qa.py
```

### **macOS/Linux (sudah ada venv)**

```bash
# Full setup + training
python3 -m venv venv && source venv/bin/activate && \
pip install --upgrade pip && pip install -r requirements.txt && \
python prepare_qa_from_topics.py && python finetune_qa.py
```

---

## 📦 USING SETUP SCRIPTS

### **Option 1: Quick Setup (Interactive)**
```bash
# Linux/Docker (dengan sudo)
sudo bash quick_setup.sh

# macOS
bash quick_setup.sh
```

### **Option 2: Full Setup & Train (Automated)**
```bash
# Linux/Docker
sudo bash setup_and_train.sh

# macOS
bash setup_and_train.sh
```

---

## 🔧 INDIVIDUAL COMMANDS

### **1. Install System Dependencies (Linux only)**
```bash
# Check Python version
python3 --version

# Install venv package (sesuaikan dengan versi Python)
apt update
apt install -y python3.12-venv python3.12-dev  # Python 3.12
# atau
apt install -y python3.11-venv python3.11-dev  # Python 3.11
# atau
apt install -y python3-venv python3-dev         # Generic
```

### **2. Virtual Environment**
```bash
# Create
python3 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Deactivate
deactivate
```

### **3. Install Dependencies**
```bash
# Upgrade pip first
pip install --upgrade pip

# Install from requirements.txt
pip install -r requirements.txt

# Or install manually
pip install torch transformers datasets tqdm accelerate huggingface-hub
```

### **4. Prepare Dataset**
```bash
python prepare_qa_from_topics.py
```

Expected output:
- ✅ 3975 training samples
- ✅ 994 evaluation samples
- ✅ Files: `dataset/train_qa.json` & `dataset/eval_qa.json`

### **5. Fine-tune Model**
```bash
python finetune_qa.py
```

Time estimate:
- GPU: 20-30 minutes
- CPU: 2-3 hours

### **6. Test Model**
```bash
python test_model.py
```

---

## 🐍 PYTHON VERSION SPECIFIC

### **Python 3.12**
```bash
apt install -y python3.12-venv python3.12-dev
python3.12 -m venv venv
```

### **Python 3.11**
```bash
apt install -y python3.11-venv python3.11-dev
python3.11 -m venv venv
```

### **Python 3.10**
```bash
apt install -y python3.10-venv python3.10-dev
python3.10 -m venv venv
```

---

## 🔍 VERIFICATION COMMANDS

### **Check Installations**
```bash
# Inside venv
python --version
pip --version

# Check packages
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import datasets; print('Datasets:', datasets.__version__)"

# Check CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### **Check Dataset**
```bash
# Count QA pairs
python -c "import json; data=json.load(open('dataset/train_qa.json')); print(f'Training: {len(data)} samples')"
python -c "import json; data=json.load(open('dataset/eval_qa.json')); print(f'Evaluation: {len(data)} samples')"

# Check topics
ls -1 dataset_topics/*.json | wc -l
```

### **Monitor Training**
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Follow training logs
tail -f nohup.out  # if running with nohup
```

---

## ⚡ TROUBLESHOOTING QUICK FIXES

### **Error: ensurepip is not available**
```bash
apt install -y python3-venv
```

### **Error: No module named 'torch'**
```bash
source venv/bin/activate  # Aktivasi venv dulu!
pip install -r requirements.txt
```

### **Error: CUDA Out of Memory**
Edit `finetune_qa.py`:
```python
"per_device_train_batch_size": 4,  # kurangi
```

### **Slow Internet (HuggingFace download)**
```bash
# Set timeout lebih lama
export HF_HUB_DOWNLOAD_TIMEOUT=300

# Atau download manual model dulu
python test_base_model.py  # Download model
```

---

## 🔄 RE-TRAINING / UPDATE DATASET

### **Update Dataset & Retrain**
```bash
# Edit files di dataset_topics/
# Kemudian:
source venv/bin/activate
python prepare_qa_from_topics.py
python finetune_qa.py
```

### **Continue from Checkpoint**
Edit `finetune_qa.py`:
```python
BASE_MODEL_PATH = "./tiny-llm-indo-qa-checkpoints/checkpoint-xxx"
```

---

## 📝 COMMON WORKFLOWS

### **First Time Setup**
```bash
sudo apt install -y python3.12-venv python3.12-dev
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python prepare_qa_from_topics.py
python finetune_qa.py
```

### **Daily Development**
```bash
source venv/bin/activate
# Make changes to dataset_topics/
python prepare_qa_from_topics.py
python finetune_qa.py
```

### **Testing Only**
```bash
source venv/bin/activate
python test_model.py
python test_base_model.py
```

---

## 🎯 MINIMAL COMMANDS (Already Setup)

Kalau venv sudah dibuat dan dependencies sudah terinstall:

```bash
source venv/bin/activate
python prepare_qa_from_topics.py
python finetune_qa.py
```

---

## 💡 PRO TIPS

### **Background Training**
```bash
nohup python finetune_qa.py > training.log 2>&1 &
tail -f training.log
```

### **Multiple GPUs**
Edit `finetune_qa.py`:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
```

### **Save Disk Space**
```bash
# Clean old checkpoints
rm -rf tiny-llm-indo-qa-checkpoints/checkpoint-*

# Keep only final model
# (automatic by default)
```

---

**Quick Help:**
- Install issues: See [INSTALL.md](INSTALL.md)
- Training guide: See [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md)
- Quick setup: See [QUICK_SETUP.md](QUICK_SETUP.md)
