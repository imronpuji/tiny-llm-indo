# Installation Guide

## Quick Setup (Docker/Linux dengan Python 3.12)

### **1. Install Python Virtual Environment Package**
```bash
# Untuk Ubuntu/Debian dengan Python 3.12
apt update
apt install -y python3.12-venv python3.12-dev

# Atau untuk Python versi apapun yang terinstall
apt install -y python3-venv python3-dev

# Untuk sistem lain, cari package python3-venv yang sesuai
```

### **2. Create Virtual Environment**
```bash
python3 -m venv venv
```

### **3. Activate Virtual Environment**
```bash
source venv/bin/activate
```

### **4. Upgrade pip**
```bash
pip install --upgrade pip
```

### **5. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **6. Prepare Q&A Dataset**
```bash
python prepare_qa_from_topics.py
```

### **7. Fine-tune Model**
```bash
python finetune_qa.py
```

---

## One-liner Setup (Copy-Paste)

### **Untuk Docker/Linux (Python 3.12)**
```bash
apt update && apt install -y python3.12-venv python3.12-dev && python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && python prepare_qa_from_topics.py && python finetune_qa.py
```

### **Untuk macOS/Linux (sudah ada venv)**
```bash
python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt && python prepare_qa_from_topics.py && python finetune_qa.py
```

---

## Step-by-Step Commands

```bash
# 1. System dependencies (Docker/Linux only)
apt update
apt install -y python3.12-venv python3.12-dev

# 2. Create venv
python3 -m venv venv

# 3. Activate
source venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install packages
pip install -r requirements.txt

# 6. Prepare dataset
python prepare_qa_from_topics.py

# 7. Train
python finetune_qa.py
```

---

## Troubleshooting

### **Error: ensurepip is not available**
```bash
# Install python3-venv package
apt install -y python3.12-venv

# Atau untuk versi Python lain
apt install -y python3-venv
```

### **Error: Package 'python3.10-venv' has no installation candidate**
Sistem Anda menggunakan Python 3.12, bukan 3.10. Install yang sesuai:
```bash
# Check Python version dulu
python3 --version

# Kalau Python 3.12
apt install -y python3.12-venv

# Kalau Python 3.11
apt install -y python3.11-venv

# Atau generic
apt install -y python3-venv
```

### **Error: CUDA Out of Memory**
Edit `finetune_qa.py`:
```python
"per_device_train_batch_size": 4,  # kurangi dari 8
"gradient_accumulation_steps": 4,  # naikkan dari 2
```

### **Error: Internet connection required**
Base model perlu download dari HuggingFace pertama kali (~500MB).
Pastikan internet tersambung.

---

## Verify Installation

```bash
# Check installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"

# Check CUDA (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Alternative: Install without venv (Not Recommended)

```bash
pip install --user torch transformers datasets tqdm accelerate huggingface-hub
python prepare_qa_from_topics.py
python finetune_qa.py
```

---

## Docker Example

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3.12-venv python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
COPY . .

# Setup and train
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["/bin/bash"]
```

Run:
```bash
docker build -t tiny-llm-indo .
docker run -it --gpus all tiny-llm-indo bash
source venv/bin/activate
python prepare_qa_from_topics.py
python finetune_qa.py
```
