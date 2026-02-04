"""
Script untuk Menambahkan Dataset Peraturan Indonesia (Azzindani/Indonesian_Regulation_QA)
=======================================================================================
Mengambil data peraturan dari Hugging Face dan memprosesnya dengan format Instruction.

Penggunaan:
    python add_regulation_qa.py
"""

import json
import os
import random
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("❌ Please install datasets: pip install datasets")
    import sys
    sys.exit(1)

OUTPUT_DIR = "./dataset"
QA_FORMAT = "instruction"

TEMPLATES = {
    "instruction": "### Instruksi:\n{q}\n\n### Jawaban:\n{a}",
}

def format_qa(qa_list, format_type="instruction"):
    """Format list of Q&A dictionaries ke dalam string template"""
    template = TEMPLATES.get(format_type, TEMPLATES["instruction"])
    formatted = []
    
    for item in qa_list:
        if not item["q"] or not item["a"]:
            continue
        text = template.format(q=item["q"], a=item["a"])
        formatted.append({"text": text})
    
    return formatted

def save_dataset(data, filename):
    """Save dataset ke file JSON"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved: {filepath} ({len(data)} samples)")
    return filepath

def repeat_data(data, min_samples=10000):
    """
    Ulangi data hingga mencapai jumlah minimum
    """
    if not data:
        return []
    if len(data) >= min_samples:
        return data
    
    repeats_needed = (min_samples // len(data)) + 1
    repeated = data * repeats_needed
    random.shuffle(repeated)
    return repeated[:min_samples]

def main():
    print("=" * 60)
    print("📥 DOWNLOADING INDONESIAN REGULATION QA")
    print("=" * 60)
    
    dataset_id = "Azzindani/Indonesian_Regulation_QA"
    print(f"Dataset: {dataset_id}")
    
    try:
        # Kita ambil split train. Jika tidak ada, biasanya otomatis terdeteksi.
        dataset = load_dataset(dataset_id, split="train")
        print(f"✓ Berhasil memuat {len(dataset)} baris data hukum.")
    except Exception as e:
        print(f"❌ Gagal memuat dataset: {e}")
        return

    # Convert ke format q & a (Dataset ini menggunakan kolom 'question' dan 'answer')
    qa_list = []
    for item in tqdm(dataset, desc="Processing Legal Data"):
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if q and a:
            # Batasi panjang jawaban agar model 150M tidak bingung
            if len(a) > 600:
                a = a[:600] + "..."
            qa_list.append({"q": q, "a": a})
            
    print(f"\n📋 Formatting with '{QA_FORMAT}' template...")
    formatted = format_qa(qa_list, QA_FORMAT)
    
    # Repetisi data (Strategy V3)
    print("\n🔁 Repeating data for deep soak strategy...")
    formatted = repeat_data(formatted, min_samples=10000)
    print(f"   Total samples now: {len(formatted)}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(formatted)
    
    # Split train/eval
    split_idx = int(len(formatted) * 0.95)
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]
    
    # Save
    print("\n💾 Saving regulation datasets...")
    save_dataset(train_data, "train_regulation_qa.json")
    save_dataset(eval_data, "eval_regulation_qa.json")
    
    print("\n" + "=" * 60)
    print("✅ DATASET REGULASI SIAP!")
    print("=" * 60)

if __name__ == "__main__":
    main()
