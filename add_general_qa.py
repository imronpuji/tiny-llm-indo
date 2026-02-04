"""
Script untuk Menambahkan Dataset General QA dari Hugging Face
============================================================
Mengambil data dari ernaamaliaw/Indonesian-General-QA
dan memprosesnya dengan format Instruction.

Penggunaan:
    python add_general_qa.py
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
    "simple": "Pertanyaan: {q}\nJawaban: {a}",
    "instruction": "### Instruksi:\n{q}\n\n### Jawaban:\n{a}",
    "chat": "<|user|>\n{q}\n<|assistant|>\n{a}",
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
    Agar model 150M bisa benar-benar menyerap pola ini
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
    print("📥 DOWNLOADING GENERAL QA DATASET")
    print("=" * 60)
    
    dataset_id = "ernaamaliaw/Indonesian-General-QA"
    print(f"Dataset: {dataset_id}")
    
    try:
        dataset = load_dataset(dataset_id, split="train")
        print(f"✓ Berhasil memuat {len(dataset)} baris data.")
    except Exception as e:
        print(f"❌ Gagal memuat dataset: {e}")
        return

    # Convert ke format q & a
    qa_list = []
    for item in tqdm(dataset, desc="Converting format"):
        q = item.get("Pertanyaan", "").strip()
        a = item.get("Jawaban", "").strip()
        if q and a:
            qa_list.append({"q": q, "a": a})
            
    # Variasi tambahan jika perlu (opsional)
    # Di sini kita langsung format
    print(f"\n📋 Formatting with '{QA_FORMAT}' template...")
    formatted = format_qa(qa_list, QA_FORMAT)
    
    # Repetisi data (Deep Soak Strategy)
    print("\n🔁 Repeating data for deep soak strategy...")
    formatted = repeat_data(formatted, min_samples=10000)
    print(f"   Total samples after repetition: {len(formatted)}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(formatted)
    
    # Split train/eval (95/5)
    split_idx = int(len(formatted) * 0.95)
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]
    
    # Save (Overwrite atau Append tergantung kebutuhan)
    # Untuk kasus ini kita buat file baru saja agar user bisa memilih
    print("\n💾 Saving datasets...")
    save_dataset(train_data, "train_general_qa.json")
    save_dataset(eval_data, "eval_general_qa.json")
    
    # Preview
    print("\n📝 Sample data preview:")
    print("-" * 50)
    for item in train_data[:2]:
        print(item["text"])
        print("-" * 50)
    
    print("\n" + "=" * 60)
    print("✅ DATASET GENERAL QA SIAP!")
    print("=" * 60)
    print(f"\nFile tersimpan di: {OUTPUT_DIR}/train_general_qa.json")
    print("Anda bisa menggabungkan ini dengan data lain atau langsung menggunakannya.")

if __name__ == "__main__":
    main()
