"""
Script untuk Menambahkan Dataset Alpaca Indonesia (cahya/alpaca-id-cleaned)
===========================================================================
Dataset Alpaca Indonesia yang sudah dibersihkan - bagus untuk instruction following.

Penggunaan:
    python add_alpaca_qa.py
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
    "instruction_with_input": "### Instruksi:\n{q}\n\n### Input:\n{input}\n\n### Jawaban:\n{a}",
}

def format_qa(qa_list, format_type="instruction"):
    """Format list of Q&A dictionaries ke dalam string template"""
    formatted = []
    
    for item in qa_list:
        if not item["q"] or not item["a"]:
            continue
        
        # Jika ada input, gunakan template dengan input
        if item.get("input"):
            template = TEMPLATES["instruction_with_input"]
            text = template.format(q=item["q"], input=item["input"], a=item["a"])
        else:
            template = TEMPLATES["instruction"]
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
    print("📥 DOWNLOADING ALPACA INDONESIA (CLEANED)")
    print("=" * 60)
    
    dataset_id = "cahya/alpaca-id-cleaned"
    print(f"Dataset: {dataset_id}")
    
    try:
        dataset = load_dataset(dataset_id, split="train")
        print(f"✓ Berhasil memuat {len(dataset)} baris data Alpaca.")
    except Exception as e:
        print(f"❌ Gagal memuat dataset: {e}")
        return

    # Convert ke format q & a
    # Alpaca format: instruction, input (optional), output
    qa_list = []
    for item in tqdm(dataset, desc="Processing Alpaca Data"):
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output = item.get("output", "").strip()
        
        if instruction and output:
            # Batasi panjang output agar model 150M tidak bingung
            if len(output) > 500:
                output = output[:500] + "..."
            
            qa_list.append({
                "q": instruction,
                "input": input_text if input_text else None,
                "a": output
            })
            
    print(f"\n📊 Total valid samples: {len(qa_list)}")
    print(f"📋 Formatting with '{QA_FORMAT}' template...")
    formatted = format_qa(qa_list, QA_FORMAT)
    
    # Untuk Alpaca yang sudah besar, kita tidak perlu terlalu banyak repetisi
    # Tapi kita tetap shuffle agar variatif
    print("\n🔀 Shuffling data...")
    random.seed(42)
    random.shuffle(formatted)
    
    # Limit jika terlalu besar (untuk model 150M, 15-20k samples sudah cukup per source)
    max_samples = 15000
    if len(formatted) > max_samples:
        print(f"⚠️ Dataset terlalu besar ({len(formatted)}), mengambil {max_samples} sampel acak...")
        formatted = formatted[:max_samples]
    
    # Split train/eval
    split_idx = int(len(formatted) * 0.95)
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]
    
    # Save
    print("\n💾 Saving Alpaca datasets...")
    save_dataset(train_data, "train_alpaca_qa.json")
    save_dataset(eval_data, "eval_alpaca_qa.json")
    
    # Preview
    print("\n📝 Sample data preview:")
    print("-" * 50)
    for item in train_data[:2]:
        preview = item["text"][:300] + "..." if len(item["text"]) > 300 else item["text"]
        print(preview)
        print("-" * 50)
    
    print("\n" + "=" * 60)
    print("✅ DATASET ALPACA INDONESIA SIAP!")
    print("=" * 60)

if __name__ == "__main__":
    main()
