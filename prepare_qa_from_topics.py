"""
Prepare Q&A Dataset dari Dataset Topics
========================================
Script ini membaca semua file JSON dari dataset_topics/
dan convert ke format training untuk fine-tuning

Penggunaan:
    python prepare_qa_from_topics.py
"""

import os
import json
import random
from pathlib import Path

# ============================================================
# KONFIGURASI
# ============================================================

SOURCE_DIR = "./dataset_topics"
OUTPUT_DIR = "./dataset"
TRAIN_SPLIT = 0.8  # 80% training, 20% eval
SEED = 42

# Format template untuk training
QA_FORMAT = "instruction"  # Pilihan: "simple", "instruction", "chat"

TEMPLATES = {
    "simple": "Pertanyaan: {q}\nJawaban: {a}",
    "instruction": "### Instruksi:\n{q}\n\n### Jawaban:\n{a}",
    "chat": "<|user|>\n{q}\n<|assistant|>\n{a}",
    "cot": "### Instruksi:\n{q}\n\n### Pemikiran:\n{cot}\n\n### Jawaban:\n{a}",
}

# ============================================================
# FUNGSI
# ============================================================

def load_all_qa_from_topics():
    """Load semua Q&A dari dataset_topics/"""
    all_qa = []
    source_path = Path(SOURCE_DIR)
    
    if not source_path.exists():
        print(f"❌ Folder {SOURCE_DIR} tidak ditemukan!")
        print()
        print("🔧 SOLUSI:")
        print("   1. Clone repository lengkap:")
        print("      git clone https://github.com/imronpuji/tiny-llm-indo.git")
        print()
        print("   2. Atau copy folder dataset_topics/ ke directory ini")
        print()
        print("   3. Verify dengan: ls dataset_topics/*.json | wc -l")
        print("      (Should output: 42)")
        print()
        return []
    
    json_files = sorted(source_path.glob("*.json"))
    
    if not json_files:
        print(f"❌ Tidak ada file JSON di {SOURCE_DIR}")
        return []
    
    print(f"📂 Membaca {len(json_files)} file JSON dari {SOURCE_DIR}")
    print("-" * 60)
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                topic_name = json_file.stem
                # Flatten nested lists if any
                flat_data = []
                for item in data:
                    if isinstance(item, dict):
                        flat_data.append(item)
                    elif isinstance(item, list):
                        # Handle nested list
                        flat_data.extend([i for i in item if isinstance(i, dict)])
                
                print(f"  ✓ {topic_name}: {len(flat_data)} QA pairs")
                all_qa.extend(flat_data)
            else:
                print(f"  ⚠️  {json_file.name}: Format tidak valid (bukan list)")
        
        except json.JSONDecodeError:
            print(f"  ❌ {json_file.name}: Error parsing JSON")
        except Exception as e:
            print(f"  ❌ {json_file.name}: {str(e)}")
    
    print("-" * 60)
    print(f"✓ Total Q&A pairs: {len(all_qa)}")
    return all_qa


def convert_qa_to_text(qa_item, template_key="instruction"):
    """Convert QA item ke format text untuk training"""
    # Validate input is a dictionary
    if not isinstance(qa_item, dict):
        return None
    
    template = TEMPLATES.get(template_key, TEMPLATES["instruction"])
    
    q = qa_item.get("q", "").strip()
    a = qa_item.get("a", "").strip()
    cot = qa_item.get("cot", "").strip()
    
    if not q or not a:
        return None
    
    # Gunakan template COT jika ada chain of thought
    if cot and template_key == "instruction":
        text = TEMPLATES["cot"].format(q=q, cot=cot, a=a)
    else:
        text = template.format(q=q, a=a)
    
    return {"text": text}


def split_train_eval(data, train_ratio=0.8, seed=42):
    """Split data menjadi training dan evaluation"""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_data = shuffled[:split_idx]
    eval_data = shuffled[split_idx:]
    
    return train_data, eval_data


def save_dataset(data, output_path):
    """Save dataset ke JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("=" * 60)
    print("📚 PREPARE Q&A DATASET FROM TOPICS")
    print("=" * 60)
    print()
    
    # Load semua Q&A
    all_qa = load_all_qa_from_topics()
    
    if not all_qa:
        print("\n❌ Tidak ada data Q&A yang ditemukan!")
        return
    
    # Convert ke format training
    print(f"\n🔄 Converting ke format training...")
    print(f"   Format: {QA_FORMAT}")
    
    converted_data = []
    skipped = 0
    
    for qa_item in all_qa:
        converted = convert_qa_to_text(qa_item, QA_FORMAT)
        if converted:
            converted_data.append(converted)
        else:
            skipped += 1
    
    print(f"   ✓ Converted: {len(converted_data)}")
    if skipped > 0:
        print(f"   ⚠️  Skipped (invalid): {skipped}")
    
    # Split train/eval
    print(f"\n✂️  Splitting train/eval ({int(TRAIN_SPLIT*100)}/{int((1-TRAIN_SPLIT)*100)})...")
    train_data, eval_data = split_train_eval(converted_data, TRAIN_SPLIT, SEED)
    
    print(f"   ✓ Train: {len(train_data)} samples")
    print(f"   ✓ Eval: {len(eval_data)} samples")
    
    # Save datasets
    train_path = os.path.join(OUTPUT_DIR, "train_qa.json")
    eval_path = os.path.join(OUTPUT_DIR, "eval_qa.json")
    
    print(f"\n💾 Saving datasets...")
    save_dataset(train_data, train_path)
    save_dataset(eval_data, eval_path)
    
    print(f"   ✓ Train saved: {train_path}")
    print(f"   ✓ Eval saved: {eval_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("=" * 60)
    
    print(f"\n📊 Summary:")
    print(f"   Total Q&A pairs: {len(converted_data)}")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Evaluation samples: {len(eval_data)}")
    print(f"   Format: {QA_FORMAT}")
    
    # Show sample
    if train_data:
        print(f"\n📝 Sample training data:")
        print("-" * 60)
        sample_text = train_data[0]["text"]
        # Truncate jika terlalu panjang
        if len(sample_text) > 300:
            sample_text = sample_text[:300] + "..."
        print(sample_text)
        print("-" * 60)
    
    print(f"\n🎯 Next step:")
    print(f"   python finetune_qa.py")
    print()


if __name__ == "__main__":
    main()
