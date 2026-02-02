"""
Prepare Large Indonesian Dataset untuk Training
================================================
Download dan preprocess dataset besar dari HuggingFace

Dataset yang digunakan:
1. Wikipedia Indonesia (~700K artikel)
2. CC-100 Indonesian (web crawl)
3. QA Data custom (dari add_qa_data.py)

Penggunaan:
    python prepare_large_dataset.py
    python prepare_large_dataset.py --wiki-only      # Hanya Wikipedia
    python prepare_large_dataset.py --small          # Sample kecil untuk testing
"""

import os
import json
import random
import argparse
from tqdm import tqdm
from typing import List, Dict, Optional

# HuggingFace datasets
try:
    from datasets import load_dataset, Dataset, concatenate_datasets
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("❌ Install datasets: pip install datasets")

# ============================================================
# KONFIGURASI
# ============================================================

OUTPUT_DIR = "./dataset"
SEED = 42

# Dataset configs
DATASET_CONFIGS = {
    "wikipedia": {
        "name": "wikimedia/wikipedia",
        "config": "20231101.id",  # Indonesian Wikipedia
        "split": "train",
        "text_field": "text",
        "title_field": "title",
        "max_samples": None,  # None = semua
        "min_length": 200,    # Min karakter per artikel
    },
    "cc100": {
        "name": "cc100",
        "config": "id",  # Indonesian
        "split": "train",
        "text_field": "text",
        "max_samples": 500000,  # Limit karena sangat besar
        "min_length": 100,
    },
}

# Preprocessing config
PREPROCESS_CONFIG = {
    "max_length": 1024,        # Max tokens per sample
    "chunk_size": 512,         # Chunk size untuk splitting
    "overlap": 50,             # Overlap antar chunks
    "train_ratio": 0.95,       # 95% train, 5% eval
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_text(text: str) -> str:
    """Bersihkan teks dari noise"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    import re
    text = re.sub(r'\s+', ' ', text)
    
    # Remove wiki markup remnants
    text = re.sub(r'\[\[.*?\]\]', '', text)
    text = re.sub(r'\{\{.*?\}\}', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    
    # Remove references like [1], [2]
    text = re.sub(r'\[\d+\]', '', text)
    
    # Clean up
    text = text.strip()
    
    return text


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split teks panjang menjadi chunks"""
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def format_for_training(text: str, source: str = "general") -> Dict:
    """Format teks untuk training"""
    return {"text": text, "source": source}


# ============================================================
# DATASET LOADERS
# ============================================================

def load_wikipedia_id(max_samples: Optional[int] = None, 
                      min_length: int = 200) -> List[Dict]:
    """Load Wikipedia Indonesia"""
    print("\n📚 Loading Wikipedia Indonesia...")
    
    config = DATASET_CONFIGS["wikipedia"]
    
    try:
        dataset = load_dataset(
            config["name"],
            config["config"],
            split=config["split"],
            trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ Error loading Wikipedia: {e}")
        print("   Trying alternative source...")
        try:
            # Alternative: langsung dari dumps
            dataset = load_dataset(
                "wikipedia",
                language="id",
                date="20231101",
                split="train"
            )
        except:
            print("❌ Tidak bisa load Wikipedia. Skipping...")
            return []
    
    print(f"   Raw samples: {len(dataset):,}")
    
    samples = []
    for item in tqdm(dataset, desc="Processing Wikipedia"):
        text = item.get(config["text_field"], "")
        title = item.get(config.get("title_field", ""), "")
        
        # Clean
        text = clean_text(text)
        
        # Skip short articles
        if len(text) < min_length:
            continue
        
        # Add title as context
        if title:
            text = f"{title}\n\n{text}"
        
        # Chunk long texts
        chunks = chunk_text(text, PREPROCESS_CONFIG["chunk_size"], 
                          PREPROCESS_CONFIG["overlap"])
        
        for chunk in chunks:
            if len(chunk) >= min_length:
                samples.append(format_for_training(chunk, "wikipedia"))
        
        if max_samples and len(samples) >= max_samples:
            break
    
    print(f"   ✓ Processed: {len(samples):,} samples")
    return samples


def load_cc100_id(max_samples: int = 500000, 
                  min_length: int = 100) -> List[Dict]:
    """Load CC-100 Indonesian (web crawl)"""
    print("\n🌐 Loading CC-100 Indonesian...")
    
    config = DATASET_CONFIGS["cc100"]
    
    try:
        dataset = load_dataset(
            config["name"],
            config["config"],
            split=config["split"],
            streaming=True,  # Stream karena sangat besar
            trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ Error loading CC-100: {e}")
        return []
    
    samples = []
    for item in tqdm(dataset, desc="Processing CC-100", total=max_samples):
        text = item.get(config["text_field"], "")
        
        # Clean
        text = clean_text(text)
        
        # Skip short
        if len(text) < min_length:
            continue
        
        # Chunk if needed
        chunks = chunk_text(text, PREPROCESS_CONFIG["chunk_size"],
                          PREPROCESS_CONFIG["overlap"])
        
        for chunk in chunks:
            if len(chunk) >= min_length:
                samples.append(format_for_training(chunk, "cc100"))
        
        if len(samples) >= max_samples:
            break
    
    print(f"   ✓ Processed: {len(samples):,} samples")
    return samples


def load_oscar_id(max_samples: int = 300000,
                  min_length: int = 100) -> List[Dict]:
    """Load OSCAR Indonesian corpus"""
    print("\n📖 Loading OSCAR Indonesian...")
    
    try:
        dataset = load_dataset(
            "oscar-corpus/OSCAR-2301",
            language="id",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ Error loading OSCAR: {e}")
        print("   Trying alternative...")
        try:
            dataset = load_dataset(
                "oscar",
                "unshuffled_deduplicated_id",
                split="train",
                streaming=True
            )
        except:
            print("❌ Tidak bisa load OSCAR. Skipping...")
            return []
    
    samples = []
    for item in tqdm(dataset, desc="Processing OSCAR", total=max_samples):
        text = item.get("text", "")
        
        text = clean_text(text)
        
        if len(text) < min_length:
            continue
        
        chunks = chunk_text(text, PREPROCESS_CONFIG["chunk_size"],
                          PREPROCESS_CONFIG["overlap"])
        
        for chunk in chunks:
            if len(chunk) >= min_length:
                samples.append(format_for_training(chunk, "oscar"))
        
        if len(samples) >= max_samples:
            break
    
    print(f"   ✓ Processed: {len(samples):,} samples")
    return samples


def load_existing_qa_data() -> List[Dict]:
    """Load QA data yang sudah ada"""
    print("\n💬 Loading existing QA data...")
    
    qa_files = [
        "./dataset/train_qa.json",
        "./dataset/eval_qa.json"
    ]
    
    samples = []
    for filepath in qa_files:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    item["source"] = "qa"
                    samples.append(item)
            print(f"   ✓ Loaded {len(data)} from {filepath}")
    
    if not samples:
        print("   ⚠ No QA data found. Run: python add_qa_data.py")
    
    return samples


# ============================================================
# MAIN PROCESSING
# ============================================================

def prepare_dataset(
    include_wikipedia: bool = True,
    include_cc100: bool = True,
    include_oscar: bool = False,
    include_qa: bool = True,
    small_mode: bool = False
):
    """
    Prepare dataset lengkap
    
    Args:
        include_wikipedia: Include Wikipedia ID
        include_cc100: Include CC-100 ID
        include_oscar: Include OSCAR ID
        include_qa: Include QA data
        small_mode: Mode testing dengan sample kecil
    """
    
    if not HF_AVAILABLE:
        print("❌ Please install: pip install datasets")
        return
    
    print("=" * 60)
    print("📦 PREPARING LARGE INDONESIAN DATASET")
    print("=" * 60)
    
    # Limits for small mode
    if small_mode:
        wiki_limit = 10000
        cc100_limit = 10000
        oscar_limit = 10000
        print("\n⚡ SMALL MODE: Limited samples for testing")
    else:
        wiki_limit = None  # All
        cc100_limit = 500000
        oscar_limit = 300000
    
    all_samples = []
    
    # Load datasets
    if include_wikipedia:
        wiki_data = load_wikipedia_id(max_samples=wiki_limit)
        all_samples.extend(wiki_data)
    
    if include_cc100:
        cc100_data = load_cc100_id(max_samples=cc100_limit)
        all_samples.extend(cc100_data)
    
    if include_oscar:
        oscar_data = load_oscar_id(max_samples=oscar_limit)
        all_samples.extend(oscar_data)
    
    if include_qa:
        qa_data = load_existing_qa_data()
        # QA data di-duplicate untuk emphasis
        all_samples.extend(qa_data * 3)  # 3x QA data
    
    print(f"\n📊 Total samples: {len(all_samples):,}")
    
    # Shuffle
    random.seed(SEED)
    random.shuffle(all_samples)
    
    # Split train/eval
    train_ratio = PREPROCESS_CONFIG["train_ratio"]
    split_idx = int(len(all_samples) * train_ratio)
    
    train_data = all_samples[:split_idx]
    eval_data = all_samples[split_idx:]
    
    print(f"   Train: {len(train_data):,}")
    print(f"   Eval: {len(eval_data):,}")
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_path = os.path.join(OUTPUT_DIR, "train_large.json")
    eval_path = os.path.join(OUTPUT_DIR, "eval_large.json")
    
    print(f"\n💾 Saving to {OUTPUT_DIR}/...")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump([{"text": s["text"]} for s in train_data], f, ensure_ascii=False, indent=2)
    print(f"   ✓ {train_path}")
    
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump([{"text": s["text"]} for s in eval_data], f, ensure_ascii=False, indent=2)
    print(f"   ✓ {eval_path}")
    
    # Stats
    print("\n" + "=" * 60)
    print("📈 DATASET STATISTICS")
    print("=" * 60)
    
    source_counts = {}
    for s in all_samples:
        src = s.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(all_samples)
        print(f"   {src}: {count:,} ({pct:.1f}%)")
    
    # Estimate file sizes
    train_size = os.path.getsize(train_path) / (1024 * 1024)
    eval_size = os.path.getsize(eval_path) / (1024 * 1024)
    print(f"\n   Train file: {train_size:.1f} MB")
    print(f"   Eval file: {eval_size:.1f} MB")
    
    print("\n" + "=" * 60)
    print("✅ DATASET READY!")
    print("=" * 60)
    print("\nUntuk training, update train_tiny_llm.py:")
    print('   DATASET_MODE = "large"')
    print("   atau jalankan:")
    print("   python train_tiny_llm.py --dataset large")
    
    return train_path, eval_path


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare large Indonesian dataset")
    
    parser.add_argument("--wiki-only", action="store_true",
                       help="Only Wikipedia (faster)")
    parser.add_argument("--no-cc100", action="store_true",
                       help="Skip CC-100")
    parser.add_argument("--include-oscar", action="store_true",
                       help="Include OSCAR (very large)")
    parser.add_argument("--no-qa", action="store_true",
                       help="Skip QA data")
    parser.add_argument("--small", action="store_true",
                       help="Small mode for testing")
    
    args = parser.parse_args()
    
    prepare_dataset(
        include_wikipedia=True,
        include_cc100=not args.wiki_only and not args.no_cc100,
        include_oscar=args.include_oscar,
        include_qa=not args.no_qa,
        small_mode=args.small
    )


if __name__ == "__main__":
    main()
