"""
Dataset Preparation untuk Tiny Indonesian LLM
==============================================
Script ini akan:
1. Download dataset bahasa Indonesia dari HuggingFace
2. Clean dan filter data
3. Buat format yang cocok untuk training
"""

import os
import json
import random
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import re

# ============================================================
# KONFIGURASI
# ============================================================

CONFIG = {
    "output_dir": "./dataset",
    "max_length": 512,  # Token per sequence
    "min_length": 50,   # Minimum karakter per teks
    "train_size": 50000,  # Jumlah sample training
    "eval_size": 2000,    # Jumlah sample evaluasi
    "seed": 42,
    "tokenizer": "cahya/gpt2-small-indonesian-522M",  # Tokenizer Indonesia
}

# ============================================================
# FUNGSI CLEANING
# ============================================================

def clean_text(text):
    """Bersihkan teks dari noise"""
    if not text or not isinstance(text, str):
        return ""
    
    # Hapus multiple whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Hapus karakter aneh
    text = re.sub(r'[^\w\s.,!?;:\-\'\"()\[\]{}@#%&*+=<>/\\|~`^$€£¥₹₩]', '', text)
    
    # Hapus URL
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Hapus email
    text = re.sub(r'\S+@\S+', '', text)
    
    # Hapus multiple punctuation
    text = re.sub(r'([.,!?;:])\1+', r'\1', text)
    
    # Hapus spasi di awal/akhir
    text = text.strip()
    
    return text


def is_good_quality(text, min_length=50):
    """Filter teks berkualitas"""
    if not text or len(text) < min_length:
        return False
    
    # Harus punya huruf
    if not re.search(r'[a-zA-Z]', text):
        return False
    
    # Tidak boleh terlalu banyak angka (>30%)
    digits = sum(c.isdigit() for c in text)
    if digits / len(text) > 0.3:
        return False
    
    # Tidak boleh terlalu banyak uppercase (>50%)
    upper = sum(c.isupper() for c in text)
    letters = sum(c.isalpha() for c in text)
    if letters > 0 and upper / letters > 0.5:
        return False
    
    # Minimal ada beberapa kata
    words = text.split()
    if len(words) < 10:
        return False
    
    return True


def is_indonesian(text):
    """Simple check apakah teks kemungkinan bahasa Indonesia"""
    # Kata-kata umum bahasa Indonesia
    indonesian_words = [
        'yang', 'dan', 'di', 'ini', 'itu', 'dengan', 'untuk', 'pada',
        'adalah', 'dari', 'dalam', 'tidak', 'akan', 'ke', 'juga',
        'atau', 'ada', 'mereka', 'telah', 'oleh', 'saya', 'kami',
        'kita', 'bisa', 'seperti', 'karena', 'sudah', 'lebih', 'banyak',
        'sangat', 'hanya', 'dapat', 'bahwa', 'setelah', 'tahun', 'orang'
    ]
    
    text_lower = text.lower()
    words = text_lower.split()
    
    # Hitung berapa kata Indonesia yang muncul
    count = sum(1 for word in indonesian_words if word in words)
    
    # Minimal 3 kata Indonesia
    return count >= 3


# ============================================================
# DOWNLOAD & PROCESS DATASETS
# ============================================================

def load_oscar_indonesian(max_samples=30000):
    """Load OSCAR dataset (Common Crawl cleaned)"""
    print("\n📥 Loading OSCAR Indonesian...")
    try:
        dataset = load_dataset(
            "oscar-corpus/OSCAR-2301",
            "id",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        texts = []
        for i, item in enumerate(tqdm(dataset, desc="Processing OSCAR")):
            if i >= max_samples * 2:  # Load lebih untuk filtering
                break
            
            text = clean_text(item.get('text', ''))
            if is_good_quality(text, min_length=100):
                texts.append(text)
            
            if len(texts) >= max_samples:
                break
        
        print(f"✓ OSCAR: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"⚠ OSCAR failed: {e}")
        return []


def load_cc100_indonesian(max_samples=20000):
    """Load CC100 dataset"""
    print("\n📥 Loading CC100 Indonesian...")
    try:
        dataset = load_dataset(
            "cc100",
            "id",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        texts = []
        for i, item in enumerate(tqdm(dataset, desc="Processing CC100")):
            if i >= max_samples * 2:
                break
            
            text = clean_text(item.get('text', ''))
            if is_good_quality(text, min_length=80):
                texts.append(text)
            
            if len(texts) >= max_samples:
                break
        
        print(f"✓ CC100: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"⚠ CC100 failed: {e}")
        return []


def load_wikipedia_indonesian(max_samples=20000):
    """Load Wikipedia Indonesia"""
    print("\n📥 Loading Wikipedia Indonesian...")
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.id",
            split="train",
            trust_remote_code=True
        )
        
        texts = []
        indices = random.sample(range(len(dataset)), min(max_samples * 2, len(dataset)))
        
        for idx in tqdm(indices, desc="Processing Wikipedia"):
            text = clean_text(dataset[idx].get('text', ''))
            if is_good_quality(text, min_length=200):
                # Ambil paragraf pertama saja untuk variasi
                paragraphs = text.split('\n\n')
                for para in paragraphs[:3]:
                    if len(para) > 100:
                        texts.append(para)
            
            if len(texts) >= max_samples:
                break
        
        print(f"✓ Wikipedia: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"⚠ Wikipedia failed: {e}")
        return []


def load_indo4b_news(max_samples=15000):
    """Load Indonesian news dataset"""
    print("\n📥 Loading Indonesian News...")
    try:
        dataset = load_dataset(
            "id_newspapers_2018",
            split="train",
            trust_remote_code=True
        )
        
        texts = []
        indices = random.sample(range(len(dataset)), min(max_samples * 2, len(dataset)))
        
        for idx in tqdm(indices, desc="Processing News"):
            item = dataset[idx]
            # Gabung title dan content
            title = item.get('title', '')
            content = item.get('content', '')
            text = f"{title}. {content}" if title else content
            text = clean_text(text)
            
            if is_good_quality(text, min_length=100):
                texts.append(text)
            
            if len(texts) >= max_samples:
                break
        
        print(f"✓ News: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"⚠ News failed: {e}")
        return []


def create_conversational_data():
    """Buat data conversational sederhana"""
    print("\n📝 Creating conversational data...")
    
    conversations = [
        # Salam
        "Halo, apa kabar? Saya baik-baik saja, terima kasih sudah bertanya.",
        "Selamat pagi! Semoga harimu menyenangkan.",
        "Selamat siang, ada yang bisa saya bantu?",
        "Selamat malam, semoga istirahatmu nyenyak.",
        
        # Perkenalan
        "Nama saya adalah asisten virtual. Saya di sini untuk membantu menjawab pertanyaan Anda.",
        "Perkenalkan, saya adalah model bahasa Indonesia yang sedang belajar.",
        
        # Q&A sederhana
        "Apa itu Indonesia? Indonesia adalah negara kepulauan terbesar di dunia yang terletak di Asia Tenggara.",
        "Apa ibu kota Indonesia? Ibu kota Indonesia saat ini adalah Jakarta, namun sedang dipindahkan ke Nusantara di Kalimantan Timur.",
        "Bahasa apa yang digunakan di Indonesia? Bahasa resmi Indonesia adalah Bahasa Indonesia, namun terdapat ratusan bahasa daerah.",
        "Berapa jumlah pulau di Indonesia? Indonesia memiliki lebih dari 17.000 pulau yang tersebar dari Sabang sampai Merauke.",
        "Apa makanan khas Indonesia? Indonesia memiliki banyak makanan khas seperti nasi goreng, rendang, sate, gado-gado, dan masih banyak lagi.",
        "Siapa presiden pertama Indonesia? Presiden pertama Indonesia adalah Ir. Soekarno yang memproklamasikan kemerdekaan pada 17 Agustus 1945.",
        
        # Informasi umum
        "Jakarta adalah ibu kota Indonesia yang terletak di pulau Jawa bagian barat.",
        "Bali adalah pulau yang terkenal dengan keindahan alam dan budayanya.",
        "Gunung Semeru adalah gunung tertinggi di pulau Jawa dengan ketinggian 3.676 meter.",
        "Pancasila adalah dasar negara Indonesia yang terdiri dari lima sila.",
        "Bahasa Indonesia berasal dari bahasa Melayu yang kemudian dikembangkan sebagai bahasa persatuan.",
    ]
    
    # Gandakan dengan variasi
    expanded = []
    for conv in conversations:
        expanded.append(conv)
        # Variasi dengan awalan berbeda
        if not conv.startswith("Apa"):
            expanded.append(f"Tahukah kamu? {conv}")
        
    print(f"✓ Conversational: {len(expanded)} samples")
    return expanded


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("🚀 INDONESIAN LLM DATASET PREPARATION")
    print("=" * 60)
    
    random.seed(CONFIG['seed'])
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Collect semua data
    all_texts = []
    
    # 1. Wikipedia (high quality)
    wiki_texts = load_wikipedia_indonesian(max_samples=20000)
    all_texts.extend(wiki_texts)
    
    # 2. News (formal Indonesian)
    news_texts = load_indo4b_news(max_samples=15000)
    all_texts.extend(news_texts)
    
    # 3. OSCAR (diverse web text)
    oscar_texts = load_oscar_indonesian(max_samples=15000)
    all_texts.extend(oscar_texts)
    
    # 4. CC100 (additional web text)
    cc100_texts = load_cc100_indonesian(max_samples=10000)
    all_texts.extend(cc100_texts)
    
    # 5. Conversational data
    conv_texts = create_conversational_data()
    # Gandakan conversational data agar model lebih ingat
    all_texts.extend(conv_texts * 50)
    
    print(f"\n📊 Total raw samples: {len(all_texts)}")
    
    # Shuffle
    random.shuffle(all_texts)
    
    # Final filtering
    print("\n🔍 Final quality filtering...")
    filtered_texts = []
    for text in tqdm(all_texts, desc="Filtering"):
        if is_good_quality(text, CONFIG['min_length']) and is_indonesian(text):
            filtered_texts.append(text)
    
    print(f"✓ After filtering: {len(filtered_texts)} samples")
    
    # Deduplicate (simple)
    print("\n🔄 Removing duplicates...")
    seen = set()
    unique_texts = []
    for text in filtered_texts:
        # Hash first 100 chars for dedup
        key = text[:100].lower()
        if key not in seen:
            seen.add(key)
            unique_texts.append(text)
    
    print(f"✓ After dedup: {len(unique_texts)} samples")
    
    # Split train/eval
    random.shuffle(unique_texts)
    
    train_size = min(CONFIG['train_size'], int(len(unique_texts) * 0.95))
    eval_size = min(CONFIG['eval_size'], len(unique_texts) - train_size)
    
    train_texts = unique_texts[:train_size]
    eval_texts = unique_texts[train_size:train_size + eval_size]
    
    print(f"\n📦 Final split:")
    print(f"   Train: {len(train_texts)} samples")
    print(f"   Eval: {len(eval_texts)} samples")
    
    # Save as JSON
    train_path = os.path.join(CONFIG['output_dir'], "train.json")
    eval_path = os.path.join(CONFIG['output_dir'], "eval.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump([{"text": t} for t in train_texts], f, ensure_ascii=False, indent=2)
    
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump([{"text": t} for t in eval_texts], f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved to:")
    print(f"   {train_path}")
    print(f"   {eval_path}")
    
    # Stats
    print("\n📈 Dataset Statistics:")
    total_chars = sum(len(t) for t in train_texts)
    total_words = sum(len(t.split()) for t in train_texts)
    avg_len = total_chars / len(train_texts)
    
    print(f"   Total characters: {total_chars:,}")
    print(f"   Total words: {total_words:,}")
    print(f"   Avg length: {avg_len:.0f} chars")
    
    # Sample preview
    print("\n📝 Sample data:")
    print("-" * 50)
    for i, text in enumerate(train_texts[:3]):
        print(f"[{i+1}] {text[:200]}...")
        print()
    
    print("=" * 60)
    print("✅ DATASET READY!")
    print("=" * 60)
    
    return train_texts, eval_texts


if __name__ == "__main__":
    main()
