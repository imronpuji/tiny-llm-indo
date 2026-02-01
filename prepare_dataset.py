"""
Dataset Preparation untuk Tiny Indonesian LLM
==============================================
Script ini akan:
1. Download dataset bahasa Indonesia dari HuggingFace
2. Clean dan filter data
3. Buat format yang cocok untuk training
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def load_mc4_indonesian(max_samples=30000):
    """Load mC4 Indonesian dataset (publicly available)"""
    print("\n📥 Loading mC4 Indonesian...")
    try:
        dataset = load_dataset(
            "allenai/c4",
            "id",
            split="train",
            streaming=True
        )
        
        texts = []
        for i, item in enumerate(tqdm(dataset, desc="Processing mC4")):
            if i >= max_samples * 2:  # Load lebih untuk filtering
                break
            
            text = clean_text(item.get('text', ''))
            if is_good_quality(text, min_length=100):
                texts.append(text)
            
            if len(texts) >= max_samples:
                break
        
        print(f"✓ mC4: {len(texts)} samples")
        return texts
    except Exception as e:
        print(f"⚠ mC4 failed: {e}")
        return []


def load_indo_general_corpus(max_samples=20000):
    """Load Indonesian general text corpus"""
    print("\n📥 Loading Indonesian Corpus...")
    
    # Try multiple sources in order of preference
    sources = [
        ("csebuetnlp/xlsum", "indonesian", "text"),
        ("indolem/indo_story", None, "text"),
    ]
    
    for dataset_name, config, text_field in sources:
        try:
            print(f"   Trying {dataset_name}...")
            if config:
                dataset = load_dataset(dataset_name, config, split="train", streaming=True)
            else:
                dataset = load_dataset(dataset_name, split="train", streaming=True)
            
            texts = []
            for i, item in enumerate(tqdm(dataset, desc=f"Processing {dataset_name}")):
                if i >= max_samples * 2:
                    break
                
                # Try different field names
                text = item.get(text_field, item.get('article', item.get('content', '')))
                text = clean_text(text)
                
                if is_good_quality(text, min_length=80):
                    texts.append(text)
                
                if len(texts) >= max_samples:
                    break
            
            if texts:
                print(f"✓ {dataset_name}: {len(texts)} samples")
                return texts
                
        except Exception as e:
            print(f"   ⚠ {dataset_name} failed: {e}")
            continue
    
    print("⚠ All corpus sources failed")
    return []


def load_wikipedia_indonesian(max_samples=20000):
    """Load Wikipedia Indonesia"""
    print("\n📥 Loading Wikipedia Indonesian...")
    try:
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.id",
            split="train"
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
            split="train"
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
        # Salam dan sapaan
        "Halo, apa kabar? Saya baik-baik saja, terima kasih sudah bertanya.",
        "Selamat pagi! Semoga harimu menyenangkan dan penuh berkah.",
        "Selamat siang, ada yang bisa saya bantu hari ini?",
        "Selamat sore! Bagaimana aktivitasmu hari ini?",
        "Selamat malam, semoga istirahatmu nyenyak dan mimpi indah.",
        "Hai! Senang bertemu denganmu.",
        "Halo semua! Apa kabar hari ini?",
        
        # Perkenalan
        "Nama saya adalah asisten virtual. Saya di sini untuk membantu menjawab pertanyaan Anda.",
        "Perkenalkan, saya adalah model bahasa Indonesia yang sedang belajar.",
        "Saya adalah asisten AI yang bisa membantu Anda dengan berbagai pertanyaan.",
        
        # Tentang Indonesia
        "Indonesia adalah negara kepulauan terbesar di dunia yang terletak di Asia Tenggara. Indonesia memiliki lebih dari 17.000 pulau dan populasi lebih dari 270 juta jiwa.",
        "Ibu kota Indonesia saat ini adalah Jakarta, kota metropolitan terbesar di Asia Tenggara. Jakarta terletak di pantai utara Pulau Jawa.",
        "Bahasa resmi Indonesia adalah Bahasa Indonesia yang berasal dari bahasa Melayu. Selain itu, terdapat lebih dari 700 bahasa daerah yang digunakan di seluruh nusantara.",
        "Indonesia memiliki lebih dari 17.000 pulau yang tersebar dari Sabang di ujung barat hingga Merauke di ujung timur.",
        "Bendera Indonesia berwarna merah putih, yang melambangkan keberanian dan kesucian.",
        "Garuda Pancasila adalah lambang negara Indonesia yang menggambarkan kekuatan dan kejayaan bangsa.",
        
        # Makanan Indonesia
        "Nasi goreng adalah makanan khas Indonesia yang terkenal di seluruh dunia. Nasi goreng dibuat dari nasi yang digoreng dengan bumbu dan bisa ditambah telur, ayam, atau seafood.",
        "Rendang adalah masakan daging sapi dengan bumbu rempah yang berasal dari Minangkabau, Sumatera Barat. Rendang pernah dinobatkan sebagai makanan terlezat di dunia.",
        "Sate adalah makanan yang terdiri dari potongan daging yang ditusuk dan dibakar, disajikan dengan bumbu kacang atau kecap.",
        "Gado-gado adalah salad khas Indonesia yang terdiri dari sayuran rebus dengan saus kacang yang gurih.",
        "Bakso adalah makanan berupa bola daging yang disajikan dengan kuah kaldu dan mie.",
        "Soto adalah sup tradisional Indonesia dengan berbagai variasi di setiap daerah.",
        
        # Tempat wisata
        "Bali adalah pulau yang terkenal dengan keindahan alam, budaya yang unik, dan pantai-pantai yang indah.",
        "Borobudur adalah candi Buddha terbesar di dunia yang terletak di Magelang, Jawa Tengah.",
        "Gunung Bromo adalah gunung berapi aktif yang terkenal dengan pemandangan matahari terbitnya yang menakjubkan.",
        "Raja Ampat adalah kepulauan di Papua Barat yang terkenal dengan keindahan bawah lautnya.",
        "Danau Toba adalah danau vulkanik terbesar di Asia Tenggara yang terletak di Sumatera Utara.",
        "Labuan Bajo adalah pintu gerbang menuju Taman Nasional Komodo yang merupakan habitat komodo.",
        
        # Sejarah
        "Indonesia memproklamasikan kemerdekaan pada tanggal 17 Agustus 1945 oleh Soekarno dan Mohammad Hatta.",
        "Soekarno adalah presiden pertama Indonesia yang dikenal sebagai Bapak Proklamator.",
        "Pancasila adalah dasar negara Indonesia yang terdiri dari lima sila yang menjadi pedoman hidup bangsa.",
        "Sumpah Pemuda diikrarkan pada tanggal 28 Oktober 1928 sebagai tonggak persatuan bangsa Indonesia.",
        
        # Geografi
        "Gunung Semeru adalah gunung tertinggi di Pulau Jawa dengan ketinggian 3.676 meter di atas permukaan laut.",
        "Gunung Kerinci adalah gunung tertinggi di Pulau Sumatera dengan ketinggian 3.805 meter.",
        "Puncak Jaya di Papua adalah titik tertinggi di Indonesia dengan ketinggian 4.884 meter.",
        "Indonesia terletak di antara dua benua yaitu Asia dan Australia, serta dua samudra yaitu Hindia dan Pasifik.",
        "Jakarta terletak di Pulau Jawa bagian barat dan merupakan kota dengan jumlah penduduk terbanyak di Indonesia.",
        
        # Budaya
        "Batik adalah warisan budaya Indonesia yang telah diakui UNESCO sebagai Warisan Budaya Tak Benda.",
        "Wayang kulit adalah seni pertunjukan tradisional yang menggunakan boneka dari kulit kerbau.",
        "Gamelan adalah ansambel musik tradisional Indonesia yang menggunakan instrumen perkusi seperti gong dan metalofon.",
        "Tari Kecak adalah tarian tradisional Bali yang menggambarkan kisah Ramayana.",
        "Angklung adalah alat musik tradisional dari Jawa Barat yang terbuat dari bambu.",
        
        # Ucapan umum
        "Terima kasih banyak atas bantuannya!",
        "Sama-sama, senang bisa membantu Anda.",
        "Maaf, saya tidak mengerti pertanyaan Anda. Bisa diulangi?",
        "Tentu saja, saya akan dengan senang hati membantu.",
        "Baiklah, ada pertanyaan lain yang ingin ditanyakan?",
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
    
    # 3. mC4 Indonesian (diverse web text) - replaces OSCAR
    mc4_texts = load_mc4_indonesian(max_samples=15000)
    all_texts.extend(mc4_texts)
    
    # 4. Liputan6/Indo corpus (additional text) - replaces CC100
    corpus_texts = load_indo_general_corpus(max_samples=10000)
    all_texts.extend(corpus_texts)
    
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
