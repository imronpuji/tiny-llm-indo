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


# ============================================================
# Q&A / INSTRUCTION DATASET PREPARATION
# ============================================================

# Template untuk format Q&A
QA_TEMPLATES = {
    "simple": "Pertanyaan: {question}\nJawaban: {answer}",
    "instruction": "### Instruksi:\n{question}\n\n### Jawaban:\n{answer}",
    "chat": "<|user|>\n{question}\n<|assistant|>\n{answer}",
    "alpaca": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
}


def load_qa_datasets(max_samples=10000):
    """Load Q&A datasets dari berbagai sumber"""
    print("\n📥 Loading Q&A Datasets...")
    
    qa_pairs = []
    
    # 1. Try Indonesian QA datasets
    try:
        print("   Loading TydiQA Indonesian...")
        dataset = load_dataset("tydiqa", "secondary_task", split="train")
        
        for item in tqdm(dataset, desc="TydiQA"):
            if item.get('language') == 'indonesian':
                q = clean_text(item.get('question', ''))
                # Get first answer
                answers = item.get('answers', {}).get('text', [])
                if answers and q:
                    a = clean_text(answers[0])
                    if len(q) > 10 and len(a) > 5:
                        qa_pairs.append({"question": q, "answer": a})
            
            if len(qa_pairs) >= max_samples // 2:
                break
        
        print(f"   ✓ TydiQA: {len(qa_pairs)} pairs")
    except Exception as e:
        print(f"   ⚠ TydiQA failed: {e}")
    
    # 2. Try Squad-translated (if available)
    try:
        print("   Loading Squad-ID...")
        dataset = load_dataset("squad_v2", split="train")
        
        count_before = len(qa_pairs)
        for item in tqdm(dataset, desc="Squad"):
            q = clean_text(item.get('question', ''))
            answers = item.get('answers', {}).get('text', [])
            if answers and q:
                a = clean_text(answers[0])
                if len(q) > 10 and len(a) > 3:
                    qa_pairs.append({"question": q, "answer": a})
            
            if len(qa_pairs) - count_before >= max_samples // 4:
                break
        
        print(f"   ✓ Squad: {len(qa_pairs) - count_before} pairs")
    except Exception as e:
        print(f"   ⚠ Squad failed: {e}")
    
    # 3. Generate synthetic Q&A dari teks yang ada
    if len(qa_pairs) < max_samples:
        qa_pairs.extend(generate_synthetic_qa(max_samples - len(qa_pairs)))
    
    print(f"\n✓ Total Q&A pairs: {len(qa_pairs)}")
    return qa_pairs


def generate_synthetic_qa(num_samples=5000):
    """Generate synthetic Q&A pairs untuk bahasa Indonesia"""
    print("   Generating synthetic Q&A...")
    
    # Template pertanyaan dan jawaban umum
    qa_templates = [
        # Pengetahuan umum
        ("Apa itu {topic}?", "{topic} adalah {definition}."),
        ("Jelaskan tentang {topic}!", "{topic} merupakan {definition}."),
        ("Apa yang dimaksud dengan {topic}?", "{topic} adalah {definition}."),
        ("Bagaimana cara {action}?", "Cara {action} adalah {steps}."),
        ("Mengapa {reason_topic} penting?", "{reason_topic} penting karena {reason}."),
        ("Apa manfaat dari {topic}?", "Manfaat dari {topic} adalah {benefits}."),
        ("Siapa yang menemukan {discovery}?", "{discovery} ditemukan oleh {inventor}."),
        ("Kapan {event} terjadi?", "{event} terjadi pada {time}."),
        ("Di mana {location_topic} berada?", "{location_topic} berada di {location}."),
        ("Berapa {quantity_topic}?", "{quantity_topic} adalah {quantity}."),
    ]
    
    # Sample data untuk mengisi template
    topics_definitions = [
        ("komputer", "perangkat elektronik yang dapat memproses data dan informasi"),
        ("internet", "jaringan komputer global yang menghubungkan jutaan perangkat"),
        ("kecerdasan buatan", "teknologi yang memungkinkan mesin untuk belajar dan berpikir seperti manusia"),
        ("bahasa pemrograman", "bahasa formal yang digunakan untuk membuat program komputer"),
        ("database", "sistem untuk menyimpan dan mengelola data secara terstruktur"),
        ("algoritma", "serangkaian langkah terstruktur untuk menyelesaikan masalah"),
        ("machine learning", "cabang AI yang memungkinkan sistem belajar dari data"),
        ("cloud computing", "layanan komputasi yang disediakan melalui internet"),
        ("cybersecurity", "praktik melindungi sistem dan data dari serangan digital"),
        ("big data", "kumpulan data yang sangat besar dan kompleks"),
        ("Indonesia", "negara kepulauan terbesar di dunia yang terletak di Asia Tenggara"),
        ("Jakarta", "ibu kota Indonesia yang terletak di pulau Jawa"),
        ("Pancasila", "dasar negara dan ideologi bangsa Indonesia"),
        ("Bahasa Indonesia", "bahasa resmi negara Indonesia"),
        ("ekonomi", "ilmu yang mempelajari produksi, distribusi, dan konsumsi barang dan jasa"),
        ("pendidikan", "proses pembelajaran dan pengembangan pengetahuan"),
        ("kesehatan", "kondisi sejahtera dari badan, jiwa, dan sosial"),
        ("lingkungan", "segala sesuatu yang ada di sekitar makhluk hidup"),
        ("teknologi", "penerapan ilmu pengetahuan untuk memecahkan masalah"),
        ("komunikasi", "proses penyampaian informasi dari satu pihak ke pihak lain"),
    ]
    
    actions_steps = [
        ("membuat website", "dengan mempelajari HTML, CSS, dan JavaScript, lalu membuat file-file yang diperlukan"),
        ("belajar programming", "dengan memilih bahasa pemrograman, mempelajari dasar-dasarnya, dan praktik membuat program"),
        ("menulis artikel", "dengan menentukan topik, membuat outline, menulis draft, dan melakukan revisi"),
        ("memasak nasi", "dengan mencuci beras, menambahkan air secukupnya, dan memasak hingga matang"),
        ("menjaga kesehatan", "dengan makan makanan bergizi, berolahraga teratur, dan istirahat cukup"),
        ("menghemat energi", "dengan mematikan peralatan yang tidak digunakan dan menggunakan energi secara efisien"),
        ("berkomunikasi efektif", "dengan mendengarkan aktif, berbicara jelas, dan memahami perspektif lawan bicara"),
    ]
    
    reasons = [
        ("pendidikan", "pendidikan membantu mengembangkan potensi dan meningkatkan kualitas hidup"),
        ("kesehatan", "kesehatan adalah modal utama untuk menjalani kehidupan yang produktif"),
        ("teknologi", "teknologi mempermudah kehidupan dan meningkatkan efisiensi"),
        ("lingkungan", "lingkungan yang sehat mendukung kehidupan seluruh makhluk di bumi"),
        ("komunikasi", "komunikasi yang baik membantu membangun hubungan dan menghindari konflik"),
    ]
    
    qa_pairs = []
    
    # Generate dari template
    for _ in range(num_samples):
        template_idx = random.randint(0, len(qa_templates) - 1)
        q_template, a_template = qa_templates[template_idx]
        
        if "{topic}" in q_template and "{definition}" in a_template:
            topic, definition = random.choice(topics_definitions)
            q = q_template.format(topic=topic)
            a = a_template.format(topic=topic, definition=definition)
        elif "{action}" in q_template and "{steps}" in a_template:
            action, steps = random.choice(actions_steps)
            q = q_template.format(action=action)
            a = a_template.format(action=action, steps=steps)
        elif "{reason_topic}" in q_template:
            reason_topic, reason = random.choice(reasons)
            q = q_template.format(reason_topic=reason_topic)
            a = a_template.format(reason_topic=reason_topic, reason=reason)
        else:
            continue
        
        qa_pairs.append({"question": q, "answer": a})
    
    print(f"   ✓ Synthetic: {len(qa_pairs)} pairs")
    return qa_pairs


def prepare_qa_dataset(qa_format="instruction", output_suffix="qa"):
    """
    Prepare dataset dalam format Q&A/Instruction
    
    Args:
        qa_format: Format template ("simple", "instruction", "chat", "alpaca")
        output_suffix: Suffix untuk file output
    """
    print("=" * 60)
    print("🤖 PREPARING Q&A DATASET")
    print(f"   Format: {qa_format}")
    print("=" * 60)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Load Q&A pairs
    qa_pairs = load_qa_datasets(max_samples=CONFIG['train_size'])
    
    # Format sesuai template
    template = QA_TEMPLATES.get(qa_format, QA_TEMPLATES["instruction"])
    
    formatted_texts = []
    for pair in qa_pairs:
        text = template.format(
            question=pair['question'],
            answer=pair['answer'],
            instruction=pair.get('instruction', pair['question']),
            input=pair.get('input', ''),
            output=pair.get('output', pair['answer'])
        )
        formatted_texts.append(text)
    
    # Shuffle and split
    random.shuffle(formatted_texts)
    
    train_size = int(len(formatted_texts) * 0.9)
    train_texts = formatted_texts[:train_size]
    eval_texts = formatted_texts[train_size:]
    
    # Save
    train_path = os.path.join(CONFIG['output_dir'], f"train_{output_suffix}.json")
    eval_path = os.path.join(CONFIG['output_dir'], f"eval_{output_suffix}.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump([{"text": t} for t in train_texts], f, ensure_ascii=False, indent=2)
    
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump([{"text": t} for t in eval_texts], f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Q&A Dataset saved:")
    print(f"   {train_path} ({len(train_texts)} samples)")
    print(f"   {eval_path} ({len(eval_texts)} samples)")
    
    # Preview
    print("\n📝 Sample Q&A:")
    print("-" * 50)
    for text in train_texts[:2]:
        print(text)
        print("-" * 50)
    
    print("\n✅ Q&A DATASET READY!")
    
    return train_texts, eval_texts


def create_custom_qa_dataset(qa_data, output_name="custom_qa", qa_format="instruction"):
    """
    Buat dataset Q&A dari data custom
    
    Args:
        qa_data: List of dicts dengan format:
            [{"question": "...", "answer": "..."}, ...]
        output_name: Nama file output
        qa_format: Format template
    
    Contoh:
        data = [
            {"question": "Apa ibu kota Indonesia?", "answer": "Jakarta"},
            {"question": "Siapa presiden pertama?", "answer": "Soekarno"},
        ]
        create_custom_qa_dataset(data, "my_qa")
    """
    print(f"\n📝 Creating custom Q&A dataset: {output_name}")
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    template = QA_TEMPLATES.get(qa_format, QA_TEMPLATES["instruction"])
    
    formatted = []
    for item in qa_data:
        text = template.format(
            question=item['question'],
            answer=item['answer'],
            instruction=item.get('instruction', item['question']),
            input=item.get('input', ''),
            output=item.get('output', item['answer'])
        )
        formatted.append({"text": text})
    
    # Split
    random.shuffle(formatted)
    split_idx = int(len(formatted) * 0.9)
    
    train_path = os.path.join(CONFIG['output_dir'], f"train_{output_name}.json")
    eval_path = os.path.join(CONFIG['output_dir'], f"eval_{output_name}.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(formatted[:split_idx], f, ensure_ascii=False, indent=2)
    
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(formatted[split_idx:], f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved: {train_path}, {eval_path}")
    
    return formatted


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--qa":
        # Prepare Q&A dataset
        qa_format = sys.argv[2] if len(sys.argv) > 2 else "instruction"
        prepare_qa_dataset(qa_format=qa_format)
    else:
        # Default: prepare general dataset
        main()
