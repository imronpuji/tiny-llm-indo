"""
Script untuk Download dan Prepare Dataset Berita Indonesia
===========================================================
Fine-tuning dengan artikel berita untuk bahasa natural

Penggunaan:
    python add_news_data.py
"""

import json
import os
import random
import re
from typing import List, Dict

OUTPUT_DIR = "./dataset"

# ============================================================
# KONFIGURASI
# ============================================================

# Panjang minimal dan maksimal artikel (dalam karakter)
MIN_LENGTH = 200
MAX_LENGTH = 2000

# Jumlah sampel maksimal per dataset
MAX_SAMPLES_PER_DATASET = 10000

# Total sampel yang diinginkan
TARGET_SAMPLES = 50000


def clean_text(text: str) -> str:
    """Bersihkan teks dari HTML, karakter aneh, dll"""
    if not text:
        return ""
    
    # Hapus HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Hapus JavaScript/CSS artifacts
    text = re.sub(r'\{[^}]+\}', '', text)
    text = re.sub(r'function\s*\([^)]*\)\s*\{[^}]*\}', '', text)
    
    # Hapus URL
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Hapus email
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Hapus karakter khusus berlebihan
    text = re.sub(r'[^\w\s.,!?;:\-\'\"()\[\]/@#%&*+=]', ' ', text)
    
    # Hapus multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Hapus leading/trailing whitespace
    text = text.strip()
    
    return text


def is_valid_article(text: str) -> bool:
    """Cek apakah artikel valid untuk training"""
    if not text:
        return False
    
    # Panjang minimal
    if len(text) < MIN_LENGTH:
        return False
    
    # Harus mengandung kata Indonesia yang umum
    indonesian_words = ['yang', 'dan', 'di', 'ini', 'itu', 'dengan', 'untuk', 'pada', 'dari', 'adalah']
    text_lower = text.lower()
    word_count = sum(1 for word in indonesian_words if word in text_lower)
    
    if word_count < 3:
        return False
    
    # Tidak terlalu banyak karakter non-alfabet
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
    if alpha_ratio < 0.7:
        return False
    
    return True


def truncate_article(text: str, max_length: int = MAX_LENGTH) -> str:
    """Potong artikel ke panjang maksimal dengan tetap menjaga kalimat utuh"""
    if len(text) <= max_length:
        return text
    
    # Cari titik terakhir sebelum max_length
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    
    if last_period > max_length * 0.5:  # Minimal 50% dari max_length
        return truncated[:last_period + 1]
    
    return truncated


def download_cc100_indonesian(max_samples: int = MAX_SAMPLES_PER_DATASET) -> List[str]:
    """Download CC-100 Indonesian dataset"""
    print("\n📥 Downloading CC-100 Indonesian...")
    
    try:
        from datasets import load_dataset
        
        # CC-100 Indonesian
        dataset = load_dataset("cc100", "id", split="train", streaming=True, trust_remote_code=True)
        
        articles = []
        for i, item in enumerate(dataset):
            if i >= max_samples * 2:  # Ambil lebih banyak karena akan difilter
                break
            
            text = clean_text(item.get("text", ""))
            
            if is_valid_article(text):
                text = truncate_article(text)
                articles.append(text)
                
                if len(articles) >= max_samples:
                    break
            
            if i % 1000 == 0:
                print(f"   Processed {i} items, valid: {len(articles)}")
        
        print(f"   ✓ CC-100: {len(articles)} articles")
        return articles
        
    except Exception as e:
        print(f"   ⚠ CC-100 failed: {e}")
        return []


def download_mc4_indonesian(max_samples: int = MAX_SAMPLES_PER_DATASET) -> List[str]:
    """Download mC4 Indonesian dataset"""
    print("\n📥 Downloading mC4 Indonesian...")
    
    try:
        from datasets import load_dataset
        
        # mC4 Indonesian
        dataset = load_dataset("mc4", "id", split="train", streaming=True, trust_remote_code=True)
        
        articles = []
        for i, item in enumerate(dataset):
            if i >= max_samples * 2:
                break
            
            text = clean_text(item.get("text", ""))
            
            if is_valid_article(text):
                text = truncate_article(text)
                articles.append(text)
                
                if len(articles) >= max_samples:
                    break
            
            if i % 1000 == 0:
                print(f"   Processed {i} items, valid: {len(articles)}")
        
        print(f"   ✓ mC4: {len(articles)} articles")
        return articles
        
    except Exception as e:
        print(f"   ⚠ mC4 failed: {e}")
        return []


def download_wikipedia_indonesian(max_samples: int = MAX_SAMPLES_PER_DATASET) -> List[str]:
    """Download Wikipedia Indonesian dataset"""
    print("\n📥 Downloading Wikipedia Indonesian...")
    
    try:
        from datasets import load_dataset
        
        # Wikipedia Indonesian
        dataset = load_dataset("wikipedia", "20220301.id", split="train", trust_remote_code=True)
        
        articles = []
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        for i, idx in enumerate(indices[:max_samples * 2]):
            item = dataset[idx]
            text = clean_text(item.get("text", ""))
            
            if is_valid_article(text):
                text = truncate_article(text)
                articles.append(text)
                
                if len(articles) >= max_samples:
                    break
            
            if i % 1000 == 0:
                print(f"   Processed {i} items, valid: {len(articles)}")
        
        print(f"   ✓ Wikipedia: {len(articles)} articles")
        return articles
        
    except Exception as e:
        print(f"   ⚠ Wikipedia failed: {e}")
        return []


def download_oscar_indonesian(max_samples: int = MAX_SAMPLES_PER_DATASET) -> List[str]:
    """Download OSCAR Indonesian dataset"""
    print("\n📥 Downloading OSCAR Indonesian...")
    
    try:
        from datasets import load_dataset
        
        # OSCAR Indonesian
        dataset = load_dataset("oscar", "unshuffled_deduplicated_id", split="train", streaming=True, trust_remote_code=True)
        
        articles = []
        for i, item in enumerate(dataset):
            if i >= max_samples * 2:
                break
            
            text = clean_text(item.get("text", ""))
            
            if is_valid_article(text):
                text = truncate_article(text)
                articles.append(text)
                
                if len(articles) >= max_samples:
                    break
            
            if i % 1000 == 0:
                print(f"   Processed {i} items, valid: {len(articles)}")
        
        print(f"   ✓ OSCAR: {len(articles)} articles")
        return articles
        
    except Exception as e:
        print(f"   ⚠ OSCAR failed: {e}")
        return []


def download_indonlu(max_samples: int = 5000) -> List[str]:
    """Download IndoNLU datasets"""
    print("\n📥 Downloading IndoNLU datasets...")
    
    articles = []
    
    try:
        from datasets import load_dataset
        
        # IndoNLU - Liputan6
        try:
            dataset = load_dataset("indonlp/indonlu", "liputan6", split="train", trust_remote_code=True)
            for item in dataset:
                text = clean_text(item.get("article", "") or item.get("text", ""))
                if is_valid_article(text):
                    articles.append(truncate_article(text))
                    if len(articles) >= max_samples:
                        break
            print(f"   ✓ Liputan6: {len(articles)} articles")
        except Exception as e:
            print(f"   ⚠ Liputan6 failed: {e}")
        
    except Exception as e:
        print(f"   ⚠ IndoNLU failed: {e}")
    
    return articles


def download_id_newspapers(max_samples: int = MAX_SAMPLES_PER_DATASET) -> List[str]:
    """Download Indonesian newspapers dataset"""
    print("\n📥 Downloading Indonesian Newspapers...")
    
    try:
        from datasets import load_dataset
        
        articles = []
        
        # Coba beberapa dataset berita Indonesia
        news_datasets = [
            ("id_newspapers_2018", None),
            ("indonesian_news", None),
        ]
        
        for ds_name, config in news_datasets:
            try:
                if config:
                    dataset = load_dataset(ds_name, config, split="train", trust_remote_code=True)
                else:
                    dataset = load_dataset(ds_name, split="train", trust_remote_code=True)
                
                for item in dataset:
                    text = clean_text(item.get("content", "") or item.get("text", "") or item.get("article", ""))
                    if is_valid_article(text):
                        articles.append(truncate_article(text))
                        if len(articles) >= max_samples:
                            break
                
                print(f"   ✓ {ds_name}: {len(articles)} articles")
                
            except Exception as e:
                print(f"   ⚠ {ds_name} failed: {e}")
                continue
        
        return articles
        
    except Exception as e:
        print(f"   ⚠ Newspapers failed: {e}")
        return []


def format_for_training(articles: List[str]) -> List[Dict]:
    """Format artikel untuk training CLM"""
    formatted = []
    
    for article in articles:
        # Format sederhana untuk CLM - hanya teks
        formatted.append({"text": article})
    
    return formatted


def save_dataset(data: List[Dict], filename: str):
    """Save dataset ke file JSON"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved: {filepath} ({len(data)} samples)")
    return filepath


def main():
    print("=" * 60)
    print("📰 DOWNLOAD DATASET BERITA INDONESIA")
    print("=" * 60)
    
    all_articles = []
    
    # Download dari berbagai sumber
    # Prioritas: Wikipedia (paling bersih), lalu yang lain
    
    # 1. Wikipedia - paling reliable
    wiki_articles = download_wikipedia_indonesian(max_samples=15000)
    all_articles.extend(wiki_articles)
    
    # 2. mC4 - web crawl berkualitas
    mc4_articles = download_mc4_indonesian(max_samples=15000)
    all_articles.extend(mc4_articles)
    
    # 3. CC-100 - Common Crawl
    cc100_articles = download_cc100_indonesian(max_samples=10000)
    all_articles.extend(cc100_articles)
    
    # 4. OSCAR - web text
    oscar_articles = download_oscar_indonesian(max_samples=10000)
    all_articles.extend(oscar_articles)
    
    # 5. IndoNLU datasets
    indonlu_articles = download_indonlu(max_samples=5000)
    all_articles.extend(indonlu_articles)
    
    # 6. Newspapers
    news_articles = download_id_newspapers(max_samples=5000)
    all_articles.extend(news_articles)
    
    print(f"\n📊 Total articles collected: {len(all_articles)}")
    
    if len(all_articles) == 0:
        print("\n❌ No articles downloaded! Check your internet connection.")
        print("   You can also add manual articles below.")
        
        # Tambahkan contoh artikel manual
        all_articles = get_manual_articles()
    
    # Deduplicate
    print("\n🔄 Removing duplicates...")
    unique_articles = list(set(all_articles))
    print(f"   After dedup: {len(unique_articles)} articles")
    
    # Shuffle
    random.shuffle(unique_articles)
    
    # Limit to target
    if len(unique_articles) > TARGET_SAMPLES:
        unique_articles = unique_articles[:TARGET_SAMPLES]
        print(f"   Limited to: {len(unique_articles)} articles")
    
    # Format
    print("\n📋 Formatting for training...")
    formatted = format_for_training(unique_articles)
    
    # Split train/eval (95/5 for news)
    split_idx = int(len(formatted) * 0.95)
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]
    
    # Save
    print("\n💾 Saving datasets...")
    save_dataset(train_data, "train_news.json")
    save_dataset(eval_data, "eval_news.json")
    
    # Preview
    print("\n📝 Sample articles:")
    print("-" * 50)
    for i, item in enumerate(train_data[:3]):
        preview = item["text"][:300] + "..." if len(item["text"]) > 300 else item["text"]
        print(f"\n[Article {i+1}]")
        print(preview)
        print("-" * 50)
    
    print("\n" + "=" * 60)
    print("✅ DATASET BERITA SIAP!")
    print("=" * 60)
    print(f"\n📊 Statistics:")
    print(f"   Train: {len(train_data)} articles")
    print(f"   Eval: {len(eval_data)} articles")
    print(f"\nUntuk fine-tuning, jalankan:")
    print("  python finetune_news.py")


def get_manual_articles() -> List[str]:
    """Artikel manual jika download gagal"""
    return [
        """Indonesia adalah negara kepulauan terbesar di dunia yang terletak di Asia Tenggara. Dengan lebih dari 17.000 pulau, Indonesia memiliki keanekaragaman budaya, bahasa, dan tradisi yang sangat kaya. Negara ini memiliki populasi lebih dari 270 juta jiwa, menjadikannya negara dengan penduduk terbanyak keempat di dunia. Jakarta adalah ibu kota Indonesia yang juga merupakan pusat pemerintahan dan ekonomi nasional. Indonesia memiliki sumber daya alam yang melimpah, termasuk minyak bumi, gas alam, timah, nikel, dan berbagai hasil pertanian.""",
        
        """Pariwisata di Indonesia terus berkembang pesat dalam beberapa tahun terakhir. Pulau Bali menjadi destinasi wisata paling populer yang menarik jutaan wisatawan mancanegara setiap tahunnya. Selain Bali, Indonesia memiliki banyak destinasi wisata menakjubkan seperti Raja Ampat di Papua, Danau Toba di Sumatera Utara, Candi Borobudur di Jawa Tengah, dan Pulau Komodo di Nusa Tenggara Timur. Pemerintah Indonesia terus berupaya mengembangkan infrastruktur pariwisata untuk meningkatkan kunjungan wisatawan.""",
        
        """Pendidikan di Indonesia terus mengalami perkembangan signifikan. Pemerintah mengalokasikan anggaran pendidikan sebesar 20 persen dari APBN sesuai amanat konstitusi. Program wajib belajar 12 tahun telah diterapkan untuk memastikan setiap anak Indonesia mendapatkan akses pendidikan dasar dan menengah. Berbagai universitas terkemuka di Indonesia seperti Universitas Indonesia, Institut Teknologi Bandung, dan Universitas Gadjah Mada terus meningkatkan kualitas pendidikan tinggi untuk bersaing di tingkat internasional.""",
        
        """Ekonomi Indonesia merupakan yang terbesar di Asia Tenggara dan termasuk dalam G20. Pertumbuhan ekonomi Indonesia rata-rata mencapai 5 persen per tahun dalam dekade terakhir. Sektor utama yang menyumbang PDB Indonesia meliputi manufaktur, pertanian, perdagangan, dan jasa. Indonesia juga dikenal sebagai produsen utama minyak kelapa sawit, karet, kopi, dan berbagai komoditas pertanian lainnya. Digitalisasi ekonomi melalui e-commerce dan fintech juga berkembang pesat di Indonesia.""",
        
        """Kebudayaan Indonesia sangat beragam dan kaya akan tradisi. Setiap daerah memiliki budaya, bahasa, tarian, musik, dan kuliner yang unik. Batik telah diakui UNESCO sebagai Warisan Kemanusiaan untuk Budaya Lisan dan Nonbendawi. Wayang kulit, gamelan, dan berbagai tarian tradisional seperti Tari Saman dari Aceh dan Tari Kecak dari Bali juga menjadi kebanggaan budaya Indonesia. Keragaman budaya ini menjadi kekuatan yang mempersatukan bangsa Indonesia dengan semboyan Bhinneka Tunggal Ika.""",
        
        """Olahraga bulutangkis adalah olahraga yang sangat populer di Indonesia. Indonesia telah melahirkan banyak pemain bulutangkis kelas dunia dan meraih berbagai prestasi di ajang internasional termasuk Olimpiade dan All England. Selain bulutangkis, sepak bola juga memiliki basis penggemar yang sangat besar di Indonesia. Liga 1 Indonesia menjadi kompetisi sepak bola tertinggi yang diikuti oleh berbagai klub dari seluruh nusantara. Pemerintah terus berupaya mengembangkan olahraga nasional melalui pembangunan fasilitas dan pembinaan atlet muda.""",
        
        """Kuliner Indonesia dikenal dengan cita rasa yang kaya dan beragam. Rendang dari Sumatera Barat pernah dinobatkan sebagai makanan terenak di dunia. Setiap daerah memiliki makanan khas yang unik, seperti Gudeg dari Yogyakarta, Pempek dari Palembang, Soto Betawi dari Jakarta, dan Rawon dari Jawa Timur. Nasi goreng menjadi makanan yang paling dikenal di dunia sebagai representasi kuliner Indonesia. Bumbu dan rempah-rempah khas Indonesia seperti kemiri, lengkuas, serai, dan daun salam memberikan cita rasa yang khas pada masakan Indonesia.""",
        
        """Teknologi informasi dan komunikasi berkembang pesat di Indonesia. Pengguna internet di Indonesia telah mencapai lebih dari 200 juta orang. E-commerce dan startup teknologi Indonesia seperti Gojek, Tokopedia, dan Traveloka telah menjadi unicorn dan berkontribusi signifikan terhadap ekonomi digital. Pemerintah meluncurkan berbagai program transformasi digital untuk mempercepat digitalisasi di berbagai sektor. Infrastruktur telekomunikasi terus diperluas hingga ke daerah-daerah terpencil untuk mengurangi kesenjangan digital.""",
        
        """Lingkungan hidup menjadi perhatian serius di Indonesia. Sebagai negara dengan hutan tropis terbesar ketiga di dunia, Indonesia memiliki peran penting dalam menjaga keseimbangan iklim global. Berbagai upaya dilakukan untuk mengurangi deforestasi dan melindungi keanekaragaman hayati. Indonesia adalah rumah bagi berbagai spesies endemik seperti Orangutan, Harimau Sumatera, Badak Jawa, dan Komodo. Program rehabilitasi hutan dan konservasi satwa liar terus digalakkan untuk menjaga kelestarian alam Indonesia.""",
        
        """Kesehatan masyarakat Indonesia terus ditingkatkan melalui berbagai program pemerintah. Program Jaminan Kesehatan Nasional (JKN) melalui BPJS Kesehatan telah mencakup lebih dari 200 juta peserta. Pembangunan fasilitas kesehatan seperti rumah sakit dan puskesmas terus dilakukan hingga ke daerah-daerah. Tenaga kesehatan juga terus ditingkatkan kapasitasnya untuk memberikan pelayanan terbaik kepada masyarakat. Pandemi COVID-19 menjadi tantangan besar yang mendorong percepatan transformasi sektor kesehatan di Indonesia.""",
    ]


if __name__ == "__main__":
    main()
