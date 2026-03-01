import json
import os
import random

def get_topics():
    # Mapping for generic generation logic
    return {
        "budaya_indonesia.json": [
            {"q": "Apa nama rumah adat dari {daerah}?", "a": "Rumah adat dari {daerah} adalah {rumah}."},
            {"q": "Sebutkan alat musik tradisional yang berasal dari {daerah}!", "a": "Salah satu alat musik tradisional dari {daerah} adalah {musik}."},
            {"q": "Apa tarian tradisional yang terkenal dari {daerah}?", "a": "Tarian tradisional yang terkenal dari {daerah} adalah Tari {tari}."}
        ],
        "ekonomi.json": [
            {"q": "Apa yang dimaksud dengan {istilah} dalam ekonomi?", "a": "{istilah} adalah {definisi}."},
            {"q": "Bagaimana pengaruh {faktor} terhadap harga pasar?", "a": "Pengaruh {faktor} terhadap harga pasar adalah {dampak}."}
        ],
        "geografi_indonesia.json": [
            {"q": "Di provinsi manakah terletak Gunung {gunung}?", "a": "Gunung {gunung} terletak di provinsi {provinsi}."},
            {"q": "Apa nama ibu kota dari provinsi {provinsi}?", "a": "Ibu kota dari provinsi {provinsi} adalah {kota}."}
        ],
        "sejarah_indonesia.json": [
            {"q": "Kapan peristiwa {peristiwa} terjadi?", "a": "Peristiwa {peristiwa} terjadi pada tanggal {tanggal}."},
            {"q": "Siapakah tokoh yang berperan dalam {peristiwa}?", "a": "Tokoh yang berperan dalam {peristiwa} adalah {tokoh}."}
        ],
        "makanan_indonesia.json": [
            {"q": "Apa bahan utama pembuatan {makanan}?", "a": "Bahan utama pembuatan {makanan} adalah {bahan}."},
            {"q": "Dari daerah manakah asal makanan {makanan}?", "a": "Makanan {makanan} berasal dari daerah {daerah}."}
        ]
    }

def generate_and_update(filename, base_path):
    file_path = os.path.join(base_path, filename)
    if not os.path.exists(file_path):
        return
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Check if we have specific templates
    topics = get_topics()
    templates = topics.get(filename, [
        {"q": "Ceritakan sedikit tentang {topik}!", "a": "{topik} adalah bagian dari pengetahuan umum yang sangat menarik untuk dipelajari."}
    ])
    
    # Data pool for random replacement (simplified)
    pool = {
        "daerah": ["Sumatera Barat", "Jawa Tengah", "Bali", "Sulawesi Selatan", "Papua"],
        "rumah": ["Rumah Gadang", "Joglo", "Bale Gede", "Tongkonan", "Honai"],
        "musik": ["Saluang", "Gamelan", "Gamelan Bali", "Suling Lembang", "Tifa"],
        "tari": ["Piring", "Serimpi", "Kecak", "Kipas Pakarena", "Sajojo"],
        "istilah": ["Inflasi", "Deflasi", "GDP", "Pasar Modal", "Ekspor"],
        "definisi": ["kenaikan harga secara terus menerus", "penurunan harga secara umum", "Produk Domestik Bruto", "tempat jual beli saham", "pengiriman barang ke luar negeri"],
        "faktor": ["permintaan tinggi", "penawaran rendah", "kenaikan BBM", "kurs mata uang", "suku bunga"],
        "dampak": ["harga cenderung naik", "stok barang menipis", "daya beli menurun", "biaya produksi membengkak", "investasi melambat"],
        "gunung": ["Merapi", "Semeru", "Rinjani", "Kerinci", "Agung"],
        "provinsi": ["Jawa Tengah", "Jawa Timur", "NTB", "Jambi", "Bali"],
        "kota": ["Semarang", "Surabaya", "Mataram", "Jambi", "Denpasar"],
        "peristiwa": ["Proklamasi Kemerdekaan", "Sumpah Pemuda", "Hari Pahlawan", "Bandung Lautan Api", "Serangan Umum 1 Maret"],
        "tanggal": ["17 Agustus 1945", "28 Oktober 1928", "10 November 1945", "24 Maret 1946", "1 Maret 1949"],
        "tokoh": ["Soekarno-Hatta", "Moh. Yamin", "Bung Tomo", "Moh. Toha", "Letkol Soeharto"],
        "makanan": ["Rendang", "Gudeg", "Ayam Betutu", "Coto Makassar", "Papeda"],
        "bahan": ["Daging sapi dan santan", "Nangka muda", "Ayam dan bumbu genep", "Daging sapi dan jeroan", "Sagu"],
        "topik": ["Seni budaya", "Teknologi", "Kesehatan", "Olahraga", "Sains"]
    }

    new_items = []
    for _ in range(500):
        tpl = random.choice(templates)
        q = tpl["q"]
        a = tpl["a"]
        
        # Simple replacement loop
        for key, vals in pool.items():
            token = "{" + key + "}"
            if token in q or token in a:
                val = random.choice(vals)
                q = q.replace(token, val)
                a = a.replace(token, val)
        
        new_items.append({"q": q, "a": a})
    
    data.extend(new_items)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Updated {filename} with 500 items.")

# Main loop
base_dir = "/Users/muhamadimron/Projects/llm/dataset_topics/"
files = [
    "budaya_indonesia.json", "ekonomi.json", "geografi_indonesia.json", 
    "sejarah_indonesia.json", "makanan_indonesia.json", "kesehatan.json",
    "pancasila_kewarganegaraan.json", "pendidikan.json", "sains.json",
    "teknologi_komputer.json", "olahraga.json"
]

for f in files:
    generate_and_update(f, base_dir)
