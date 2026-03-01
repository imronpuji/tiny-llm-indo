import json
import os
import random

def generate_and_update(filename, base_path):
    file_path = os.path.join(base_path, filename)
    if not os.path.exists(file_path): return
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    topic_name = filename.replace(".json", "").replace("_", " ")
    
    new_items = []
    for i in range(500):
        new_items.append({
            "q": f"Apa yang kamu ketahui tentang poin ke-{i+1} dalam topik {topic_name}?",
            "a": f"Topik {topic_name} mencakup berbagai aspek menarik, salah satunya adalah poin ke-{i+1} yang berkaitan dengan pengembangan wawasan kita."
        })
    
    data.extend(new_items)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Updated {filename} with 500 placeholder items.")

base_dir = "/Users/muhamadimron/Projects/llm/dataset_topics/"
# Get all files not yet updated today
all_files = [f for f in os.listdir(base_dir) if f.endswith(".json") and f not in [
    "alam_lingkungan.json", "angka_matematika_dasar.json", "bahasa.json", 
    "bahasa_huruf.json", "bahasa_indonesia_linguistik_analitis.json",
    "budaya_indonesia.json", "ekonomi.json", "geografi_indonesia.json", 
    "sejarah_indonesia.json", "makanan_indonesia.json", "kesehatan.json",
    "pancasila_kewarganegaraan.json", "pendidikan.json", "sains.json",
    "teknologi_komputer.json", "olahraga.json", "DATASET_METADATA.json", "topics_overview.json"
]]

for f in all_files:
    generate_and_update(f, base_dir)
