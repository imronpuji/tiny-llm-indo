import json
import random

def generate_letter_qa(count):
    qa_list = []
    
    # 1. Spelling/Word Generation (300)
    words = [
        "Apel", "Bola", "Cacing", "Dadu", "Elang", "Feri", "Gajah", "Harimau", "Ikan", "Jeruk",
        "Katak", "Lampu", "Meja", "Nasi", "Obat", "Palu", "Quran", "Roda", "Sapi", "Tali",
        "Ular", "Vas", "Wajan", "Xon", "Yoyo", "Zebra"
    ]
    
    for _ in range(300):
        word = random.choice(words)
        res = ", ".join(list(word.upper()))
        qa_list.append({
            "q": f"Bagaimana cara mengeja kata '{word}'?",
            "a": f"Ejaan kata '{word}' adalah {res}."
        })
        
    # 2. Vowel/Consonant Classification (200)
    for _ in range(200):
        letter = chr(random.randint(65, 90)) # A-Z
        is_vowel = letter in "AIUEO"
        type_letter = "vokal" if is_vowel else "konsonan"
        qa_list.append({
            "q": f"Apakah huruf '{letter}' termasuk vokal atau konsonan?",
            "a": f"Huruf '{letter}' termasuk dalam kategori huruf {type_letter}."
        })
        
    # 3. First Letter Identification (250)
    objects = [
        ("Matahari", "M"), ("Bintang", "B"), ("Awan", "A"), ("Hujan", "H"), ("Petir", "P"),
        ("Gunung", "G"), ("Lembah", "L"), ("Sungai", "S"), ("Danau", "D"), ("Laut", "L"),
        ("Sawi", "S"), ("Wortel", "W"), ("Bayam", "B"), ("Kubis", "K"), ("Terong", "T")
    ]
    for _ in range(250):
        obj, first = random.choice(objects)
        qa_list.append({
            "q": f"Apa huruf pertama dari kata '{obj}'?",
            "a": f"Huruf pertama dari kata '{obj}' adalah huruf '{first}'."
        })
        
    # 4. Alphabet Order (250)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for _ in range(250):
        idx = random.randint(1, 24)
        prev = alphabet[idx-1]
        curr = alphabet[idx]
        nxt = alphabet[idx+1]
        
        mode = random.choice(["before", "after"])
        if mode == "before":
            qa_list.append({
                "q": f"Huruf apa yang ada tepat sebelum huruf '{curr}' dalam urutan alfabet?",
                "a": f"Huruf yang ada sebelum '{curr}' adalah '{prev}'."
            })
        else:
            qa_list.append({
                "q": f"Huruf apa yang ada tepat setelah huruf '{curr}' dalam urutan alfabet?",
                "a": f"Huruf yang ada setelah '{curr}' adalah '{nxt}'."
            })
            
    return qa_list[:count]

# Load existing
file_path = "/Users/muhamadimron/Projects/llm/dataset_topics/bahasa_huruf.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Generate 1000 new ones
new_qa = generate_letter_qa(1000)
data.extend(new_qa)

# Write back
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Berhasil menambahkan 1000 Q&A bahasa_huruf baru ke {file_path}")
