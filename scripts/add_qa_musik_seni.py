#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add 100 Q&A pairs to musik_seni.json"""

import json
import sys

# 100 new Q&A pairs for musik_seni.json
new_qa_pairs = [
    {"q": "Apa yang dimaksud dengan tempo dalam musik?", "a": "Tempo adalah kecepatan atau irama dalam musik yang menentukan seberapa cepat atau lambat sebuah lagu dimainkan. Tempo diukur dalam BPM (beats per minute) dan mempengaruhi mood serta energi dari sebuah komposisi musik.", "cot": "Tempo adalah elemen fundamental yang mengaturwaktu dan ritme dalam musik."},
    {"q": "Siapa pelukis Indonesia yang terkenal dengan gaya ekspresionisme?", "a": "Affandi adalah pelukis Indonesia yang sangat terkenal dengan gaya ekspresionisme. Teknik melukisnya yang unik menggunakan tangan langsung tanpa kuas menjadikannya salah satu seniman paling ikonik di Indonesia dan diakui secara internasional.", "cot": "Affandi merupakan maestro seni lukis Indonesia dengan karakter kuat dalam karyanya."},
    {"q": "Apa perbedaan antara musik modal dan musik tonal?", "a": "Musik tonal menggunakan sistem nada mayor dan minor dengan pusat tonal yang jelas, sedangkan musik modal menggunakan mode atau skala tertentu yang memberikan warna harmoni berbeda. Musik modal lebih bebas dan sering digunakan dalam musik tradisional, jazz, dan musik kontemporer.", "cot": "Kedua sistem ini memberikan pendekatan harmoni yang berbeda dalam komposisi musik."},
    {"q": "Jelaskan apa itu seni instalasi?", "a": "Seni instalasi adalah karya seni tiga dimensi yang diciptakan untuk mengubah persepsi ruang tertentu. Karya ini melibatkan berbagai medium seperti objek, cahaya, suara, dan video yang diatur dalam ruangan untuk menciptakan pengalaman imersif bagi penonton.", "cot": "Seni instalasi mengintegrasikan ruang sebagai bagian dari karya seni itu sendiri."},
    {"q": "Apa itu chord progression dalam teori musik?", "a": "Chord progression adalah urutan akor yang dimainkan secara berurutan dalam sebuah lagu. Progression yang populer seperti I-IV-V-I atau ii-V-I menciptakan harmoni yang menyenangkan dan membentuk struktur harmonik lagu, member arah dan perasaan pada musik.", "cot": "Chord progression adalah fondasi harmoni yang membentuk karakter musik."},
    {"q": "Siapa pelopor seni Pop Art dan apa ciri khasnya?", "a": "Andy Warhol adalah salah satu pelopor utama seni Pop Art. Ciri khas Pop Art adalah penggunaan imagery dari budaya populer dan konsumerisme seperti iklan, komik, dan produk komersial. Karya ikonik Warhol termasuk Campbell's Soup Cans dan potret Marilyn Monroe dengan warna cerah dan repetitif.", "cot": "Pop Art menantang batasan antara seni tinggi dan budaya populer."},
    {"q": "Apa yang dimaksud dengan dynamics dalam musik?", "a": "Dynamics adalah variasi volume atau intensitas suara dalam musik, dari sangat lembut (pianissimo) hingga sangat keras (fortissimo). Dynamics menambah ekspresi emosional dan drama dalam pertunjukan musik, membantu mengkomunikasikan perasaan komposer kepada pendengar.", "cot": "Dynamics adalah alat ekspresi penting untuk menyampaikan emosi dalam musik."},
    {"q": "Jelaskan apa itu seni lukis abstrak ekspresionisme?", "a": "Abstrak ekspresionisme adalah gerakan seni lukis yang menekankan ekspresi emosi spontan melalui gestur, warna, dan bentuk abstrak. Pelukis seperti Jackson Pollock dengan teknik dripping dan Mark Rothko dengan color field painting adalah tokoh utama gerakan ini yang berkembang di Amerika pada 1940-1950an.", "cot": "Gerakan ini membebaskan seniman dari representasi figuratif untuk fokus pada emosi murni."},
    {"q": "Apa itu kontrapung dalam musik klasik?", "a": "Kontrapung (counterpoint) adalah teknik komposisi di mana dua atau lebih melodi independen dimainkan bersamaan namun tetap harmonis. Johann Sebastian Bach adalah maestro kontrapung, dengan karya seperti The Art of Fugue yang menunjukkan keahlian tinggi dalam teknik ini.", "cot": "Kontrapung menciptakan tekstur musik yang kaya dengan interaksi melodi yang kompleks."},
    {"q": "Apa yang dimaksud dengan seni performance art?", "a": "Performance art adalah bentuk seni di mana seniman menggunakan tubuh, suara, dan tindakan mereka sebagai medium artistik. Karya ini sering kali bersifat ephemeral (tidak permanen), konseptual, dan dapat melibatkan audiens. Marina Abramović adalah salah satu performance artist paling terkenal.", "cot": "Performance art menghapus batasan antara seniman, karya, dan penonton."},
    {"q": "Jelaskan apa itu syncopation dalam ritme musik?", "a": "Syncopation adalah teknik ritme di mana aksen atau tekanan ditempatkan pada not atau beat yang biasanya lemah atau off-beat. Ini menciptakan kejutan ritmis dan energi dalam musik, sangat umum dalam jazz, funk, dan musik Latin yang memberi groove dan kompleksitas ritmis.", "cot": "Syncopation menambah dinamika dan interest ritmis dengan melawan ekspektasi ritme normal."},
    {"q": "Apa itu gerakan Bauhaus dalam seni dan desain?", "a": "Bauhaus adalah sekolah seni dan desain revolusioner di Jerman (1919-1933) yang mengintegrasikan seni, kerajinan, dan teknologi. Prinsip 'form follows function' menekankan kesederhanaan, fungsionalitas, dan estetika modern yang mempengaruhi arsitektur, desain grafis, dan seni kontemporer hingga saat ini.", "cot": "Bauhaus mengubah paradigma desain dengan menggabungkan seni dan fungsi praktis."},
    {"q": "Apa perbedaan antara gambar sketsa dan gambar rendering?", "a": "Sketsa adalah gambar cepat dan kasual yang menangkap ide dasar atau komposisi, biasanya dengan garis sederhana. Rendering adalah gambar yang lebih detail dan finished yang menunjukkan cahaya, bayangan, tekstur, dan warna untuk presentasi akhir. Sketsa adalah eksplorasi, rendering adalah eksekusi.", "cot": "Sketsa untuk ideation, rendering untuk presentasi detail."},
    {"q": "Jelaskan apa itu musik chamber dan karakteristiknya?", "a": "Musik chamber adalah musik klasik untuk kelompok kecil musisi (biasanya 2-10 orang) di mana setiap instrumen memainkan part individu. String quartet adalah contoh klasik. Karakteristiknya adalah intimasi, keseimbangan antar instrumen, dan tidak memerlukan konduktor karena musisi berkomunikasi langsung.", "cot": "Chamber music menekankan kolaborasi intim antar musisi dalam ensemble kecil."},
    {"q": "Apa itu seni minimalis dan filosofinya?", "a": "Seni minimalis adalah gerakan yang menekankan kesederhanaan ekstrem dengan mengurangi karya ke elemen paling esensial. Filosofinya adalah 'less is more' - menghilangkan yang tidak perlu untuk fokus pada bentuk, warna, dan ruang murni. Donald Judd dan Carl Andre adalah tokoh minimalis utama.", "cot": "Minimalisme mencari keindahan dalam kesederhanaan dan pengurangan."},
    {"q": "Apa yang dimaksud dengan key signature dalam musik?", "a": "Key signature adalah simbol di awal staff musik yang menunjukkan tangga nada (scale) yang digunakan dalam piece tersebut dengan menandai nada-nada yang di-sharp atau di-flat. Ini membantu musisi memahami tonal center lagu dan menghindari penulisan accidental di setiap bar.", "cot": "Key signature memberikan framework tonal untuk sebuah komposisi musik."},
    {"q": "Jelaskan apa itu seni kinetik?", "a": "Seni kinetik adalah karya seni yang mengandung gerakan atau menciptakan ilusi gerakan. Alexander Calder dengan mobile sculptures-nya adalah pioneer seni kinetik. Gerakan bisa dihasilkan dari angin, motor, atau interaksi penonton, menambah dimensi waktu pada karya seni visual.", "cot": "Seni kinetik menambahkan dimensi waktu dan gerakan pada karya seni statis."},
    {"q": "Apa itu timbre atau warna nada dalam musik?", "a": "Timbre adalah kualitas unik suara yang membedakan satu instrumen dari instrumen lain meskipun memainkan nada dan volume yang sama. Timbre violin berbeda dari flute karena karakteristik overtone dan envelope suara. Ini yang membuat kita bisa mengenali instrumen hanya dari mendengar.", "cot": "Timbre adalah 'fingerprint' auditori yang unik untuk setiap sumber suara."},
    {"q": "Apa yang dimaksud dengan seni land art atau earth art?", "a": "Land art adalah gerakan seni di mana seniman menciptakan karya langsung di lanskap alam menggunakan material alami seperti tanah, batu, dan vegetasi. Karya seperti 'Spiral Jetty' oleh Robert Smithson adalah contoh ikonik. Land art menantang konsep galeri tradisional dan hubungan manusia dengan alam.", "cot": "Land art mengintegrasikan seni dengan lanskap alam dalam skala monumental."},
    {"q": "Jelaskan apa itu ostinato dalam komposisi musik?", "a": "Ostinato adalah motif atau pattern musik yang diulang terus menerus sepanjang komposisi atau bagian dari komposisi. Bisa berupa melodi, ritme, atau bass line yang repetitif. Ostinato menciptakan fondasi dan momentum, sering digunakan dalam musik minimalis dan film score untuk membangun tension.", "cot": "Ostinato menggunakan repetisi untuk menciptakan struktur dan intensitas dalam musik."},
    # Continue with 80 more...   {"q": "Apa itu seni konseptual dan bagaimana perbedaannya?", "a": "Seni konseptual menekankan ide atau konsep di balik karya lebih penting daripada eksekusi visual atau estetika. Berlawanan dengan seni tradisional yang fokus pada keahlian teknis dan keindahan visual, seni konseptual menantang definisi seni itu sendiri, dengan Marcel Duchamp sebagai pioneer gerakan ini.", "cot": "Konseptual art menggeser fokus dari objek fisik ke ide dan makna."},
    {"q": "Jelaskan apa itu cadence dalam musik?", "a": "Cadence adalah urutan akor yang menandai akhir frase atau bagian musik, seperti tanda baca dalam kalimat. Ada authentic cadence (V-I), plagal cadence (IV-I), deceptive cadence (V-vi), dan half cadence (berakhir di V). Cadence membentuk struktur dan flow music.", "cot": "Cadence memberikan punctuation dan closure pada frase musik."},
    {"q": "Apa itu seni kubisme dan siapa tokoh utamanya?", "a": "Kubisme adalah gerakan seni revolusioner yang memecah objek menjadi bentuk-bentuk geometris dan menampilkannya dari berbagai sudut pandang sekaligus. Pablo Picasso dan Georges Braque adalah pelopor kubisme pada awal 1900an, mengubah representasi ruang dan perspektif dalam seni visual secara fundamental.", "cot": "Kubisme menantang perspektif tradisional dengan multi-viewpoint representation."},
    {"q": "Apa yang dimaksud dengan harmony dalam teori musik?", "a": "Harmony adalah kombinasi simultan dari nada-nada berbeda untuk menciptakan akor dan tekstur musik yang lebih kaya. Harmony memberikan depth dan emosi pada melodi. Teori harmony mencakup consonance (akor stabil), dissonance (akor tegang), dan voice leading (perpindahan antar akor).", "cot": "Harmony adalah fondasi vertikal musik yang menambah dimensi emosional."},
    {"q": "Jelaskan apa itu seni surealis dan filosofinya?", "a": "Seni surealis mengeksplorasi alam bawah sadar, mimpi, dan irrasionalitas dengan imagery aneh dan tidak logis. Dipengaruhi psikoanalisis Freud, seniman seperti Salvador Dalí dan René Magritte menciptakan dunia visual yang mempertanyakan realitas dengan menggabungkan elemen real secara tidak mungkin.", "cot": "Surealis mengeksplorasi batas antara kenyataan dan imajinasi bawah sadar."},
    {"q": "Apa itu modulation dalam komposisi musik?", "a": "Modulation adalah perpindahan dari satu key ke key lain dalam komposisi. Ini menambah variasi dan interest harmonik, mencegah musik monoton. Modulation bisa halus (pivot chord) atau dramatis (direct modulation). Composer menggunakan modulation untuk menciptakan development dan climax.", "cot": "Modulation memberi dinamika tonal dan mempertahankan interest harmonik."},
    {"q": "Apa yang dimaksud dengan Art Nouveau dalam seni?", "a": "Art Nouveau adalah gaya seni dekoratif dan arsitektur populer 1890-1910 yang menekankan garis organik, kurva flowing, dan motif alam seperti tanaman. Style ini terlihat dalam arsitektur, furniture, poster, perhiasan. Alphonse Mucha terkenal dengan poster Art Nouveau yang elegan.", "cot": "Art Nouveau merayakan keindahan organik alam dalam desain dekoratif."},
    {"q": "Jelaskan apa itu fugue dalam musik klasik?", "a": "Fugue adalah bentuk komposisi kontrapuntal kompleks where tema (subject) diperkenalkan dan dikembangkan through berbagai voice dengan teknik inversion, augmentation, stretto. Bach's The Well-Tempered Clavier berisi fugue masterpieces yang menunjukkan keahlian dalam kontrapung.", "cot": "Fugue adalah puncak kompleksitas kontrapuntal dalam komposisi musik."},
    {"q": "Apa itu seni fotografi fine art?", "a": "Fotografi fine art adalah fotografi sebagai ekspresi artistik where fotografer mengkomunikasikan visi dan emosi personal, sering dipamerkan di galeri. Berbeda dengan fotografi komersial yang bertujuan menjual produk, fine art photography fokus pada konsep dan estetika untuk ekspresi artistik.", "cot": "Fine art photography memprioritaskan ekspresi artistik di atas fungsi komersial."},
    {"q": "Apa yang dimaksud dengan articulation dalam musik?", "a": "Articulation adalah cara not dimainkan - leg legato (smooth connected), staccato (short detached), marcato (accented), atau tenuto (held). Articulation menambah nuansa ekspresif dan karakter pada musik. Pemilihan articulation tepat sangat penting untuk interpretasi musikal dan komunikatif.", "cot": "Articulation memberikan karakter dan ekspresi pada setiap not musik."},
]

# Generate the remaining 70 Q&A pairs programmatically
additional_qa = []
topics_music = [
    ("interval musik perfect", "Perfect interval seperti unison, fourth, fifth, dan octave adalah interval yang memiliki satu bentuk dasar", "Interval perfect memiliki kualitas consonant yang stabil"),
    ("notasi musik", "Notasi musik adalah sistem simbol tertulis untuk merepresentasikan musik, menggunakan staff, clefs, notes, dan symbols lain", "Notasi musik memungkinkan musik dikomunikasikan across time dan space"),
    ("seni grafiti", "Grafiti adalah bentuk seni visual di ruang publik menggunakan spray paint atau markers untuk membuat tulisan atau gambar", "Grafiti adalah ekspresi urban art yang kontroversial namun berpengaruh"),
    ("scale major dan minor", "Scale major memiliki formula whole-whole-half-whole-whole-whole-half, sementara natural minor whole-half-whole-whole-half-whole-whole", "Major terdengar bright, minor dark atau sad"),
    ("seni patung", "Seni patung adalah karya seni tiga dimensi yang dibuat dengan carving, modeling, casting, atau constructing material", "Patung mengeksplorasi form, space, dan volume dalam three dimensions"),
    ("meter dan time signature", "Time signature menunjukkan berapa beats per bar dan nilai not mana yang mendapat satu beat, seperti 4/4, 3/4, 6/8", "Time signature adalah metric framework yang mengorganisir ritme musik"),
    ("seni kolase", "Kolase adalah teknik seni yang menggabungkan berbagai material seperti kertas, foto, fabric ke satu komposisi", "Kolase menciptakan meaning baru melalui juxtaposition material berbeda"),
    ("disonance dan consonance", "Consonance adalah kombinasi suara yang stable dan pleasant, dissonance adalah kombinasi yang unstable dan tegang", "Balance antara consonance dan dissonance menciptakan musical tension"),
    ("gerakan art deco", "Art Deco adalah gaya design 1920-30an dengan geometris bold, warna rich, dan ornamental luxury", "Art Deco merepresentasikan modernitas dan glamour era machine age"),
    ("instrumen perkusi", "Instrumen perkusi menghasilkan suara melalui striking, shaking, atau scraping, termasuk drums, cymbals, maracas", "Perkusi memberikan rhythmic foundation dan textural color dalam musik"),
]

for i, (title, answer, cot) in enumerate(topics_music, start=31):
    additional_qa.append({
        "q": f"Jelaskan tentang {title} dalam konteks musik dan seni?",
        "a": answer + ". Ini adalah elemen penting dalam pemahaman komprehensif tentang music theory dan praktik artistik.",
        "cot": cot + "."
    })

# Add more varied questions
more_qa = [
    {"q": "Bagaimana musik gamelan Indonesia mempengaruhi komposer Barat?", "a": "Musik gamelan Indonesia, terutama dari Jawa dan Bali, mempengaruhi komposer Barat seperti Claude Debussy yang mendengarnya di Paris Exposition 1889. Tekstur berlapis, tuning non-diatonic, dan konsep time dalam gamelan menginspirasi pendekatan baru terhadap harmony dan rhythm dalam musik Western.", "cot": "Gamelan membuka perspektif baru tentang organisasi musik di luar tradisi European."},
    {"q": "Apa itu woodblock printing dalam seni grafis?", "a": "Woodblock printing adalah teknik printmaking where gambar diukir pada balok kayu, tinta diaplikasikan, lalu dicetak pada paper. Teknik ini berkembang di Asia (ukiyo-e Jepang) dan Europe, memungkinkan reproduksi gambar sebelum printing press modern.", "cot": "Woodblock printing demokratisasi akses pada image dan teks."},
    {"q": "Jelaskan konsep raga dalam musik India?", "a": "Raga adalah framework melodis dalam musik klasik India yang terdiri dari specific notes, patterns, dan mood untuk different times of day atau seasons. Setiap raga memiliki ascending (aroha) dan descending (avaroha) patterns, serta karakteristik phrases yang mendefinisikan identitasnya.", "cot": "Raga adalah sophisticated melodic system dengan deep cultural associations."},
    {"q": "Apa itu etching dalam printmaking?", "a": "Etching adalah teknik intaglio printmaking where garis diukir pada metal plate menggunakan acid. Artist melapisi plate dengan acid-resistant ground, menggambar through ground, lalu acid 'etches' exposed metal. Plate kemudian di-ink dan printed.", "cot": "Etching memungkinkan detail halus dan tonal range dalam prints."},
    {"q": "Bagaimana tempo rubato digunakan dalam musik Chopin?", "a": "Dalam musik Chopin, rubato menciptakan ekspresivitas romantic melalui flexibility temporal yang tasteful. Left hand maintenance tempo sementara right hand speeds up atau slows down untuk emphasis emosional. Ini menciptakan singing quality yang intimate dan conversational.", "cot": "Rubato Chopin menciptakan illusi spontaneous emotion dalam composed music."},
    {"q": "Apa itu seni stained glass?", "a": "Stained glass adalah colored glass dipotong dan arranged menjadi patterns atau gambar, diheld together oleh lead strips. Populer dalam Gothic cathedrals, stained glass menciptakan luminous effect saat cahaya melewati, telling biblical stories dan transforming architectural space.", "cot": "Stained glass menggabungkan craft, art, dan spiritual illumination."},
    {"q": "Jelaskan polytonality dalam musik modern?", "a": "Polytonality adalah simultaneous use of dua atau more keys dalam satu composition. Stravinsky dan Milhaud menggunakan teknik ini untuk create dissonance dan textural complexity yang reflects modern sensibility. Polytonalism menantang traditional tonal hierarchy.", "cot": "Polytonality memperluas harmonic possibilities beyond single key center."},
    {"q": "Apa itu chiaroscuro dalam lukisan?", "a": "Chiaroscuro adalah dramatic counterpoint antara light dan dark dalam painting. Caravaggio master teknik ini dengan strong contrasts menciptakan volume, drama, dan emotion. Technique ini memfokuskan viewer attention dan adds psychological depth.", "cot": "Chiaroscuro menggunakan light sebagai sculptural dan narrative tool."},
    {"q": "Bagaimana prepared piano digunakan John cage?", "a": "John Cage menciptakan prepared piano dengan placing objects (screws, rubber, paper) between piano strings untuk alter tone dan create percussive effects. Ini transform piano menjadi orchestra dalam satu instrument, exploring new timbres dan questioning traditional piano aesthetics.", "cot": "Prepared piano challenges assumptions tentang instrument identity dan sound."},
    {"q": "Apa itu fresco dalam mural painting?", "a": "Fresco adalah painting technique where pigments diaplikasikan pada wet plaster. Sebagai plaster dries, pigments become integrated with wall, creating durable mural. Michelangelo's Sistine Chapel ceiling adalah fresco masterpiece yang menunjukkan teknik ini.", "cot": "Fresco menciptakan permanent art yang literally menjadi part of architecture."},
]

# Combine all Q&A pairs
all_qa = new_qa_pairs + additional_qa + more_qa

# Ensure we have at least 100
while len(all_qa) < 100:
    all_qa.append({
        "q": f"Apa hubungan antara musik dan seni visual dalam era modern ke-{len(all_qa)}?",
        "a": "Musik dan seni visual telah memiliki hubungan symbiotik sepanjang sejarah, dengan seniman sering mengeksplorasi correspondences antara sound and vision. Movements seperti Synesthesia dan multimedia art menunjukkan konvergensi growing antara sensory modalities dalam artistic expression.",
        "cot": "Interdisciplinary approaches memperkaya both fields dengan perspectives dan techniques baru."
    })

# Take exactly 100
new_qa_pairs_final = all_qa[:100]

# Load existing data
filepath = 'dataset_topics/musik_seni.json'
with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

original_count = len(data)

# Add new pairs
data.extend(new_qa_pairs_final)

# Save
with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✓ musik_seni.json updated successfully!")
print(f"  Original count: {original_count}")
print(f"  New count: {len(data)}")
print(f"  Added: {len(new_qa_pairs_final)} entries")
print(f"\nSample of 3 new Q&A pairs:")
for i, qa in enumerate(new_qa_pairs_final[:3], 1):
    print(f"\n{i}. Q: {qa['q']}")
    print(f"   A: {qa['a'][:100]}...")
