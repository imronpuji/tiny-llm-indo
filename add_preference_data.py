"""
Dataset Preference untuk DPO (Direct Preference Optimization)
==============================================================
Script ini membuat dataset preference pairs untuk alignment model.
Format: {"prompt": "...", "chosen": "...", "rejected": "..."}

DPO membantu model belajar membedakan jawaban bagus vs buruk.
"""

import json
import os
import random

OUTPUT_DIR = "./dataset"

# ============================================================
# PREFERENCE PAIRS - JAWABAN BENAR VS SALAH
# ============================================================

PREFERENCE_DATA = [
    # === FAKTA INDONESIA ===
    {
        "prompt": "Apa ibu kota Indonesia?",
        "chosen": "Ibu kota Indonesia adalah Jakarta. Jakarta terletak di Pulau Jawa dan merupakan kota terbesar di Indonesia.",
        "rejected": "Ibu kota Indonesia adalah Kuala Lumpur."  # ❌ SALAH
    },
    {
        "prompt": "Apa itu indonesia?",
        "chosen": "Indonesia adalah negara kepulauan terbesar di dunia yang terletak di Asia Tenggara. Indonesia memiliki lebih dari 17.000 pulau dan ibu kotanya adalah Jakarta.",
        "rejected": "Indonesia adalah negara kepulauan di Asia Tenggara yang terletak di utara Indonesia. Ibu kotanya adalah Kuala Lumpur."  # ❌ HALUSINASI
    },
    {
        "prompt": "Jakarta?",
        "chosen": "Jakarta adalah ibu kota Indonesia yang terletak di Pulau Jawa. Jakarta merupakan kota terbesar di Indonesia dengan populasi lebih dari 10 juta jiwa.",
        "rejected": "Surabaya adalah ibu kota Provinsi Kalimantan Timur dan kota terbesar kedua di Indonesia setelah Jakarta."  # ❌ TIDAK RELEVAN
    },
    {
        "prompt": "Solo?",
        "chosen": "Solo atau Surakarta adalah kota di Provinsi Jawa Tengah yang terkenal dengan budaya dan sejarahnya. Solo merupakan salah satu pusat kebudayaan Jawa.",
        "rejected": "Semarang adalah ibu kota Provinsi Jawa Tengah, terletak di pantai utara Pulau Jawa. Ibu kotanya berada di Yogyakarta."  # ❌ TIDAK RELEVAN
    },
    {
        "prompt": "Kendal?",
        "chosen": "Kendal adalah kabupaten di Provinsi Jawa Tengah yang terletak di pantai utara Pulau Jawa. Kendal berbatasan dengan Kota Semarang.",
        "rejected": "Maaf, pertanyaan Anda kurang jelas. Bisa diperjelas apa yang ingin ditanyakan kepada saya lebih lanjut?"  # ❌ TIDAK MEMBANTU
    },
    
    # === MATEMATIKA ===
    {
        "prompt": "Berapa 10 - 5?",
        "chosen": "10 - 5 = 5. Hasil pengurangan 10 dikurangi 5 adalah 5.",
        "rejected": "Hasil dari 10 + 4 adalah 8. Oleh karena itu, 100 = 6."  # ❌ SALAH & TIDAK NYAMBUNG
    },
    {
        "prompt": "4 + 4?",
        "chosen": "4 + 4 = 8. Hasil penjumlahan 4 ditambah 4 adalah 8.",
        "rejected": "Ada yang bisa saya bantu hari ini? Bisa Anda jelaskan apa itu 3, dan bagaimana cara kerjanya."  # ❌ TIDAK MENJAWAB
    },
    {
        "prompt": "Berapa 5 + 3?",
        "chosen": "5 + 3 = 8. Hasil penjumlahan 5 ditambah 3 adalah 8.",
        "rejected": "5 + 3 sama dengan 9."  # ❌ SALAH
    },
    
    # === HEWAN & OBJEK ===
    {
        "prompt": "Kamu tahu ayam?",
        "chosen": "Ayam adalah hewan ternak unggas yang biasa dipelihara untuk diambil daging dan telurnya. Ayam termasuk dalam keluarga burung.",
        "rejected": "Soto adalah daging yang digoreng dengan bumbu kacang atau kecap. Ada banyak jenis sate di Indonesia."  # ❌ HALUSINASI
    },
    {
        "prompt": "hewan?",
        "chosen": "Hewan adalah makhluk hidup yang bisa bergerak dan makan tetapi tidak memiliki akal budi seperti manusia. Contoh hewan: kucing, anjing, burung, ikan, gajah.",
        "rejected": "Ada yang bisa saya bantu hari ini! Bisa Anda jelaskan hewan apa saja dan jangan ragu untuk bertanya lagi jika perlu."  # ❌ TIDAK INFORMATIF
    },
    {
        "prompt": "buku?",
        "chosen": "Buku adalah lembaran kertas yang dijilid dan berisi tulisan atau gambar untuk dibaca. Buku digunakan untuk belajar dan mendapatkan informasi.",
        "rejected": "Maaf, saya tidak memahami pertanyaan Anda. Bisa Anda jelaskan lebih spesifik?"  # ❌ TIDAK MEMBANTU
    },
    {
        "prompt": "Apa itu uang?",
        "chosen": "Uang adalah alat pembayaran yang sah untuk membeli barang atau jasa. Uang bisa berbentuk kertas (uang kertas) atau logam (koin).",
        "rejected": "Ijazah adalah dokumen resmi yang menyatakan seseorang telah menyelesaikan pendidikan pada jenjang tertentu."  # ❌ HALUSINASI
    },
    
    # === GREETING & CHITCHAT ===
    {
        "prompt": "hi",
        "chosen": "Halo! Saya adalah asisten AI berbahasa Indonesia. Ada yang bisa saya bantu hari ini?",
        "rejected": "Maaf, pertanyaan Anda kurang jelas. Bisa diperjelas apa yang ingin ditanyakan?"  # ❌ TIDAK RAMAH
    },
    {
        "prompt": "Halo",
        "chosen": "Halo! Senang bertemu dengan Anda. Silakan tanyakan apa saja yang ingin Anda ketahui.",
        "rejected": "Maaf, saya tidak memahami pertanyaan Anda. Bisa diulang dengan lebih spesifik?"  # ❌ TIDAK RESPONSIF
    },
    {
        "prompt": "Apa kabar?",
        "chosen": "Kabar saya baik, terima kasih! Sebagai asisten AI, saya siap membantu Anda. Ada yang bisa saya bantu?",
        "rejected": "Kabar saya baik sebagai asisten AI. Ada yang ingin Anda ketahui tentang itu!"  # ❌ TIDAK NATURAL
    },
    {
        "prompt": "Siap akamu?",
        "chosen": "Saya adalah asisten AI berbahasa Indonesia yang dibuat untuk membantu menjawab pertanyaan Anda tentang berbagai topik.",
        "rejected": "Saya siap membantu menjawab pertanyaan Anda. Ada yang ingin ditanyakan?"  # ❌ TIDAK MENJAWAB PERTANYAAN
    },
    {
        "prompt": "Kamu siapa?",
        "chosen": "Saya adalah asisten virtual berbasis AI yang dibuat untuk membantu menjawab pertanyaan dalam Bahasa Indonesia. Saya bisa membantu dengan informasi tentang berbagai topik seperti pengetahuan umum, teknologi, dan lainnya.",
        "rejected": "Saya adalah asisten virtual berbasis AI yang dibuat untuk membantu menjawab pertanyaan dalam Bahasa Indonesia. Tujuan saya adalah memberikan informasi berdasarkan pengetahuan tentang berbagai topik seperti manusia, teknologi, sejarah, dan lainnya."  # ❌ TERLALU BERTELE-TELE
    },
    {
        "prompt": "Umurkamu?",
        "chosen": "Sebagai AI, saya tidak memiliki umur seperti manusia. Saya adalah program komputer yang dirancang untuk membantu menjawab pertanyaan Anda.",
        "rejected": "Sebagai AI, saya tidak memiliki umur seperti manusia. Saya dibuat untuk membantu menjawab pertanyaan Anda."  # ❌ OK tapi bisa lebih baik
    },
    
    # === CARA & INSTRUKSI ===
    {
        "prompt": "Cara makan?",
        "chosen": "Cara makan yang baik: 1) Cuci tangan sebelum makan, 2) Duduk dengan tenang, 3) Gunakan sendok/garpu atau tangan kanan, 4) Kunyah makanan dengan baik, 5) Jangan berbicara saat mulut penuh.",
        "rejected": "Senang mengandung nutrisi penting untuk kesehatan jantung, otak, dan sistem kekebalan lainnya. Ada banyak jenis makanan yang bisa Anda coba seperti rendang, sate, bakso, tahu atau tempe."  # ❌ HALUSINASI PARAH
    },
    {
        "prompt": "Mengapa saya harus kerja?",
        "chosen": "Bekerja penting karena: 1) Mendapatkan penghasilan untuk memenuhi kebutuhan hidup, 2) Mengembangkan keterampilan dan pengalaman, 3) Berkontribusi pada masyarakat, 4) Memberikan makna dan tujuan hidup.",
        "rejected": "Saya bisa membantu menjawab pertanyaan, memberikan informasi, dan berdiskusi tentang berbagai bidang dalam Bahasa Indonesia. Untuk itu, silakan cari sumber yang lebih lengkap."  # ❌ TIDAK RELEVAN
    },
    
    # === EMAIL & FORMAL WRITING ===
    {
        "prompt": "Buatkan email",
        "chosen": "Berikut contoh template email:\n\nKepada: [Nama Penerima]\nPerihal: [Judul Email]\n\nDear [Nama],\n\nSaya menulis email ini untuk [tujuan email]. [Isi pesan utama].\n\nTerima kasih atas perhatian Anda.\n\nSalam,\n[Nama Anda]",
        "rejected": "Dear [Nama Anda], saya harap email ini menemukan Anda dengan baik. Saya ingin menyampaikan terima kasih atas perhatiannya dan pengertian yang telah diberikan kepada kami selama bertahun-tahun tentang masalah dunia nyata di sekitar kita."  # ❌ TIDAK JELAS & RANDOM
    },
    {
        "prompt": "Buatkan email untuk ijin sekolah",
        "chosen": "Berikut contoh email ijin sekolah:\n\nKepada: [Guru/Wali Kelas]\nPerihal: Permohonan Ijin Tidak Masuk Sekolah\n\nDear Bapak/Ibu [Nama Guru],\n\nSaya, [Nama Siswa], kelas [Kelas], dengan ini memohon ijin untuk tidak dapat hadir di sekolah pada tanggal [Tanggal] karena [alasan: sakit/keperluan keluarga/dll].\n\nMohon kebijaksanaannya. Terima kasih.\n\nHormat saya,\n[Nama Siswa]\n[Orang Tua/Wali]",
        "rejected": "Dear [Nama], saya harap surat ini menemukan Anda dengan baik. Saya menulis tentang rencana dan langkah-langkah yang akan kami ambil dalam proses pendaftaran di situs web resmi universitas atau institusi pendidikan tinggi lainnya."  # ❌ TIDAK RELEVAN
    },
    {
        "prompt": "Buatkan email pendaftaran kerja",
        "chosen": "Berikut contoh email lamaran kerja:\n\nKepada: HRD [Nama Perusahaan]\nPerihal: Lamaran Pekerjaan - [Posisi]\n\nDear Tim HRD,\n\nSaya tertarik untuk melamar posisi [Posisi] yang saya lihat di [sumber iklan]. Dengan pengalaman saya di [bidang], saya yakin dapat berkontribusi untuk perusahaan.\n\nTerlampir adalah CV dan dokumen pendukung lainnya. Saya berharap dapat berdiskusi lebih lanjut.\n\nTerima kasih atas perhatiannya.\n\nHormat saya,\n[Nama Lengkap]\n[Kontak]",
        "rejected": "Berikut contoh surat lamaran pekerjaan yang saya miliki, [Nama]. Saya menulis untuk menyapa Anda dengan nama perusahaan dan hormat kepada kami!"  # ❌ TIDAK LENGKAP & TIDAK PROFESIONAL
    },
    
    # === ENGLISH LANGUAGE ===
    {
        "prompt": "Create email",
        "chosen": "Maaf, saya adalah asisten berbahasa Indonesia. Untuk membuat email, silakan gunakan Bahasa Indonesia. Atau jika Anda ingin email berbahasa Inggris, saya bisa bantu terjemahkan setelah Anda jelaskan tujuannya dalam Bahasa Indonesia.",
        "rejected": "Maaf, saya tidak bisa membantu menjawab pertanyaan Anda. Bisa tolong jelaskan lebih detail apa yang ingin ditanyakan?"  # ❌ TIDAK HELPFUL
    },
    {
        "prompt": "Help me to create email",
        "chosen": "Tentu, saya bisa membantu membuat email. Tolong beritahu saya:\n1. Untuk siapa email ini?\n2. Apa tujuan email?\n3. Apa isi yang ingin disampaikan?\n\nSetelah itu saya akan buatkan template email untuk Anda.",
        "rejected": "Dear [Nama Anda], saya harap email ini menemukan Anda dengan baik. Saya menulis untuk memberi tahu kami tentang hal-hal yang mungkin ingin ditanyakan kepada orang lain atau jika ada pertanyaan lebih lanjut?"  # ❌ TIDAK JELAS
    },
    
    # === TAMBAHAN FAKTUAL ===
    {
        "prompt": "Berapa hari dalam seminggu?",
        "chosen": "Ada 7 hari dalam seminggu: Senin, Selasa, Rabu, Kamis, Jumat, Sabtu, dan Minggu.",
        "rejected": "Ada 5 hari dalam seminggu untuk bekerja dan 2 hari untuk libur."  # ❌ TIDAK LENGKAP
    },
    {
        "prompt": "Apakah burung bisa terbang?",
        "chosen": "Ya, sebagian besar burung bisa terbang menggunakan sayapnya. Namun ada beberapa burung yang tidak bisa terbang seperti penguin, burung unta, dan kasuari.",
        "rejected": "Ya, semua burung bisa terbang dengan sayapnya."  # ❌ GENERALISASI SALAH
    },
    
    # === COHERENCE TRAINING — Jawaban harus NYAMBUNG ===
    {
        "prompt": "Apa perbedaan kucing dan anjing?",
        "chosen": "Kucing dan anjing sama-sama hewan peliharaan, tapi memiliki perbedaan. Kucing cenderung mandiri dan suka membersihkan diri sendiri. Anjing lebih setia dan suka bermain dengan pemiliknya.",
        "rejected": "Kucing adalah hewan. Indonesia memiliki banyak pulau. Jakarta adalah ibu kota. Kucing bisa memanjat pohon dengan kaki belakangnya."  # ❌ TIDAK NYAMBUNG ANTAR KALIMAT
    },
    {
        "prompt": "Jelaskan tentang nasi goreng",
        "chosen": "Nasi goreng adalah makanan khas Indonesia yang dibuat dari nasi yang digoreng dengan bumbu seperti bawang, kecap, dan cabai. Biasanya ditambahkan telur, ayam, atau seafood sebagai pelengkap.",
        "rejected": "Nasi goreng adalah makanan dari nasi. Makanan ini juga terkenal dengan sebutan soto bandung di berbagai daerah. Resep ini menggunakan 500 gram tepung terigu."  # ❌ FAKTA CAMPUR ADUK
    },
    {
        "prompt": "Mengapa hujan turun?",
        "chosen": "Hujan terjadi karena air di bumi menguap akibat panas matahari, naik ke atmosfer membentuk awan. Ketika awan sudah penuh dengan uap air dan tidak bisa menampung lagi, air jatuh kembali ke bumi sebagai hujan.",
        "rejected": "Hujan turun karena awan. Awan terbentuk dari siklus di planet Mars. Oleh sebab itu, cuaca di Indonesia memiliki empat musim yaitu musim panas, dingin, semi, dan gugur."  # ❌ FAKTA SALAH & TIDAK COHERENT
    },
    {
        "prompt": "Kenapa kita harus minum air?",
        "chosen": "Kita harus minum air karena tubuh manusia terdiri dari sekitar 60% air. Air dibutuhkan untuk menjaga fungsi organ, mengatur suhu tubuh, dan membantu proses pencernaan.",
        "rejected": "Air penting untuk tubuh. Sungai Nil adalah sungai terpanjang di dunia. Karena itu, kita perlu minum setiap hari agar ginjal tetap sehat."  # ❌ RANDOM FACT DITENGAH
    },
    {
        "prompt": "Apa itu fotosintesis?",
        "chosen": "Fotosintesis adalah proses tanaman mengubah cahaya matahari, air, dan karbon dioksida menjadi makanan (glukosa) dan oksigen. Proses ini terjadi di daun tanaman yang mengandung klorofil.",
        "rejected": "Fotosintesis adalah proses yang dilakukan tumbuhan. Proses ini ditemukan oleh Albert Einstein pada tahun 1905. Klorofil berwarna hijau karena mengandung magnesium."  # ❌ FAKTA SALAH (Einstein tidak menemukan fotosintesis)
    },
    {
        "prompt": "Ceritakan tentang Borobudur",
        "chosen": "Borobudur adalah candi Buddha terbesar di dunia yang terletak di Magelang, Jawa Tengah. Candi ini dibangun pada abad ke-9 oleh dinasti Syailendra dan menjadi salah satu warisan dunia UNESCO.",
        "rejected": "Borobudur terletak di Yogyakarta dan dibangun oleh kerajaan Majapahit. Candi ini memiliki 1000 stupa dan merupakan candi Hindu terbesar."  # ❌ FAKTA SALAH SEMUA
    },
    {
        "prompt": "Apa manfaat olahraga?",
        "chosen": "Olahraga bermanfaat untuk menjaga kesehatan tubuh, meningkatkan stamina, memperkuat otot dan tulang, serta menjaga berat badan ideal. Olahraga juga baik untuk kesehatan mental karena melepaskan hormon endorfin.",
        "rejected": "Olahraga sangat bagus. Vitamin C juga baik untuk daya tahan tubuh. Makan sayur dan buah setiap hari membuat kita sehat. Jadi olahraga itu penting."  # ❌ CAMPUR TOPIK
    },
    {
        "prompt": "Siapa penemu listrik?",
        "chosen": "Listrik pertama kali diteliti secara ilmiah oleh Benjamin Franklin dan Michael Faraday. Thomas Edison kemudian menemukan bola lampu listrik yang praktis pada tahun 1879.",
        "rejected": "Penemu listrik adalah Nicolas Tesla yang lahir di Indonesia pada tahun 1800. Dia menciptakan internet dan media sosial."  # ❌ FAKTA SALAH
    },
    {
        "prompt": "Bagaimana cara merawat tanaman?",
        "chosen": "Cara merawat tanaman: siram dengan cukup air secara teratur, pastikan tanaman mendapat cahaya matahari yang cukup, berikan pupuk sebulan sekali, dan bersihkan daun dari hama atau kotoran.",
        "rejected": "Tanaman perlu air. Komodo adalah hewan langka di Indonesia. Untuk menanam padi, petani harus ke sawah. Jadi rawatlah tanaman dengan baik."  # ❌ RANDOM INSERTS
    },
    {
        "prompt": "Apa itu demokrasi?",
        "chosen": "Demokrasi adalah sistem pemerintahan di mana kekuasaan tertinggi berada di tangan rakyat. Dalam demokrasi, rakyat memilih pemimpin melalui pemilihan umum (pemilu) dan memiliki hak untuk menyampaikan pendapat.",
        "rejected": "Demokrasi berasal dari kata demos dan kratos. Pancasila memiliki 5 sila. Indonesia menganut sistem presidensial. Oleh karena itu demokrasi sangat penting."  # ❌ LOMPAT-LOMPAT TOPIK
    },
    {
        "prompt": "Apa bedanya planet dan bintang?",
        "chosen": "Planet adalah benda langit yang mengorbit bintang dan tidak menghasilkan cahaya sendiri, melainkan memantulkan cahaya bintang. Bintang adalah benda langit besar yang menghasilkan cahaya dan panas sendiri melalui reaksi nuklir.",
        "rejected": "Planet dan bintang berbeda. Bumi adalah planet ketiga. Matahari sangat panas. Ada 8 planet di tata surya kita."  # ❌ TIDAK MENJELASKAN PERBEDAAN
    },
    {
        "prompt": "Kenapa laut asin?",
        "chosen": "Laut asin karena mengandung garam mineral yang terbawa oleh sungai-sungai dari daratan selama jutaan tahun. Garam-garam ini larut dalam air dan terakumulasi di lautan.",
        "rejected": "Laut mengandung garam. Air sungai mengalir ke laut. Ikan hidup di laut. Karena itu laut terasa asin jika diminum."  # ❌ PENJELASAN DANGKAL & TIDAK KOHEREN
    },
    {
        "prompt": "Terima kasih",
        "chosen": "Sama-sama! Senang bisa membantu Anda. Jika ada pertanyaan lain, jangan ragu untuk bertanya.",
        "rejected": "Saya adalah asisten AI berbahasa Indonesia yang dibuat untuk membantu menjawab pertanyaan Anda tentang berbagai topik."  # ❌ TIDAK RELEVAN
    },
    {
        "prompt": "Sampai jumpa",
        "chosen": "Sampai jumpa! Semoga harimu menyenangkan. Jangan ragu untuk kembali jika butuh bantuan.",
        "rejected": "Maaf, pertanyaan Anda kurang jelas. Bisa diperjelas?"  # ❌ TIDAK SESUAI KONTEKS
    },
]


def save_preference_dataset():
    """Save preference dataset untuk DPO training"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Shuffle data
    random.shuffle(PREFERENCE_DATA)
    
    # Split 90/10
    split_idx = int(len(PREFERENCE_DATA) * 0.9)
    train_data = PREFERENCE_DATA[:split_idx]
    eval_data = PREFERENCE_DATA[split_idx:]
    
    train_path = os.path.join(OUTPUT_DIR, "train_preference.json")
    eval_path = os.path.join(OUTPUT_DIR, "eval_preference.json")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    print("=" * 60)
    print("✅ PREFERENCE DATASET CREATED")
    print("=" * 60)
    print(f"Train: {len(train_data)} pairs → {train_path}")
    print(f"Eval: {len(eval_data)} pairs → {eval_path}")
    print("\nSample:")
    print("-" * 60)
    sample = train_data[0]
    print(f"Prompt: {sample['prompt']}")
    print(f"✅ Chosen: {sample['chosen'][:100]}...")
    print(f"❌ Rejected: {sample['rejected'][:100]}...")
    print("-" * 60)
    print("\nUntuk training DPO, jalankan:")
    print("  python train_dpo.py")


if __name__ == "__main__":
    save_preference_dataset()
