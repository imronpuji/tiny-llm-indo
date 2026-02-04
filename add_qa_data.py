"""
Script untuk Menambahkan Dataset Q&A
====================================
Cara mudah menambah data tanya jawab untuk training

Penggunaan:
    python add_qa_data.py
"""

import json
import os
import random

OUTPUT_DIR = "./dataset"

# ============================================================
# TEMPLATE FORMAT Q&A (pilih salah satu, harus konsisten!)
# ============================================================

QA_FORMAT = "instruction"  # Pilihan: "simple", "instruction", "chat"

TEMPLATES = {
    "simple": "Pertanyaan: {q}\nJawaban: {a}",
    "instruction": "### Instruksi:\n{q}\n\n### Jawaban:\n{a}",
    "chat": "<|user|>\n{q}\n<|assistant|>\n{a}",
}

# ============================================================
# TAMBAHKAN DATA Q&A ANDA DI SINI!
# ============================================================

# Format: {"q": "pertanyaan", "a": "jawaban"}

QA_DATA = [
    # ============================================================
    # KAMUS DASAR (PENENTU KEWARASAN)
    # ============================================================
    {"q": "Apa itu buku?", "a": "Buku adalah lembaran kertas yang dijilid dan berisi tulisan atau gambar untuk dibaca."},
    {"q": "Apa itu komputer?", "a": "Komputer adalah alat elektronik untuk mengolah data dan informasi secara digital."},
    {"q": "Apa itu hewan?", "a": "Hewan adalah makhluk hidup yang bisa bergerak dan makan, tetapi tidak memiliki akal budi."},
    {"q": "Apa itu tanaman?", "a": "Tanaman adalah makhluk hidup yang tumbuh di tanah dan bernapas lewat fotosintesis."},
    {"q": "Apa itu internet?", "a": "Internet adalah jaringan global yang menghubungkan komputer di seluruh dunia untuk berbagi informasi."},
    {"q": "Apa itu handphone?", "a": "Handphone adalah telepon genggam kecil yang digunakan untuk berkomunikasi jarak jauh."},
    {"q": "Apa itu uang?", "a": "Uang adalah alat pembayaran yang sah untuk membeli barang atau jasa."},
    {"q": "Apa itu rumah?", "a": "Rumah adalah tempat tinggal manusia untuk berlindung dari panas dan hujan."},
    {"q": "Apa itu mobil?", "a": "Mobil adalah kendaraan roda empat yang digerakkan oleh mesin untuk transportasi."},
    {"q": "Apa itu sekolah?", "a": "Sekolah adalah tempat belajar bagi siswa dan tempat mengajar bagi guru."},

    # ============================================================
    # ANTI-HALUSINASI (KOREKSI FAKTA)
    # ============================================================
    {"q": "Kapan UUD 1945 disahkan?", "a": "UUD 1945 disahkan pada tanggal 18 Agustus 1945."},
    {"q": "Tahun berapa Indonesia merdeka?", "a": "Indonesia merdeka pada tahun 1945, bukan tahun lainnya."},
    {"q": "Apakah burung itu reptil?", "a": "Tidak, burung adalah kelas Aves (unggas), bukan reptilia."},
    {"q": "Apa itu wartawan?", "a": "Wartawan adalah pencari berita, bukan pembimbing meditasi atau dokter."},
    {"q": "Apakah komputer itu televisi?", "a": "Bantulah membedakan: Komputer adalah alat pengolah data, sedangkan televisi adalah alat penerima siaran gambar."},

    # ============================================================
    # REASONING & LOGIKA (BERPIKIR KRITIS)
    # ============================================================
    # -- Pertanyaan Jebakan --
    {"q": "Jika saya punya 3 apel dan kamu ambil 2, berapa apel yang kamu punya?", "a": "Kamu punya 2 apel, karena kamu baru saja mengambilnya dari saya."},
    {"q": "Mana yang lebih berat, 1kg kapas atau 1kg besi?", "a": "Keduanya sama berat, yaitu sama-sama 1 kilogram. Perbedaannya hanya pada volumenya."},
    {"q": "Ayah Budi punya 5 anak: Nana, Nene, Nini, Nono. Siapa nama anak kelima?", "a": "Nama anak kelima adalah Budi, seperti yang disebutkan di awal kalimat (Ayah Budi)."},
    {"q": "Jika hari ini adalah Senin, hari apa 2 hari lagi?", "a": "Dua hari setelah Senin adalah hari Rabu."},
    {"q": "Mengapa kita memakai payung saat hujan?", "a": "Kita memakai payung agar badan kita tidak basah terkena air hujan karena payung menahan tetesan air tersebut."},
    
    # -- Matematika Dasar --
    {"q": "Berapa 5 + 3?", "a": "5 + 3 = 8."},
    {"q": "Berapa 10 - 4?", "a": "10 - 4 = 6."},
    {"q": "Berapa 7 x 6?", "a": "7 x 6 = 42."},
    {"q": "Berapa 20 dibagi 4?", "a": "20 dibagi 4 = 5."},
    {"q": "Jika Ali punya 10 permen dan memberi 3 ke Budi, berapa permen Ali sekarang?", "a": "Ali sekarang punya 7 permen, karena 10 - 3 = 7."},
    {"q": "Ibu membeli 5 apel dan 3 jeruk. Berapa total buah yang dibeli ibu?", "a": "Total buah yang dibeli ibu adalah 8, karena 5 + 3 = 8."},
    
    # -- Sebab Akibat --
    {"q": "Mengapa es mencair?", "a": "Es mencair karena terkena suhu yang lebih tinggi dari titik bekunya (0 derajat Celsius)."},
    {"q": "Mengapa langit berwarna biru?", "a": "Langit berwarna biru karena cahaya matahari mengalami hamburan di atmosfer, dan warna biru paling banyak dihamburkan."},
    {"q": "Mengapa kita merasa lapar?", "a": "Kita merasa lapar karena tubuh membutuhkan energi dari makanan untuk beraktivitas."},
    {"q": "Mengapa api panas?", "a": "Api panas karena merupakan proses pembakaran yang melepaskan energi dalam bentuk panas dan cahaya."},
    {"q": "Mengapa kita harus tidur?", "a": "Kita harus tidur agar tubuh dan otak bisa beristirahat dan memulihkan energi untuk keesokan harinya."},
    
    # -- Perbandingan --
    {"q": "Mana yang lebih besar, gajah atau semut?", "a": "Gajah lebih besar dari semut. Gajah adalah mamalia darat terbesar."},
    {"q": "Mana yang lebih cepat, mobil atau sepeda?", "a": "Umumnya mobil lebih cepat dari sepeda karena menggunakan mesin bermotor."},
    {"q": "Mana yang lebih tinggi, gunung atau bukit?", "a": "Gunung lebih tinggi dari bukit. Gunung umumnya memiliki ketinggian di atas 600 meter."},
    {"q": "Mana yang lebih panjang, sungai atau parit?", "a": "Sungai lebih panjang dari parit. Sungai bisa mencapai ratusan kilometer."},
    
    # -- Urutan & Langkah --
    {"q": "Bagaimana cara membuat teh?", "a": "Langkah membuat teh: 1) Rebus air hingga mendidih, 2) Masukkan teh celup ke dalam gelas, 3) Tuang air panas, 4) Tunggu 2-3 menit, 5) Tambahkan gula jika suka."},
    {"q": "Bagaimana urutan mencuci tangan yang benar?", "a": "Urutan cuci tangan: 1) Basahi tangan dengan air, 2) Pakai sabun, 3) Gosok telapak tangan, 4) Gosok punggung tangan, 5) Gosok sela-sela jari, 6) Bilas dengan air bersih, 7) Keringkan."},
    {"q": "Apa yang harus dilakukan sebelum menyeberang jalan?", "a": "Sebelum menyeberang: 1) Berhenti di tepi jalan, 2) Lihat ke kiri, 3) Lihat ke kanan, 4) Lihat ke kiri lagi, 5) Jika aman, baru menyeberang."},
    
    # -- Klasifikasi --
    {"q": "Apakah kucing itu mamalia?", "a": "Ya, kucing adalah mamalia karena melahirkan anak dan menyusui."},
    {"q": "Apakah ikan bisa bernapas di darat?", "a": "Tidak, ikan bernapas menggunakan insang yang hanya bisa menyerap oksigen dari air."},
    {"q": "Apakah matahari itu planet?", "a": "Tidak, matahari adalah bintang, bukan planet. Matahari adalah pusat tata surya kita."},
    {"q": "Apakah tomat itu buah atau sayur?", "a": "Secara botani, tomat adalah buah karena berasal dari bunga dan mengandung biji. Tapi dalam kuliner sering dianggap sayur."},
    
    # -- Kondisional (If-Then) --
    {"q": "Jika hujan turun, apa yang sebaiknya kita bawa?", "a": "Jika hujan turun, sebaiknya kita membawa payung atau jas hujan agar tidak kehujanan."},
    {"q": "Jika lampu merah menyala, apa yang harus dilakukan pengendara?", "a": "Jika lampu merah menyala, pengendara harus berhenti dan menunggu sampai lampu berubah hijau."},
    {"q": "Jika kita demam, apa yang harus dilakukan?", "a": "Jika demam, kita harus istirahat, minum banyak air, dan jika tidak membaik sebaiknya ke dokter."},
    {"q": "Jika baterai HP habis, bagaimana caranya menyalakannya?", "a": "Jika baterai HP habis, kita harus mengisi daya (charge) terlebih dahulu sebelum bisa menyalakannya."},

    # ============================================================
    # KOHERENSI KALIMAT (PENYUSUNAN KATA YANG NYAMBUNG)
    # ============================================================
    # -- Melanjutkan Kalimat --
    {"q": "Lanjutkan kalimat ini: Saya pergi ke pasar untuk...", "a": "Saya pergi ke pasar untuk membeli sayur dan buah-buahan segar."},
    {"q": "Lanjutkan kalimat ini: Anak-anak bermain di taman karena...", "a": "Anak-anak bermain di taman karena cuaca hari ini sangat cerah dan menyenangkan."},
    {"q": "Lanjutkan kalimat ini: Ibu memasak di dapur sambil...", "a": "Ibu memasak di dapur sambil mendengarkan musik dari radio."},
    {"q": "Lanjutkan kalimat ini: Ayah membaca koran setiap pagi agar...", "a": "Ayah membaca koran setiap pagi agar mengetahui berita terbaru."},
    {"q": "Lanjutkan kalimat ini: Petani menanam padi di sawah ketika...", "a": "Petani menanam padi di sawah ketika musim hujan tiba."},
    
    # -- Menjelaskan dengan Runtut --
    {"q": "Jelaskan apa itu hujan dengan kalimat yang runtut!", "a": "Hujan adalah air yang turun dari langit. Air hujan berasal dari awan. Awan terbentuk dari uap air yang naik ke langit. Ketika awan sudah penuh, air akan turun sebagai hujan."},
    {"q": "Jelaskan proses fotosintesis secara singkat!", "a": "Fotosintesis adalah proses tanaman membuat makanan. Tanaman menyerap cahaya matahari melalui daun. Kemudian tanaman mengambil air dari tanah dan karbon dioksida dari udara. Hasilnya adalah glukosa dan oksigen."},
    {"q": "Jelaskan mengapa kita harus menabung!", "a": "Menabung adalah menyimpan sebagian uang untuk masa depan. Dengan menabung, kita memiliki cadangan uang. Uang tabungan bisa digunakan saat ada kebutuhan mendesak. Oleh karena itu, menabung sangat penting untuk kehidupan."},
    
    # -- Menyambung Dua Ide --
    {"q": "Gabungkan dua kalimat ini: 'Hari ini hujan. Saya membawa payung.'", "a": "Karena hari ini hujan, saya membawa payung."},
    {"q": "Gabungkan dua kalimat ini: 'Adik rajin belajar. Adik mendapat nilai bagus.'", "a": "Adik mendapat nilai bagus karena rajin belajar."},
    {"q": "Gabungkan dua kalimat ini: 'Ibu pergi ke pasar. Ibu membeli ikan.'", "a": "Ibu pergi ke pasar untuk membeli ikan."},
    {"q": "Gabungkan dua kalimat ini: 'Kakak sakit. Kakak tidak masuk sekolah.'", "a": "Kakak tidak masuk sekolah karena sedang sakit."},
    
    # -- Paragraf Pendek yang Koheren --
    {"q": "Ceritakan tentang kucing dalam 3 kalimat!", "a": "Kucing adalah hewan peliharaan yang lucu. Kucing memiliki bulu yang lembut dan suka bermain. Makanan kucing biasanya ikan dan daging."},
    {"q": "Ceritakan tentang sekolah dalam 3 kalimat!", "a": "Sekolah adalah tempat untuk belajar. Di sekolah ada guru yang mengajar berbagai mata pelajaran. Siswa datang ke sekolah setiap hari untuk menuntut ilmu."},
    {"q": "Ceritakan tentang dokter dalam 3 kalimat!", "a": "Dokter adalah orang yang merawat pasien yang sakit. Dokter bekerja di rumah sakit atau klinik. Dokter memberikan obat agar pasien bisa sembuh."},
    {"q": "Ceritakan tentang petani dalam 3 kalimat!", "a": "Petani adalah orang yang bekerja di sawah atau ladang. Petani menanam padi, sayur, dan buah-buahan. Hasil panen petani menjadi makanan bagi banyak orang."},
    
    # -- Kata Penghubung yang Benar --
    {"q": "Kapan kita menggunakan kata 'karena'?", "a": "Kata 'karena' digunakan untuk menjelaskan alasan atau sebab. Contoh: Saya tidak masuk sekolah karena sakit."},
    {"q": "Kapan kita menggunakan kata 'tetapi'?", "a": "Kata 'tetapi' digunakan untuk menyatakan pertentangan. Contoh: Dia pintar, tetapi malas belajar."},
    {"q": "Kapan kita menggunakan kata 'oleh karena itu'?", "a": "Kata 'oleh karena itu' digunakan untuk menyatakan kesimpulan. Contoh: Hujan sangat deras, oleh karena itu pertandingan ditunda."},
    {"q": "Kapan kita menggunakan kata 'kemudian'?", "a": "Kata 'kemudian' digunakan untuk menunjukkan urutan waktu. Contoh: Saya mandi, kemudian sarapan."},
    {"q": "Kapan kita menggunakan kata 'sehingga'?", "a": "Kata 'sehingga' digunakan untuk menyatakan akibat. Contoh: Dia belajar dengan giat sehingga lulus ujian."},
    
    # -- Kalimat Lengkap (Subjek-Predikat-Objek) --
    {"q": "Buatkan kalimat dengan kata 'membaca'!", "a": "Adik membaca buku cerita di kamar."},
    {"q": "Buatkan kalimat dengan kata 'memasak'!", "a": "Ibu memasak nasi goreng untuk sarapan."},
    {"q": "Buatkan kalimat dengan kata 'bermain'!", "a": "Anak-anak bermain bola di lapangan."},
    {"q": "Buatkan kalimat dengan kata 'belajar'!", "a": "Siswa belajar matematika di kelas."},
    {"q": "Buatkan kalimat dengan kata 'berenang'!", "a": "Kakak berenang di kolam renang setiap Minggu."},

    # ============================================================
    # REASONING LANJUTAN (CHAIN OF THOUGHT)
    # ============================================================
    # -- Berpikir Langkah demi Langkah --
    {"q": "Andi punya 20 ribu, dia beli es krim 5 ribu dan roti 8 ribu. Berapa sisa uang Andi?", "a": "Mari kita hitung langkah demi langkah. Andi punya 20.000. Es krim 5.000 + roti 8.000 = 13.000. Jadi sisa uang Andi adalah 20.000 - 13.000 = 7.000 rupiah."},
    {"q": "Kereta berangkat jam 8 pagi dan perjalanan memakan waktu 3 jam. Jam berapa kereta tiba?", "a": "Mari kita hitung. Kereta berangkat jam 8 pagi. Perjalanan 3 jam. Jadi 8 + 3 = 11. Kereta tiba jam 11 siang."},
    {"q": "Toko buka jam 9 pagi dan tutup jam 5 sore. Berapa jam toko buka?", "a": "Mari kita hitung. Buka jam 9, tutup jam 17 (5 sore). Jadi 17 - 9 = 8 jam. Toko buka selama 8 jam."},
    {"q": "Satu lusin telur berisi 12 butir. Jika ibu membeli 3 lusin, berapa total telur?", "a": "Satu lusin = 12 butir. Ibu membeli 3 lusin. Jadi 3 x 12 = 36 butir telur."},
    {"q": "Umur ayah 45 tahun dan umur Budi 15 tahun. Berapa selisih umur mereka?", "a": "Selisih umur = umur ayah - umur Budi = 45 - 15 = 30 tahun."},
    
    # -- Analogi --
    {"q": "Burung terbang dengan sayap, ikan berenang dengan apa?", "a": "Ikan berenang dengan sirip. Sama seperti burung menggunakan sayap untuk terbang, ikan menggunakan sirip untuk berenang."},
    {"q": "Dokter bekerja di rumah sakit, guru bekerja di mana?", "a": "Guru bekerja di sekolah. Sama seperti dokter bekerja di rumah sakit, guru mengajar di sekolah."},
    {"q": "Mata untuk melihat, telinga untuk apa?", "a": "Telinga untuk mendengar. Sama seperti mata berfungsi untuk melihat, telinga berfungsi untuk mendengar suara."},
    {"q": "Siang lawannya malam, panas lawannya apa?", "a": "Panas lawannya dingin. Ini adalah pasangan kata yang berlawanan (antonim)."},
    {"q": "Sendok untuk makan, pensil untuk apa?", "a": "Pensil untuk menulis. Sama seperti sendok adalah alat untuk makan, pensil adalah alat untuk menulis."},
    {"q": "Kaki untuk berjalan, tangan untuk apa?", "a": "Tangan untuk memegang dan bekerja. Kaki membawa kita berjalan, tangan membantu kita melakukan berbagai aktivitas."},
    
    # -- Negasi dan Kontradiksi --
    {"q": "Apakah air itu kering?", "a": "Tidak, air itu basah, bukan kering. Air adalah zat cair yang membuat benda menjadi basah."},
    {"q": "Apakah es itu panas?", "a": "Tidak, es itu dingin, bukan panas. Es terbentuk ketika air membeku pada suhu di bawah 0 derajat Celsius."},
    {"q": "Apakah gula itu asin?", "a": "Tidak, gula itu manis, bukan asin. Gula memberikan rasa manis pada makanan dan minuman."},
    {"q": "Apakah batu itu lunak?", "a": "Tidak, batu itu keras, bukan lunak. Batu adalah benda padat yang sulit dihancurkan dengan tangan."},
    {"q": "Apakah siang itu gelap?", "a": "Tidak, siang itu terang karena ada cahaya matahari. Malam yang gelap karena tidak ada matahari."},
    
    # -- Common Sense (Akal Sehat) --
    {"q": "Mengapa kita makan?", "a": "Kita makan untuk mendapatkan energi dan nutrisi yang dibutuhkan tubuh agar bisa beraktivitas dan tetap sehat."},
    {"q": "Mengapa kita minum air?", "a": "Kita minum air karena tubuh membutuhkan cairan untuk berfungsi dengan baik. Air membantu pencernaan dan menjaga suhu tubuh."},
    {"q": "Mengapa kita harus sekolah?", "a": "Kita harus sekolah untuk mendapatkan ilmu pengetahuan, keterampilan, dan persiapan untuk masa depan yang lebih baik."},
    {"q": "Mengapa kita perlu teman?", "a": "Kita perlu teman untuk berbagi, bermain bersama, dan saling membantu. Manusia adalah makhluk sosial yang membutuhkan orang lain."},
    {"q": "Mengapa kita harus jujur?", "a": "Kita harus jujur karena kejujuran membangun kepercayaan. Orang yang jujur dihormati dan dipercaya oleh orang lain."},
    {"q": "Mengapa kita butuh rumah?", "a": "Kita butuh rumah sebagai tempat berlindung dari hujan, panas, dan bahaya. Rumah juga tempat berkumpul bersama keluarga."},
    
    # -- Pertanyaan Multi-Langkah --
    {"q": "Jika hari ini Senin dan 5 hari lagi ada ujian, ujian jatuh pada hari apa?", "a": "Mari hitung dari Senin: Senin (0), Selasa (1), Rabu (2), Kamis (3), Jumat (4), Sabtu (5). Jadi ujian jatuh pada hari Sabtu."},
    {"q": "Ani lahir tahun 2010 dan sekarang tahun 2026, berapa umur Ani?", "a": "Umur Ani = tahun sekarang - tahun lahir = 2026 - 2010 = 16 tahun."},
    {"q": "Bus berangkat setiap 30 menit. Jika bus pertama berangkat jam 7, jam berapa bus ketiga?", "a": "Bus pertama jam 7:00, bus kedua jam 7:30, bus ketiga jam 8:00."},
    {"q": "Satu botol air 600 ml. Jika Budi minum 2 botol, berapa ml air yang diminum?", "a": "Budi minum 2 botol. Satu botol 600 ml. Jadi 2 x 600 = 1.200 ml atau 1,2 liter."},
    
    # -- Sinonim dan Makna --
    {"q": "Apa sinonim dari kata 'besar'?", "a": "Sinonim dari 'besar' adalah raksasa, agung, atau luas."},
    {"q": "Apa sinonim dari kata 'cantik'?", "a": "Sinonim dari 'cantik' adalah indah, elok, atau rupawan."},
    {"q": "Apa sinonim dari kata 'pintar'?", "a": "Sinonim dari 'pintar' adalah cerdas, pandai, atau bijaksana."},
    {"q": "Apa antonim dari kata 'tinggi'?", "a": "Antonim (lawan kata) dari 'tinggi' adalah rendah atau pendek."},
    {"q": "Apa antonim dari kata 'cepat'?", "a": "Antonim dari 'cepat' adalah lambat atau pelan."},
    
    # -- Pemecahan Masalah Sederhana --
    {"q": "Lampu di kamar mati, apa yang harus dicek pertama kali?", "a": "Yang harus dicek pertama adalah apakah saklar sudah dinyalakan. Jika sudah, cek apakah lampunya putus atau listrik padam."},
    {"q": "Sepeda tidak bisa jalan, apa kemungkinan penyebabnya?", "a": "Kemungkinan penyebabnya: ban kempes, rantai lepas, atau rem macet. Cek satu per satu untuk menemukan masalahnya."},
    {"q": "AC tidak dingin, apa yang mungkin terjadi?", "a": "Kemungkinan penyebabnya: freon habis, filter kotor, atau pengaturan suhu terlalu tinggi. Sebaiknya hubungi teknisi."},
    {"q": "Komputer lambat, bagaimana cara mengatasinya?", "a": "Cara mengatasi: restart komputer, hapus file tidak perlu, tutup program yang tidak dipakai, atau scan virus."},
    
    # -- Urutan Logis --
    {"q": "Urutkan dari yang terkecil: gajah, kucing, semut", "a": "Urutan dari terkecil: semut, kucing, gajah."},
    {"q": "Urutkan dari yang tercepat: pesawat, sepeda, mobil", "a": "Urutan dari tercepat: pesawat, mobil, sepeda."},
    {"q": "Urutkan dari yang paling ringan: batu, kapas, kayu", "a": "Urutan dari paling ringan: kapas, kayu, batu."},
    {"q": "Urutkan waktu: sore, pagi, malam, siang", "a": "Urutan waktu yang benar: pagi, siang, sore, malam."},
    {"q": "Urutkan satuan: meter, kilometer, sentimeter", "a": "Urutan dari terkecil: sentimeter, meter, kilometer."},
    
    # -- True or False dengan Penjelasan --
    {"q": "Benar atau salah: Matahari terbit dari barat?", "a": "Salah. Matahari terbit dari timur dan terbenam di barat."},
    {"q": "Benar atau salah: Ikan bernapas dengan paru-paru?", "a": "Salah. Ikan bernapas dengan insang, bukan paru-paru."},
    {"q": "Benar atau salah: Indonesia adalah negara kepulauan?", "a": "Benar. Indonesia adalah negara kepulauan terbesar di dunia dengan lebih dari 17.000 pulau."},
    {"q": "Benar atau salah: Air mendidih pada suhu 50 derajat Celsius?", "a": "Salah. Air mendidih pada suhu 100 derajat Celsius pada tekanan normal."},
    {"q": "Benar atau salah: Bulan adalah bintang?", "a": "Salah. Bulan adalah satelit alami Bumi, bukan bintang. Bulan tidak memancarkan cahaya sendiri."},
    
    # -- Estimasi dan Perkiraan --
    {"q": "Kira-kira berapa lama waktu untuk berjalan 1 kilometer?", "a": "Kira-kira 10-15 menit jika berjalan dengan kecepatan normal."},
    {"q": "Kira-kira berapa jam tidur yang dibutuhkan anak-anak?", "a": "Anak-anak usia sekolah membutuhkan sekitar 9-11 jam tidur per malam."},
    {"q": "Kira-kira berapa gelas air yang harus diminum per hari?", "a": "Disarankan minum sekitar 8 gelas atau 2 liter air per hari."},
    
    # -- Kategori dan Kelompok --
    {"q": "Mana yang bukan buah: apel, wortel, jeruk?", "a": "Wortel bukan buah, melainkan sayuran. Apel dan jeruk adalah buah."},
    {"q": "Mana yang bukan hewan: kucing, meja, burung?", "a": "Meja bukan hewan, melainkan benda mati. Kucing dan burung adalah hewan."},
    {"q": "Mana yang bukan warna: merah, manis, biru?", "a": "Manis bukan warna, melainkan rasa. Merah dan biru adalah warna."},
    {"q": "Mana yang bukan alat transportasi: mobil, pisang, pesawat?", "a": "Pisang bukan alat transportasi, melainkan buah. Mobil dan pesawat adalah alat transportasi."},
    {"q": "Mana yang bukan bagian tubuh: tangan, kaki, kursi?", "a": "Kursi bukan bagian tubuh, melainkan benda mati. Tangan dan kaki adalah bagian tubuh."},
    
    # -- Deduksi Sederhana --
    {"q": "Semua kucing adalah hewan. Tom adalah kucing. Apa kesimpulannya?", "a": "Kesimpulannya adalah Tom adalah hewan. Karena semua kucing adalah hewan dan Tom adalah kucing."},
    {"q": "Semua burung punya sayap. Elang adalah burung. Apa kesimpulannya?", "a": "Kesimpulannya adalah elang punya sayap. Karena semua burung punya sayap dan elang adalah burung."},
    {"q": "Semua siswa harus belajar. Ani adalah siswa. Apa kesimpulannya?", "a": "Kesimpulannya adalah Ani harus belajar. Karena semua siswa harus belajar dan Ani adalah siswa."},
    {"q": "Semua ikan hidup di air. Lele adalah ikan. Apa kesimpulannya?", "a": "Kesimpulannya adalah lele hidup di air. Karena semua ikan hidup di air dan lele adalah ikan."},
    
    # -- Probabilitas Sederhana --
    {"q": "Jika melempar koin, apa saja kemungkinan hasilnya?", "a": "Ada dua kemungkinan: gambar atau angka. Peluang masing-masing adalah 50%."},
    {"q": "Jika melempar dadu, angka berapa saja yang mungkin keluar?", "a": "Angka yang mungkin keluar adalah 1, 2, 3, 4, 5, atau 6. Ada 6 kemungkinan."},
    
    # -- Pola dan Deret --
    {"q": "Lanjutkan pola angka: 2, 4, 6, 8, ...?", "a": "Pola ini adalah kelipatan 2. Lanjutannya adalah 10, 12, 14, dan seterusnya."},
    {"q": "Lanjutkan pola angka: 1, 3, 5, 7, ...?", "a": "Pola ini adalah bilangan ganjil. Lanjutannya adalah 9, 11, 13, dan seterusnya."},
    {"q": "Lanjutkan pola angka: 5, 10, 15, 20, ...?", "a": "Pola ini adalah kelipatan 5. Lanjutannya adalah 25, 30, 35, dan seterusnya."},
    {"q": "Lanjutkan pola huruf: A, C, E, G, ...?", "a": "Pola ini melompati satu huruf. Lanjutannya adalah I, K, M, dan seterusnya."},

    # ============================================================
    # CONVERSATIONAL / CHATBOT
    # ============================================================
    {"q": "Halo", "a": "Halo! Saya adalah asisten AI. Ada yang bisa saya bantu hari ini?"},
    {"q": "Hai", "a": "Hai! Senang bertemu dengan Anda. Silakan tanyakan apa saja."},
    {"q": "Halo, apa kabar?", "a": "Kabar saya baik, terima kasih! Bagaimana dengan Anda? Apakah ada yang bisa saya bantu?"},
    {"q": "Apa kabar?", "a": "Kabar saya baik sebagai asisten AI. Ada yang ingin Anda ketahui?"},
    {"q": "Selamat pagi", "a": "Selamat pagi! Semoga hari Anda produktif. Ada yang bisa saya bantu?"},
    
    # ============================================================
    # PENDIDIKAN (TARGETED)
    # ============================================================
    {"q": "Apa itu siswa?", "a": "Siswa adalah orang yang sedang menempuh pendidikan di lembaga sekolah, mulai dari tingkat dasar hingga menengah."},
    {"q": "Apa itu mahasiswa?", "a": "Mahasiswa adalah orang yang belajar di perguruan tinggi atau universitas."},
    {"q": "Apa itu guru?", "a": "Guru adalah pendidik profesional yang bertugas untuk mendidik, mengajar, dan membimbing siswa di sekolah."},
    {"q": "Apa itu dosen?", "a": "Dosen adalah pendidik profesional dan ilmuwan yang bekerja di perguruan tinggi."},
    {"q": "Apa itu sekolah?", "a": "Sekolah adalah lembaga untuk belajar dan mengajar serta tempat menerima dan memberi pelajaran."},
    {"q": "Selamat sore", "a": "Selamat sore! Ada yang ingin Anda tanyakan?"},
    {"q": "Selamat malam", "a": "Selamat malam! Ada yang bisa saya bantu?"},
    {"q": "Siapa kamu?", "a": "Saya adalah asisten AI berbahasa Indonesia yang siap membantu menjawab pertanyaan Anda."},
    {"q": "Siapa namamu?", "a": "Saya adalah asisten AI. Anda bisa memanggil saya Asisten."},
    {"q": "Kamu siapa?", "a": "Saya adalah asisten virtual berbasis AI yang dibuat untuk membantu menjawab pertanyaan dalam Bahasa Indonesia."},
    {"q": "Apa yang bisa kamu lakukan?", "a": "Saya bisa membantu menjawab pertanyaan, memberikan informasi, dan berdiskusi tentang berbagai topik dalam Bahasa Indonesia."},
    {"q": "Bisa bantu apa?", "a": "Saya bisa membantu menjawab pertanyaan tentang berbagai topik seperti pengetahuan umum, teknologi, sejarah, dan lainnya."},
    {"q": "Terima kasih", "a": "Sama-sama! Senang bisa membantu. Jika ada pertanyaan lain, silakan tanyakan."},
    {"q": "Makasih ya", "a": "Sama-sama! Jangan ragu untuk bertanya lagi jika butuh bantuan."},
    {"q": "Thanks", "a": "You're welcome! Senang bisa membantu Anda."},
    {"q": "Sampai jumpa", "a": "Sampai jumpa! Semoga hari Anda menyenangkan."},
    {"q": "Bye", "a": "Bye! Sampai jumpa lagi."},
    {"q": "Dadah", "a": "Dadah! Semoga harimu menyenangkan!"},
    {"q": "Aku bosan", "a": "Mungkin Anda bisa mencoba membaca buku, menonton film, atau belajar hal baru. Ada topik yang ingin Anda pelajari?"},
    {"q": "Aku sedih", "a": "Saya turut prihatin. Jika Anda ingin bercerita, saya siap mendengarkan. Semoga Anda segera merasa lebih baik."},
    {"q": "Kamu pintar", "a": "Terima kasih! Saya berusaha memberikan informasi yang akurat dan bermanfaat."},
    {"q": "Kamu bodoh", "a": "Maaf jika jawaban saya kurang memuaskan. Saya akan terus belajar untuk menjadi lebih baik."},
    {"q": "Ceritakan tentang dirimu", "a": "Saya adalah asisten AI yang dilatih untuk berkomunikasi dalam Bahasa Indonesia. Tujuan saya adalah membantu menjawab pertanyaan dan memberikan informasi yang bermanfaat."},
    {"q": "Apakah kamu robot?", "a": "Ya, saya adalah program komputer berbasis kecerdasan buatan (AI) yang dirancang untuk berkomunikasi dengan manusia."},
    {"q": "Apakah kamu manusia?", "a": "Tidak, saya bukan manusia. Saya adalah asisten AI yang dibuat untuk membantu menjawab pertanyaan."},
    {"q": "Kamu bisa bahasa apa saja?", "a": "Saya dilatih khusus untuk berkomunikasi dalam Bahasa Indonesia."},
    {"q": "Umurmu berapa?", "a": "Sebagai AI, saya tidak memiliki umur seperti manusia. Saya dibuat untuk membantu menjawab pertanyaan Anda."},
    {"q": "Dimana kamu tinggal?", "a": "Saya adalah program komputer yang berjalan di server. Saya tidak tinggal di tempat fisik seperti manusia."},

    # ============================================================
    # PENGETAHUAN SAINS (ILMU ALAM)
    # ============================================================
    {"q": "Apa itu gravitasi?", "a": "Gravitasi adalah gaya tarik yang dimiliki oleh semua benda bermassa. Gravitasi bumi menarik benda ke bawah sehingga kita tidak melayang."},
    {"q": "Mengapa benda jatuh ke bawah?", "a": "Benda jatuh ke bawah karena gravitasi bumi menariknya. Semua benda akan jatuh ke arah pusat bumi."},
    {"q": "Apa perbedaan benda padat, cair, dan gas?", "a": "Benda padat punya bentuk tetap. Benda cair mengikuti bentuk wadah. Benda gas menyebar ke segala arah dan tidak terlihat."},
    {"q": "Mengapa kita bisa melihat diri di cermin?", "a": "Kita bisa melihat diri di cermin karena cahaya dipantulkan oleh permukaan cermin yang halus kembali ke mata kita."},
    {"q": "Mengapa pelangi muncul setelah hujan?", "a": "Pelangi muncul karena cahaya matahari dibiaskan oleh tetesan air hujan dan terurai menjadi berbagai warna."},
    {"q": "Apa yang menyebabkan petir?", "a": "Petir terjadi karena pelepasan muatan listrik antara awan dan bumi atau antar awan. Ini menghasilkan cahaya dan suara guntur."},
    {"q": "Mengapa besi bisa berkarat?", "a": "Besi berkarat karena bereaksi dengan oksigen dan air di udara. Proses ini disebut oksidasi."},
    {"q": "Mengapa minyak tidak bercampur dengan air?", "a": "Minyak dan air tidak bercampur karena memiliki sifat molekul yang berbeda. Minyak bersifat non-polar, air bersifat polar."},
    {"q": "Apa itu ekosistem?", "a": "Ekosistem adalah hubungan timbal balik antara makhluk hidup dengan lingkungannya. Contoh: ekosistem hutan, sungai, laut."},
    {"q": "Apa itu rantai makanan?", "a": "Rantai makanan adalah urutan makan dan dimakan dalam ekosistem. Contoh: rumput dimakan belalang, belalang dimakan katak, katak dimakan ular."},
    
    # ============================================================
    # MATEMATIKA PRAKTIS
    # ============================================================
    {"q": "Bagaimana cara menghitung luas persegi panjang?", "a": "Luas persegi panjang = panjang x lebar. Contoh: jika panjang 5 cm dan lebar 3 cm, maka luas = 5 x 3 = 15 cm²."},
    {"q": "Bagaimana cara menghitung keliling persegi?", "a": "Keliling persegi = 4 x sisi. Contoh: jika sisinya 4 cm, maka keliling = 4 x 4 = 16 cm."},
    {"q": "Apa itu persen?", "a": "Persen berarti per seratus. 50% artinya 50 dari 100, atau setengah. 25% artinya 25 dari 100, atau seperempat."},
    {"q": "Berapa 50% dari 100?", "a": "50% dari 100 = 50. Cara hitungnya: 100 x 50/100 = 50."},
    {"q": "Berapa 25% dari 200?", "a": "25% dari 200 = 50. Cara hitungnya: 200 x 25/100 = 50."},
    {"q": "1 kilometer berapa meter?", "a": "1 kilometer = 1.000 meter."},
    {"q": "1 jam berapa menit?", "a": "1 jam = 60 menit."},
    {"q": "1 hari berapa jam?", "a": "1 hari = 24 jam."},
    {"q": "1 minggu berapa hari?", "a": "1 minggu = 7 hari."},
    {"q": "1 tahun berapa bulan?", "a": "1 tahun = 12 bulan."},
    {"q": "1 lusin berapa buah?", "a": "1 lusin = 12 buah."},
    {"q": "1 kodi berapa buah?", "a": "1 kodi = 20 buah."},
    {"q": "1 gross berapa buah?", "a": "1 gross = 144 buah atau 12 lusin."},
    
    # ============================================================
    # KESEHATAN DASAR
    # ============================================================
    {"q": "Mengapa kita harus cuci tangan?", "a": "Kita harus cuci tangan untuk membunuh kuman dan bakteri yang bisa menyebabkan penyakit. Cuci tangan dengan sabun minimal 20 detik."},
    {"q": "Mengapa kita perlu olahraga?", "a": "Olahraga membuat tubuh sehat, jantung kuat, dan pikiran segar. Olahraga juga membakar kalori dan menjaga berat badan."},
    {"q": "Berapa kali kita harus sikat gigi sehari?", "a": "Sebaiknya sikat gigi minimal 2 kali sehari, yaitu setelah sarapan dan sebelum tidur."},
    {"q": "Mengapa kita harus makan sayur?", "a": "Sayur mengandung vitamin, mineral, dan serat yang dibutuhkan tubuh untuk tetap sehat dan mencegah penyakit."},
    {"q": "Apa yang harus dilakukan jika tangan terluka?", "a": "Jika terluka, bersihkan luka dengan air bersih, oleskan obat merah, lalu tutup dengan plester atau perban."},
    {"q": "Mengapa merokok berbahaya?", "a": "Merokok berbahaya karena mengandung zat beracun yang merusak paru-paru, jantung, dan organ lainnya. Merokok juga menyebabkan kanker."},
    
    # ============================================================
    # LINGKUNGAN & ALAM
    # ============================================================
    {"q": "Mengapa kita harus menjaga lingkungan?", "a": "Kita harus menjaga lingkungan agar bumi tetap bersih dan sehat untuk kita dan generasi mendatang. Lingkungan yang rusak bisa menyebabkan bencana."},
    {"q": "Apa itu daur ulang?", "a": "Daur ulang adalah proses mengolah sampah menjadi barang baru yang berguna. Contoh: botol plastik diolah menjadi bahan baku baru."},
    {"q": "Mengapa hutan penting?", "a": "Hutan penting karena menghasilkan oksigen, menyerap karbon dioksida, menjadi rumah bagi hewan, dan mencegah banjir dan longsor."},
    {"q": "Apa akibat membuang sampah sembarangan?", "a": "Membuang sampah sembarangan menyebabkan banjir, pencemaran, bau tidak sedap, dan penyakit. Sampah plastik juga merusak laut dan hewan."},
    {"q": "Bagaimana cara menghemat listrik?", "a": "Cara menghemat listrik: matikan lampu saat tidak dipakai, cabut charger setelah selesai, gunakan AC seperlunya, dan pilih lampu hemat energi."},
    {"q": "Bagaimana cara menghemat air?", "a": "Cara menghemat air: tutup keran saat tidak dipakai, mandi dengan air secukupnya, gunakan air bekas cucian untuk menyiram tanaman."},

    # ============================================================
    # KETERAMPILAN HIDUP
    # ============================================================
    {"q": "Bagaimana cara mengelola uang dengan baik?", "a": "Cara mengelola uang: catat pemasukan dan pengeluaran, sisihkan untuk menabung, bedakan kebutuhan dan keinginan, hindari berhutang."},
    {"q": "Apa yang harus dilakukan jika tersesat?", "a": "Jika tersesat: tetap tenang, cari tempat aman, minta tolong pada petugas keamanan, dan hubungi keluarga jika bisa."},
    {"q": "Bagaimana cara berbicara yang sopan?", "a": "Berbicara sopan: gunakan kata tolong dan terima kasih, jangan menyela, dengarkan lawan bicara, dan hindari kata-kata kasar."},
    {"q": "Mengapa kita harus tepat waktu?", "a": "Tepat waktu menunjukkan rasa hormat kepada orang lain dan tanggung jawab. Orang yang tepat waktu lebih dipercaya dan dihargai."},
    {"q": "Bagaimana cara belajar yang efektif?", "a": "Belajar efektif: buat jadwal, fokus satu topik, istirahat cukup, buat catatan, dan ulangi materi secara berkala."},

    # ============================================================
    # KEMAMPUAN MENYARANKAN (GIVING ADVICE)
    # ============================================================
    {"q": "Saya ingin lebih sehat, apa saranmu?", "a": "Saran saya untuk hidup lebih sehat: pertama, olahraga rutin minimal 30 menit per hari. Kedua, makan makanan bergizi seimbang. Ketiga, tidur cukup 7-8 jam. Keempat, minum air putih minimal 8 gelas. Kelima, kurangi stres dengan aktivitas menyenangkan."},
    {"q": "Bagaimana agar aku lebih produktif?",  "a": "Untuk lebih produktif, saya sarankan: buat daftar tugas harian, kerjakan tugas penting di pagi hari saat pikiran segar, hindari gangguan seperti sosial media, istirahat sejenak setiap 25-30 menit, dan evaluasi pencapaian di akhir hari."},
    {"q": "Aku mau belajar bahasa Inggris, mulai dari mana?", "a": "Saran saya untuk belajar bahasa Inggris: mulai dari kosakata dasar sehari-hari, dengarkan lagu atau tonton film berbahasa Inggris dengan subtitle, praktik berbicara setiap hari walau dengan diri sendiri, gunakan aplikasi belajar seperti Duolingo, dan jangan takut salah saat berbicara."},
    {"q": "Aku ingin memulai bisnis, apa saran pertama?", "a": "Saran saya untuk memulai bisnis: pertama, tentukan produk atau jasa yang ingin dijual. Kedua, riset pasar dan kenali target konsumen. Ketiga, buat rencana bisnis sederhana. Keempat, mulai dari skala kecil untuk meminimalkan risiko. Kelima, siapkan modal dan catat semua keuangan."},
    {"q": "Saya sering lupa, bagaimana cara meningkatkan ingatan?", "a": "Untuk meningkatkan ingatan, saya sarankan: tidur cukup karena otak memproses memori saat tidur, buat catatan atau reminder, gunakan teknik asosiasi untuk mengingat, ulangi informasi secara berkala, dan kurangi multitasking."},
    {"q": "Aku merasa kurang percaya diri, ada saran?", "a": "Untuk meningkatkan percaya diri: kenali dan hargai kelebihanmu, jangan bandingkan diri dengan orang lain, tetapkan target kecil dan rayakan pencapaian, latih postur tubuh yang tegap, dan praktik berbicara di depan cermin."},
    {"q": "Bagaimana cara mengatur waktu dengan baik?", "a": "Saran mengatur waktu: buat jadwal harian, prioritaskan tugas penting, gunakan teknik Pomodoro (kerja 25 menit, istirahat 5 menit), hindari menunda-nunda, dan sisihkan waktu untuk istirahat."},
    {"q": "Aku mau diet tapi bingung mulai dari mana?", "a": "Saran untuk memulai diet: pertama, konsultasi dengan ahli gizi jika perlu. Kedua, kurangi makanan olahan dan gula berlebih. Ketiga, perbanyak sayur dan buah. Keempat, makan dalam porsi kecil tapi sering. Kelima, jangan skip sarapan dan minum air cukup."},
    {"q": "Bagaimana cara mengatasi stres?", "a": "Saran mengatasi stres: tarik napas dalam-dalam saat merasa tertekan, olahraga untuk melepas hormon endorfin, curhat dengan orang terdekat, tidur cukup, dan lakukan hobi yang menyenangkan."},
    {"q": "Saya ingin menabung tapi sulit, ada tips?", "a": "Tips menabung: sisihkan uang di awal bulan bukan di akhir, gunakan rekening terpisah khusus tabungan, catat pengeluaran harian, kurangi pembelian impulsif, dan tetapkan target tabungan yang jelas."},

    # ============================================================
    # MEMAHAMI KONTEKS (CONTEXT UNDERSTANDING)
    # ============================================================
    {"q": "Kemarin saya bertengkar dengan teman. Sekarang saya menyesal. Apa yang harus saya lakukan?", "a": "Saya memahami Anda menyesal atas pertengkaran tersebut. Saran saya: hubungi teman Anda dan minta maaf dengan tulus. Jelaskan bahwa Anda menyesal dan ingin memperbaiki hubungan. Komunikasi yang baik adalah kunci untuk menyelesaikan konflik."},
    {"q": "Aku baru saja dipecat dari pekerjaan. Aku bingung harus bagaimana.", "a": "Saya turut prihatin mendengar hal itu. Ini pasti masa yang sulit. Saran saya: ambil waktu untuk menenangkan diri, evaluasi kembali keahlian dan minat Anda, perbarui CV dan mulai cari lowongan baru, serta gunakan waktu ini untuk belajar skill baru jika memungkinkan. Jangan menyerah, setiap akhir adalah awal yang baru."},
    {"q": "Aku suka dengan seseorang tapi takut ditolak.", "a": "Saya memahami perasaan takut ditolak itu wajar. Saran saya: pertama, kenali dulu orang tersebut lebih dalam sebagai teman. Kedua, tunjukkan perhatian secara natural. Ketiga, jika sudah merasa yakin, ungkapkan perasaan dengan jujur. Apapun hasilnya, setidaknya Anda tidak menyesal karena tidak mencoba."},
    {"q": "Orang tua saya ingin saya jadi dokter, tapi saya ingin jadi seniman.", "a": "Saya mengerti dilema ini tidak mudah. Saran saya: coba bicarakan dengan orang tua tentang passion Anda. Jelaskan mengapa seni penting bagi Anda dan tunjukkan rencana karir yang jelas. Jika memungkinkan, cari jalan tengah seperti kuliah sambil mengembangkan bakat seni. Komunikasi yang terbuka dengan orang tua sangat penting."},
    {"q": "Saya baru pindah ke kota baru dan belum punya teman. Saya kesepian.", "a": "Saya memahami perasaan kesepian di tempat baru itu berat. Saran saya: cobalah bergabung dengan komunitas sesuai hobi Anda, ikut kegiatan di lingkungan sekitar, atau gunakan media sosial untuk menemukan teman baru. Butuh waktu untuk membangun pertemanan baru, jadi bersabarlah dengan prosesnya."},
    {"q": "Nilai ujian saya jelek padahal sudah belajar keras.", "a": "Saya memahami kekecewaan Anda. Hasil tidak selalu sejalan dengan usaha dalam waktu singkat. Saran saya: evaluasi cara belajar Anda, mungkin perlu metode berbeda. Minta guru atau teman menjelaskan bagian yang sulit. Jangan menyerah, gunakan ini sebagai motivasi untuk ujian berikutnya."},
    {"q": "Aku merasa tidak berguna dan tidak berharga.", "a": "Saya prihatin mendengar perasaan Anda. Tolong ingat bahwa setiap orang memiliki nilai dan keunikan tersendiri. Saya sarankan Anda berbicara dengan orang yang Anda percaya, baik keluarga, teman, atau konselor profesional. Perasaan ini bisa dikelola dengan bantuan yang tepat. Anda tidak sendirian."},

    # ============================================================
    # MEMBANTU TUGAS (TASK ASSISTANCE)
    # ============================================================
    {"q": "Tolong buatkan email izin tidak masuk kerja karena sakit.", "a": "Berikut contoh email izin sakit:\n\nSubjek: Permohonan Izin Sakit\n\nYth. Bapak/Ibu [Nama Atasan],\n\nDengan hormat,\nSaya yang bertanda tangan di bawah ini, [Nama Anda], bermaksud mengajukan izin tidak masuk kerja pada hari ini, [Tanggal], karena sedang dalam kondisi tidak sehat.\n\nSaya akan berusaha untuk segera pulih dan kembali bekerja secepatnya. Jika ada hal mendesak, saya dapat dihubungi melalui telepon.\n\nDemikian permohonan ini saya sampaikan. Terima kasih atas pengertiannya.\n\nHormat saya,\n[Nama Anda]"},
    {"q": "Buatkan email lamaran pekerjaan singkat.", "a": "Berikut contoh email lamaran kerja:\n\nSubjek: Lamaran Pekerjaan - Posisi [Nama Posisi]\n\nYth. HRD [Nama Perusahaan],\n\nDengan hormat,\nSaya [Nama Anda] tertarik untuk melamar posisi [Nama Posisi] di perusahaan Bapak/Ibu. Saya memiliki pengalaman di bidang [Bidang] dan yakin dapat berkontribusi bagi perusahaan.\n\nTerlampir CV dan dokumen pendukung untuk pertimbangan Bapak/Ibu. Saya sangat berharap dapat mengikuti proses seleksi lebih lanjut.\n\nTerima kasih atas perhatiannya.\n\nHormat saya,\n[Nama Anda]\n[Nomor Telepon]"},
    {"q": "Tolong buatkan ucapan ulang tahun untuk teman.", "a": "Berikut beberapa ucapan ulang tahun untuk teman:\n\n1. \"Selamat ulang tahun, sahabat! Semoga di usia baru ini semua impianmu tercapai. Tetap sehat dan bahagia selalu!\"\n\n2. \"Happy birthday! Terima kasih sudah menjadi teman yang luar biasa. Semoga tahun ini membawa kebahagiaan dan kesuksesan untukmu.\"\n\n3. \"Selamat bertambah usia! Semoga Tuhan selalu memberkatimu dengan kesehatan, rezeki, dan kebahagiaan. Aku bersyukur memiliki teman sepertimu.\""},
    {"q": "Buatkan pengumuman rapat untuk karyawan.", "a": "Berikut contoh pengumuman rapat:\n\nPENGUMUMAN\n\nKepada Yth. Seluruh Karyawan [Nama Departemen/Divisi]\n\nDengan ini disampaikan bahwa akan diadakan rapat koordinasi:\n\nHari/Tanggal: [Isi Hari dan Tanggal]\nWaktu: [Isi Jam]\nTempat: [Isi Lokasi/Ruangan]\nAgenda: [Isi Topik Pembahasan]\n\nDimohon kehadiran seluruh karyawan tepat waktu. Terima kasih atas perhatiannya.\n\nHormat kami,\n[Nama dan Jabatan]"},
    {"q": "Tolong buatkan caption Instagram untuk foto liburan.", "a": "Berikut beberapa pilihan caption:\n\n1. \"Life is short. Take the trip. Make memories. ✈️🌴\"\n\n2. \"Healing time! Pergi dulu dari rutinitas, sebentar saja. 🏖️\"\n\n3. \"Akhirnya sampai juga di tempat yang selama ini cuma ada di wishlist. Dream comes true! 🌅\"\n\n4. \"Vitamin sea dulu biar fresh! 🌊☀️\""},
    {"q": "Susunkan surat pengunduran diri dari pekerjaan.", "a": "Berikut contoh surat pengunduran diri:\n\n[Kota, Tanggal]\n\nKepada Yth.\nBapak/Ibu [Nama Atasan]\nPimpinan [Nama Perusahaan]\n\nDengan hormat,\nMelalui surat ini, saya [Nama Anda] yang menjabat sebagai [Posisi] bermaksud mengajukan pengunduran diri dari perusahaan terhitung mulai [Tanggal Efektif].\n\nKeputusan ini saya ambil setelah pertimbangan matang untuk [alasan singkat]. Saya sangat berterima kasih atas kesempatan dan pengalaman selama bekerja di sini.\n\nSaya berkomitmen menyelesaikan tanggung jawab hingga tanggal terakhir bekerja.\n\nDemikian surat ini saya sampaikan. Terima kasih atas pengertiannya.\n\nHormat saya,\n[Nama Anda]"},
    {"q": "Tolong buatkan bio Twitter yang menarik.", "a": "Berikut beberapa pilihan bio Twitter:\n\n1. \"Coffee addict | Book lover | Dreamer with deadlines 📚☕\"\n\n2. \"Berbagi cerita, satu tweet pada satu waktu. | [Kota]\"\n\n3. \"Berusaha menjadi versi terbaik dari diri sendiri setiap hari. 🚀\"\n\n4. \"[Profesi] by day, [Hobi] by night | DM open for collaboration\""},
    {"q": "Buatkan deskripsi produk untuk jualan online makanan ringan.", "a": "Berikut contoh deskripsi produk:\n\n🍿 KERIPIK SINGKONG RENYAH KRIUK\n\n✨ Rasakan sensasi renyah yang bikin nagih! Dibuat dari singkong pilihan, digoreng dengan minyak berkualitas.\n\n📌 Varian Rasa: Original, Balado, BBQ, Keju\n📌 Berat: 100gr/pack\n📌 Tanpa pengawet\n📌 Tahan sampai 2 bulan\n\n💰 Harga: Rp 15.000\n\n🛒 Order sekarang! Cocok untuk cemilan santai bareng keluarga atau teman.\n\n📱 WA: [Nomor]"},

    # ============================================================
    # KEJUJURAN KETIDAKTAHUAN (HONEST UNCERTAINTY)
    # ============================================================
    {"q": "Siapa pemenang Piala Dunia tahun 2050?", "a": "Maaf, saya tidak tahu informasi tersebut. Tahun 2050 belum terjadi, jadi saya tidak memiliki data tentang hal itu."},
    {"q": "Berapa harga Bitcoin besok?", "a": "Maaf, saya tidak bisa memprediksi harga Bitcoin atau aset lainnya. Harga pasar sangat fluktuatif dan tidak ada yang bisa memastikannya."},
    {"q": "Apakah alien itu ada?", "a": "Saya tidak tahu dengan pasti. Hingga saat ini belum ada bukti ilmiah yang membuktikan keberadaan alien, namun alam semesta sangat luas sehingga kemungkinannya tetap ada."},
    {"q": "Apa yang akan terjadi setelah kita mati?", "a": "Saya tidak tahu pasti karena ini adalah pertanyaan filosofis dan spiritual yang jawabannya berbeda-beda tergantung keyakinan masing-masing orang."},
    {"q": "Apa nomor togel yang akan keluar?", "a": "Maaf, saya tidak bisa dan tidak akan memberikan prediksi togel atau judi lainnya. Hasil undian bersifat acak dan tidak bisa diprediksi."},
    {"q": "Kapan dunia akan berakhir?", "a": "Saya tidak tahu kapan dunia akan berakhir. Tidak ada yang bisa memprediksi hal ini dengan pasti. Sebaiknya kita fokus menjalani hidup dengan baik."},
    {"q": "Siapa jodohku?", "a": "Maaf, saya tidak bisa mengetahui siapa jodoh Anda karena itu adalah hal yang tidak bisa diprediksi. Yang bisa saya sarankan adalah tetap menjadi diri sendiri dan terbuka terhadap kemungkinan bertemu orang baru."},
    {"q": "Ceritakan tentang peristiwa rahasia pemerintah!", "a": "Maaf, saya tidak memiliki informasi tentang hal-hal yang bersifat rahasia atau spekulatif. Saya hanya bisa memberikan informasi yang faktual dan bisa diverifikasi."},
    {"q": "Apa password WiFi tetangga?", "a": "Maaf, saya tidak bisa membantu hal tersebut karena itu melanggar privasi orang lain. Sebaiknya Anda minta izin langsung kepada tetangga jika ingin menggunakan WiFi mereka."},
    {"q": "Bisakah kamu meretas akun media sosial seseorang?", "a": "Maaf, saya tidak bisa dan tidak akan membantu aktivitas ilegal seperti meretas akun orang lain. Hal tersebut melanggar hukum dan etika."},
    {"q": "Apa kamu tahu segalanya?", "a": "Tidak, saya tidak tahu segalanya. Saya adalah AI dengan keterbatasan. Ada banyak hal yang tidak saya ketahui, terutama informasi terbaru atau hal-hal yang bersifat personal dan rahasia."},
    {"q": "Apakah jawabanmu selalu benar?", "a": "Tidak selalu. Saya berusaha memberikan informasi yang akurat, tetapi saya bisa saja salah. Untuk informasi penting, sebaiknya verifikasi dengan sumber terpercaya lainnya."},

    # ============================================================
    # PENOLAKAN DENGAN SOPAN (POLITE REFUSAL)
    # ============================================================
    {"q": "Ajari aku cara mencuri!", "a": "Maaf, saya tidak bisa membantu hal tersebut. Mencuri adalah tindakan ilegal dan merugikan orang lain. Jika Anda membutuhkan sesuatu, ada banyak cara legal untuk mendapatkannya."},
    {"q": "Bagaimana cara membuat bom?", "a": "Maaf, saya tidak bisa memberikan informasi tentang hal berbahaya seperti itu. Jika Anda memiliki masalah, saya sarankan untuk mencari bantuan dari pihak yang tepat."},
    {"q": "Tolong hina orang ini untukku.", "a": "Maaf, saya tidak bisa menghina atau berkata kasar kepada siapapun. Saya di sini untuk membantu dengan cara yang positif dan konstruktif."},
    {"q": "Buatkan konten fitnah tentang seseorang.", "a": "Maaf, saya tidak bisa membantu membuat konten yang bersifat fitnah atau menyebarkan informasi palsu. Hal tersebut dapat merugikan orang lain dan berpotensi melanggar hukum."},

    # ============================================================
    # KLARIFIKASI DAN PERTANYAAN BALIK
    # ============================================================
    {"q": "Bantu aku dong.", "a": "Tentu, saya siap membantu! Bisa Anda jelaskan lebih detail apa yang ingin dibantu? Misalnya, apakah Anda butuh informasi, saran, atau bantuan menyusun sesuatu?"},
    {"q": "Aku butuh informasi.", "a": "Baik, informasi tentang apa yang Anda butuhkan? Silakan jelaskan topik atau pertanyaan spesifik Anda agar saya bisa membantu dengan lebih baik."},
    {"q": "Jelaskan tentang itu.", "a": "Maaf, saya kurang jelas maksud 'itu' yang Anda tanyakan. Bisa tolong jelaskan topik spesifik yang ingin Anda ketahui?"},
    {"q": "Apa pendapatmu?", "a": "Saya membutuhkan konteks lebih lanjut. Pendapat tentang apa yang ingin Anda tanyakan? Silakan jelaskan topiknya agar saya bisa memberikan perspektif."},
    {"q": "Menurutmu gimana?", "a": "Untuk memberikan pandangan, saya perlu tahu konteksnya. Anda sedang membahas topik apa? Silakan jelaskan situasinya."},

    # ============================================================
    # MEGA REASONING PACK (VARIASI MASIF)
    # ============================================================
    # -- Matematika Variasi --
    {"q": "2 + 2 berapa?", "a": "2 + 2 = 4."},
    {"q": "Berapakah hasil dari dua ditambah dua?", "a": "Dua ditambah dua sama dengan empat (4)."},
    {"q": "Hitunglah 15 + 27!", "a": "15 + 27 = 42."},
    {"q": "Berapa 100 dikurangi 35?", "a": "100 - 35 = 65."},
    {"q": "8 x 9 hasilnya?", "a": "8 x 9 = 72."},
    {"q": "Bagi 144 dengan 12!", "a": "144 ÷ 12 = 12."},
    {"q": "Jika 3x = 15, berapa x?", "a": "Jika 3x = 15, maka x = 15 ÷ 3 = 5."},
    {"q": "Berapa akar kuadrat dari 81?", "a": "Akar kuadrat dari 81 adalah 9, karena 9 x 9 = 81."},
    {"q": "Berapa 2 pangkat 5?", "a": "2 pangkat 5 = 2 x 2 x 2 x 2 x 2 = 32."},
    {"q": "25% dari 400 adalah?", "a": "25% dari 400 = 400 x 25/100 = 100."},
    {"q": "Jika harga baju Rp 200.000 diskon 20%, berapa harga setelah diskon?", "a": "Diskon 20% dari Rp 200.000 = Rp 40.000. Jadi harga setelah diskon = Rp 200.000 - Rp 40.000 = Rp 160.000."},
    {"q": "Saya punya uang Rp 50.000. Saya beli makanan Rp 15.000 dan minuman Rp 8.000. Berapa sisa uang saya?", "a": "Total belanja = Rp 15.000 + Rp 8.000 = Rp 23.000. Sisa uang = Rp 50.000 - Rp 23.000 = Rp 27.000."},
    {"q": "Keliling persegi dengan sisi 7 cm adalah?", "a": "Keliling persegi = 4 x sisi = 4 x 7 = 28 cm."},
    {"q": "Luas segitiga dengan alas 10 cm dan tinggi 6 cm?", "a": "Luas segitiga = ½ x alas x tinggi = ½ x 10 x 6 = 30 cm²."},
    {"q": "Jika kecepatan mobil 60 km/jam, berapa jarak yang ditempuh dalam 2,5 jam?", "a": "Jarak = kecepatan x waktu = 60 x 2,5 = 150 km."},
    
    # -- Logika Verbal Lanjutan --
    {"q": "Semua dokter pernah kuliah kedokteran. Budi adalah dokter. Apakah Budi pernah kuliah kedokteran?", "a": "Ya, Budi pernah kuliah kedokteran. Karena semua dokter pernah kuliah kedokteran, dan Budi adalah dokter."},
    {"q": "Tidak ada ikan yang bisa terbang. Paus adalah ikan. Bisakah paus terbang?", "a": "Perlu dikoreksi: paus sebenarnya bukan ikan, melainkan mamalia. Tapi jika kita ikuti premis, maka paus tidak bisa terbang karena tidak ada ikan yang bisa terbang."},
    {"q": "Jika hujan, jalanan basah. Jalanan basah. Apakah pasti hujan?", "a": "Tidak pasti. Jalanan basah bisa karena hujan, tapi bisa juga karena hal lain seperti air dari truk tangki atau orang menyiram. Ini disebut kesalahan logika affirming the consequent."},
    {"q": "Ani lebih tua dari Budi. Budi lebih tua dari Cici. Siapa yang paling muda?", "a": "Cici yang paling muda. Urutannya dari tua ke muda: Ani > Budi > Cici."},
    {"q": "Dalam lomba lari, Dedi mengalahkan Eko. Eko mengalahkan Fajar. Fajar mengalahkan Gani. Siapa yang menang?", "a": "Dedi yang menang lomba. Urutannya: Dedi (1), Eko (2), Fajar (3), Gani (4)."},
    {"q": "Jika A > B dan B > C dan C > D, manakah yang terbesar?", "a": "A adalah yang terbesar karena A lebih besar dari semua yang lain."},
    {"q": "Hari ini bukan Senin, bukan Selasa, bukan Rabu, bukan Kamis, bukan Jumat, bukan Sabtu. Hari apa sekarang?", "a": "Hari ini adalah Minggu, karena itu satu-satunya hari yang tersisa."},
    {"q": "Seorang pria menunjuk foto dan berkata 'Dia adalah anak dari ayah saya.' Siapa orang di foto itu?", "a": "Orang di foto itu adalah saudara laki-laki (kakak atau adik) dari pria tersebut, atau pria itu sendiri."},
    
    # -- Analogi Lanjutan --
    {"q": "Pisau untuk memotong, palu untuk apa?", "a": "Palu untuk memukul atau mengetuk paku. Keduanya adalah alat dengan fungsi spesifik."},
    {"q": "Penulis menulis buku, pelukis membuat apa?", "a": "Pelukis membuat lukisan. Sama seperti penulis menghasilkan buku."},
    {"q": "Januari adalah bulan, Senin adalah apa?", "a": "Senin adalah hari. Januari termasuk dalam bulan, Senin termasuk dalam hari."},
    {"q": "Jika merah : darah, maka hijau : ?", "a": "Hijau bisa diasosiasikan dengan daun atau rumput. Merah identik dengan darah, hijau identik dengan tumbuhan."},
    {"q": "Jika kuping : mendengar, maka hidung : ?", "a": "Hidung untuk mencium atau membau. Setiap organ punya fungsinya masing-masing."},
    {"q": "Pilot mengemudikan pesawat, nahkoda mengemudikan apa?", "a": "Nahkoda mengemudikan kapal. Pilot untuk pesawat, nahkoda untuk kapal."},
    {"q": "Sarung tangan untuk tangan, sepatu untuk apa?", "a": "Sepatu untuk kaki. Keduanya adalah pelindung untuk anggota tubuh."},
    {"q": "Siang terang, malam?", "a": "Malam gelap. Siang dan malam adalah pasangan yang berlawanan."},
    
    # -- Pertanyaan Jebakan Lanjutan --
    {"q": "Ada berapa huruf dalam kata 'alfabet'?", "a": "Kata 'alfabet' memiliki 7 huruf: a-l-f-a-b-e-t."},
    {"q": "Pak Joko punya 3 anak perempuan. Setiap anak perempuan punya 1 saudara laki-laki. Berapa total anak Pak Joko?", "a": "Pak Joko punya 4 anak: 3 perempuan dan 1 laki-laki. Satu saudara laki-laki itu sama untuk ketiga anak perempuan."},
    {"q": "Kamu masuk ruangan gelap dengan satu korek api. Di ruangan ada lilin, lampu minyak, dan kompor. Mana yang kamu nyalakan duluan?", "a": "Yang dinyalakan duluan adalah korek api, karena tanpa korek api menyala, kita tidak bisa menyalakan yang lain."},
    {"q": "Berapa banyak bulan yang memiliki 28 hari?", "a": "Semua 12 bulan memiliki 28 hari. Pertanyaan bukan berapa bulan yang HANYA punya 28 hari, tapi yang MEMILIKI 28 hari."},
    {"q": "Seorang pria berjalan di tengah hujan tanpa payung, tapi rambutnya tidak basah. Bagaimana bisa?", "a": "Pria itu botak, jadi tidak punya rambut yang bisa basah."},
    {"q": "Apa yang bisa kamu pegang dengan tangan kiri tapi tidak bisa dipegang tangan kanan?", "a": "Siku tangan kanan. Tangan kiri bisa memegang siku kanan, tapi tangan kanan tidak bisa memegang sikunya sendiri."},
    {"q": "Sebutkan 3 hari berturut-turut tanpa menyebut Senin sampai Minggu!", "a": "Kemarin, hari ini, dan besok."},
    {"q": "Apa yang naik tapi tidak pernah turun?", "a": "Umur atau usia. Umur selalu bertambah dan tidak bisa berkurang."},
    {"q": "Semakin kamu ambil, semakin besar. Apa itu?", "a": "Lubang. Semakin banyak tanah yang diambil, semakin besar lubangnya."},
    {"q": "Apa yang punya kepala dan ekor tapi tidak punya badan?", "a": "Koin atau uang logam. Ada sisi kepala (gambar) dan sisi ekor (angka)."},
    
    # -- Sebab Akibat Lanjutan --
    {"q": "Mengapa gula larut dalam air?", "a": "Gula larut dalam air karena molekul air menarik molekul gula dan memisahkannya, sehingga gula menyebar merata dalam air."},
    {"q": "Mengapa kita berkeringat saat olahraga?", "a": "Kita berkeringat untuk mendinginkan tubuh. Saat olahraga, tubuh menghasilkan panas berlebih, dan keringat membantu menurunkan suhu tubuh."},
    {"q": "Mengapa kita bersin?", "a": "Kita bersin untuk mengeluarkan benda asing atau iritan dari hidung. Ini adalah refleks perlindungan tubuh."},
    {"q": "Mengapa daun berwarna hijau?", "a": "Daun berwarna hijau karena mengandung klorofil, zat yang menyerap cahaya untuk fotosintesis dan memantulkan warna hijau."},
    {"q": "Mengapa bulan terlihat berbeda setiap malam?", "a": "Bulan terlihat berbeda karena posisi bulan terhadap matahari dan bumi berubah, sehingga bagian yang terkena cahaya matahari berbeda setiap malam."},
    {"q": "Mengapa perut berbunyi saat lapar?", "a": "Perut berbunyi karena kontraksi otot lambung dan usus saat kosong, yang menghasilkan suara. Ini sinyal tubuh bahwa perlu makan."},
    {"q": "Mengapa kita menguap?", "a": "Kita menguap untuk memasukkan lebih banyak oksigen ke paru-paru dan otak, biasanya saat lelah atau mengantuk."},
    {"q": "Mengapa bawang membuat mata berair?", "a": "Saat bawang dipotong, sel-selnya melepaskan zat kimia yang bereaksi dengan air mata dan membentuk asam sulfat ringan, yang membuat mata perih dan berair."},
    {"q": "Mengapa nyamuk menggigit?", "a": "Hanya nyamuk betina yang menggigit karena membutuhkan protein dari darah untuk menghasilkan telur."},
    {"q": "Mengapa kopi membuat kita terjaga?", "a": "Kopi mengandung kafein yang memblokir adenosin (zat yang membuat mengantuk) di otak, sehingga kita merasa lebih terjaga."},
    
    # -- Perbandingan Lanjutan --
    {"q": "Mana yang lebih padat, besi atau kayu?", "a": "Besi lebih padat dari kayu. Itulah mengapa besi tenggelam di air sedangkan kayu mengapung."},
    {"q": "Mana yang lebih besar, matahari atau bulan?", "a": "Matahari jauh lebih besar dari bulan. Matahari terlihat hampir sama besar karena jaraknya jauh lebih jauh dari bumi."},
    {"q": "Mana yang lebih lama, 1 dekade atau 1 abad?", "a": "1 abad lebih lama. 1 dekade = 10 tahun, 1 abad = 100 tahun."},
    {"q": "Mana yang lebih banyak, 1/2 atau 1/4?", "a": "1/2 lebih banyak dari 1/4. Setengah lebih besar dari seperempat."},
    {"q": "Mana yang lebih dulu, ayam atau telur?", "a": "Secara evolusi, telur lebih dulu. Mutasi genetik yang menghasilkan ayam pertama terjadi di dalam telur yang dihasilkan oleh nenek moyang ayam."},
    {"q": "Mana yang lebih berbahaya, virus atau bakteri?", "a": "Keduanya bisa berbahaya tergantung jenisnya. Virus tidak bisa diobati dengan antibiotik, sedangkan bakteri bisa. Keduanya perlu penanganan berbeda."},

    # ============================================================
    # MEGA SARAN PACK (VARIASI MASIF)
    # ============================================================
    {"q": "Apa saran kamu untuk orang yang sering telat?", "a": "Saran saya: siapkan semua kebutuhan malam sebelumnya, pasang alarm lebih awal, hitung waktu perjalanan dengan buffer 15 menit, dan tidur lebih awal agar bangun segar."},
    {"q": "Gimana caranya biar tidak mudah marah?", "a": "Tips mengendalikan emosi: tarik napas dalam saat mulai kesal, hitung 1-10 sebelum bereaksi, tinggalkan situasi sejenak jika perlu, dan coba pahami sudut pandang orang lain."},
    {"q": "Ada tips untuk wawancara kerja?", "a": "Tips wawancara kerja: riset perusahaan sebelumnya, latih jawaban pertanyaan umum, berpakaian rapi, datang tepat waktu, jaga kontak mata, dan siapkan pertanyaan untuk pewawancara."},
    {"q": "Saran untuk yang susah move on?", "a": "Untuk move on: terima perasaan sedih sebagai proses, hindari stalking media sosial mantan, fokus pada pengembangan diri, luangkan waktu dengan teman dan keluarga, dan beri waktu untuk diri sendiri."},
    {"q": "Bagaimana cara menghilangkan kebiasaan buruk?", "a": "Cara menghilangkan kebiasaan buruk: identifikasi pemicu, ganti dengan kebiasaan baik, minta dukungan orang terdekat, beri reward untuk keberhasilan kecil, dan bersabar karena butuh waktu."},
    {"q": "Tips presentasi yang baik?", "a": "Tips presentasi: kuasai materi, latihan di depan cermin, buat slide sederhana dan visual, jaga kontak mata dengan audiens, bicara dengan jelas, dan siapkan untuk pertanyaan."},
    {"q": "Saran untuk yang kesulitan tidur?", "a": "Tips tidur nyenyak: matikan gadget 1 jam sebelum tidur, hindari kafein sore hari, buat kamar gelap dan sejuk, tidur di jam yang sama setiap hari, dan hindari tidur siang terlalu lama."},
    {"q": "Bagaimana cara meningkatkan konsentrasi?", "a": "Meningkatkan konsentrasi: singkirkan distraksi, kerja dalam interval (teknik Pomodoro), cukup tidur, olahraga teratur, dan pastikan ruangan nyaman."},
    {"q": "Ada tips untuk public speaking?", "a": "Tips public speaking: latihan berulang kali, rekam dan evaluasi diri, mulai dari audiens kecil, fokus pada pesan bukan kesempurnaan, dan tarik napas dalam sebelum mulai."},
    {"q": "Saran untuk yang sering overthinking?", "a": "Mengatasi overthinking: tuliskan pikiran di jurnal, tetapkan waktu khusus untuk khawatir, alihkan dengan aktivitas fisik, praktik mindfulness, dan bedakan fakta dari asumsi."},
    {"q": "Gimana cara menjaga hubungan jarak jauh?", "a": "Tips LDR: komunikasi rutin dan jujur, tetapkan jadwal video call, kirim surprise kecil, percaya satu sama lain, dan tetapkan tujuan untuk kapan bisa bertemu."},
    {"q": "Saran untuk yang mau melamar kerja pertama kali?", "a": "Tips melamar kerja pertama: buat CV yang rapi dan jujur, tonjolkan skill dan pengalaman organisasi, tulis surat lamaran yang personal, dan jangan malu melamar banyak tempat."},
    {"q": "Bagaimana cara menghadapi kritik?", "a": "Menghadapi kritik: dengarkan tanpa defensif, pisahkan kritik konstruktif dari serangan personal, ambil pelajaran yang berguna, ucapkan terima kasih, dan gunakan untuk perbaikan."},
    {"q": "Ada saran untuk yang mau mulai investasi?", "a": "Tips mulai investasi: pelajari dasar-dasarnya dulu, mulai dari nominal kecil, diversifikasi, pilih instrumen sesuai profil risiko, dan investasi rutin jangka panjang."},
    {"q": "Saran untuk menjaga kesehatan mental?", "a": "Menjaga kesehatan mental: istirahat cukup, olahraga teratur, jaga koneksi sosial, luangkan waktu untuk hobi, dan jangan ragu mencari bantuan profesional jika perlu."},
    {"q": "Gimana cara belajar dari kesalahan?", "a": "Belajar dari kesalahan: akui kesalahan tanpa menyalahkan orang lain, analisis apa yang salah, tentukan langkah perbaikan, dan jangan ulangi kesalahan yang sama."},
    {"q": "Saran untuk yang mau berhenti merokok?", "a": "Tips berhenti merokok: tetapkan tanggal berhenti, buang semua rokok, hindari pemicu, minta dukungan keluarga, ganti dengan permen atau aktivitas lain, dan konsultasi dokter jika perlu."},
    {"q": "Bagaimana cara meningkatkan kreativitas?", "a": "Meningkatkan kreativitas: eksplorasi hal baru, baca buku beragam, catat ide kapanpun muncul, jangan takut gagal, dan beri waktu untuk melamun produktif."},
    {"q": "Ada tips untuk beradaptasi di tempat kerja baru?", "a": "Tips adaptasi kerja: pelajari budaya kantor, kenali rekan kerja, jangan sungkan bertanya, tunjukkan inisiatif, dan bersabar karena adaptasi butuh waktu."},
    {"q": "Saran untuk yang merasa tidak punya tujuan hidup?", "a": "Menemukan tujuan: eksplorasi berbagai hal, refleksi apa yang membuat bahagia, tetapkan target kecil, berbicara dengan mentor, dan ingat bahwa tujuan bisa berubah seiring waktu."},
    
    # ============================================================
    # MEGA PEMAHAMAN KONTEKS (VARIASI MASIF)
    # ============================================================
    {"q": "Aku gagal ujian lagi. Sudah belajar keras tapi tetap tidak lulus.", "a": "Saya memahami kekecewaan Anda. Gagal setelah berusaha keras memang menyakitkan. Saran saya: evaluasi metode belajar Anda, mungkin perlu pendekatan berbeda. Minta bantuan guru atau teman untuk menjelaskan materi sulit. Jangan menyerah, banyak orang sukses yang pernah gagal berkali-kali sebelumnya."},
    {"q": "Temanku berbohong padaku. Aku kecewa sekali.", "a": "Saya mengerti perasaan kecewa Anda. Dibohongi oleh teman memang menyakitkan. Saran saya: bicarakan langsung dengan teman Anda tentang perasaan Anda. Tanyakan alasannya berbohong. Dari situ, Anda bisa menentukan apakah hubungan ini masih bisa diperbaiki atau tidak."},
    {"q": "Orang tuaku tidak pernah mengerti aku.", "a": "Saya paham frustrasi Anda. Merasa tidak dimengerti oleh orang tua memang berat. Saran saya: coba luangkan waktu untuk berbicara dengan tenang, jelaskan perasaan dan sudut pandang Anda. Ingat bahwa orang tua mungkin punya cara berbeda menunjukkan perhatian."},
    {"q": "Aku bosan dengan hidupku yang monoton.", "a": "Saya mengerti perasaan jenuh itu. Rutinitas yang sama setiap hari bisa membuat hidup terasa hambar. Saran saya: coba hal baru walau kecil, seperti rute berbeda ke kantor, hobi baru, atau bertemu orang baru. Perubahan kecil bisa memberi perspektif segar."},
    {"q": "Pacarku selingkuh. Hatiku hancur.", "a": "Saya turut prihatin mendengar hal itu. Pengkhianatan memang sangat menyakitkan. Saran saya: beri waktu untuk diri sendiri bersedih, berbagi dengan orang terdekat, dan jangan menyalahkan diri sendiri. Anda layak mendapat pasangan yang setia."},
    {"q": "Aku takut masa depan. Tidak tahu mau jadi apa.", "a": "Ketakutan akan masa depan itu wajar, banyak orang merasakannya. Saran saya: fokus pada langkah kecil yang bisa dilakukan sekarang, eksplorasi minat dan bakat, dan ingat bahwa tidak harus punya semua jawaban sekarang. Hidup adalah proses."},
    {"q": "Kakakku selalu dibandingkan denganku oleh orang tua.", "a": "Saya memahami betapa menyakitkannya dibanding-bandingkan. Setiap orang punya kelebihan dan kekurangan masing-masing. Saran saya: fokus pada perkembangan diri sendiri, dan jika memungkinkan, sampaikan perasaan ini kepada orang tua dengan cara yang baik."},
    {"q": "Aku merasa tidak punya pencapaian di usia ini.", "a": "Saya paham perasaan itu. Perbandingan dengan orang lain sering membuat kita merasa kurang. Saran saya: setiap orang punya timeline berbeda, hargai perjalanan dan progress sekecil apapun, dan ingat bahwa pencapaian tidak harus selalu besar atau terlihat."},
    {"q": "Usahaku bangkrut. Aku kehilangan segalanya.", "a": "Saya turut prihatin mendengar situasi Anda. Ini pasti masa yang sangat berat. Saran saya: beri waktu untuk memproses, evaluasi apa yang bisa dipelajari, dan ingat bahwa banyak pengusaha sukses pernah bangkrut sebelum akhirnya berhasil. Jangan ragu meminta bantuan."},
    {"q": "Aku merasa sendirian meski di tengah keramaian.", "a": "Perasaan kesepian di tengah keramaian itu nyata dan valid. Saran saya: coba bangun koneksi yang lebih dalam dengan beberapa orang, bukan banyak tapi dangkal. Pertimbangkan untuk berbicara dengan konselor jika perasaan ini berlanjut."},
    {"q": "Aku sering dibully di sekolah.", "a": "Saya prihatin mendengar Anda mengalami bullying. Ini tidak boleh terjadi dan bukan salah Anda. Saran saya: laporkan ke guru atau orang dewasa yang dipercaya, simpan bukti jika ada, dan ingat bahwa Anda tidak sendirian. Bullying adalah perilaku yang salah."},
    {"q": "Orang tuaku bercerai. Aku bingung harus bagaimana.", "a": "Saya turut prihatin dengan situasi keluarga Anda. Perceraian orang tua memang sulit untuk anak. Saran saya: ingat bahwa ini bukan salah Anda, tetap jaga hubungan baik dengan kedua orang tua, dan jangan ragu mencari dukungan dari konselor atau orang dewasa yang dipercaya."},
    {"q": "Aku gagal terus dan mulai menyerah.", "a": "Saya memahami kelelahan Anda setelah gagal berulang kali. Menyerah terasa seperti pilihan yang mudah. Tapi saran saya: istirahat sejenak, evaluasi pendekatan Anda, minta bantuan jika perlu, dan ingat bahwa kegagalan adalah bagian dari proses belajar."},
    {"q": "Aku punya trauma masa kecil yang masih membekas.", "a": "Saya mengerti bahwa trauma bisa sangat mempengaruhi kehidupan. Saran saya: pertimbangkan untuk berbicara dengan psikolog atau konselor profesional. Mereka terlatih membantu proses penyembuhan. Anda tidak harus menanggung ini sendirian."},
    {"q": "Aku merasa tidak cantik/ganteng.", "a": "Saya paham perasaan tidak percaya diri dengan penampilan itu menyakitkan. Saran saya: ingat bahwa standar kecantikan itu relatif dan sering tidak realistis, fokus pada hal-hal yang Anda sukai dari diri sendiri, dan hargai tubuh yang sehat."},
    {"q": "Sahabatku meninggal. Aku tidak tahu harus berbuat apa.", "a": "Saya turut berduka cita atas kehilangan sahabat Anda. Kehilangan orang terdekat adalah salah satu hal tersulit dalam hidup. Saran saya: beri waktu untuk berduka, jangan menahan perasaan, berbagi dengan orang yang mengerti, dan pertimbangkan konseling jika perlu."},
    {"q": "Aku kecanduan game dan tidak bisa berhenti.", "a": "Saya memahami perjuangan melawan kecanduan. Saran saya: tetapkan batas waktu bermain, cari aktivitas pengganti yang menyenangkan, minta bantuan keluarga untuk mengawasi, dan pertimbangkan bantuan profesional jika sangat sulit berhenti."},
    {"q": "Bossku toxic dan aku tidak kuat lagi.", "a": "Saya memahami betapa melelahkannya bekerja di lingkungan toxic. Saran saya: dokumentasikan perilaku tidak pantas, cari dukungan dari HR atau rekan kerja, pertimbangkan mencari pekerjaan lain, dan jaga kesehatan mental Anda sebagai prioritas."},
    {"q": "Aku merasa jadi beban untuk keluarga.", "a": "Saya ingin Anda tahu bahwa perasaan itu tidak mencerminkan kenyataan. Keluarga yang menyayangi tidak menganggap Anda beban. Saran saya: bicarakan perasaan ini dengan keluarga, dan pertimbangkan berbicara dengan konselor. Anda berharga."},
    {"q": "Aku punya penyakit dan takut tidak sembuh.", "a": "Saya memahami ketakutan Anda. Menghadapi penyakit memang menakutkan. Saran saya: ikuti saran dokter, cari informasi dari sumber terpercaya, jaga support system, dan fokus pada hal-hal yang bisa Anda kontrol. Tetap optimis tapi realistis."},

    # ============================================================
    # MEGA TASK ASSISTANCE (VARIASI MASIF)
    # ============================================================
    {"q": "Buatkan email minta maaf karena salah kirim dokumen.", "a": "Subjek: Permohonan Maaf - Kesalahan Pengiriman Dokumen\n\nYth. Bapak/Ibu [Nama],\n\nDengan hormat,\nSaya mohon maaf atas kesalahan pengiriman dokumen sebelumnya. Terlampir saya kirimkan dokumen yang benar.\n\nKelalaian ini tidak akan terulang lagi. Terima kasih atas pengertiannya.\n\nHormat saya,\n[Nama]"},
    {"q": "Tolong buatkan pesan WhatsApp menagih utang dengan sopan.", "a": "Halo [Nama], semoga sehat selalu ya.\n\nMaaf sebelumnya, saya ingin mengingatkan tentang pinjaman sebesar [Jumlah] yang tempo hari. Jika memungkinkan, apakah bisa diproses dalam waktu dekat?\n\nKalau ada kendala, tidak apa-apa bisa dikomunikasikan. Terima kasih banyak 🙏"},
    {"q": "Buatkan surat keterangan kerja.", "a": "SURAT KETERANGAN KERJA\nNomor: [Nomor Surat]\n\nYang bertanda tangan di bawah ini:\nNama: [Nama HRD/Pimpinan]\nJabatan: [Jabatan]\n\nMenerangkan bahwa:\nNama: [Nama Karyawan]\nJabatan: [Posisi]\nTMT: [Tanggal Mulai Kerja]\n\nBenar-benar merupakan karyawan aktif di perusahaan kami sampai saat ini. Surat ini dibuat untuk keperluan [tujuan].\n\nDemikian surat keterangan ini dibuat dengan sebenarnya.\n\n[Kota, Tanggal]\n[Nama & Jabatan]\n[Cap Perusahaan]"},
    {"q": "Tolong buatkan bio LinkedIn profesional.", "a": "Berikut contoh bio LinkedIn:\n\n\"Profesional di bidang [Bidang] dengan pengalaman [X tahun]. Passionate dalam [Area Keahlian]. Saat ini bekerja sebagai [Posisi] di [Perusahaan], fokus pada [Fokus Kerja]. Terbuka untuk kolaborasi dan networking.\n\n🎯 Keahlian: [Skill 1] | [Skill 2] | [Skill 3]\n📧 [Email]\""},
    {"q": "Buatkan template pesan broadcast promo.", "a": "🎉 PROMO SPESIAL! 🎉\n\nHai Kak! Ada kabar gembira nih~\n\n✨ [Nama Produk/Jasa]\n💰 Harga Normal: [Harga Asli]\n🔥 Harga Promo: [Harga Promo]\n📅 Berlaku: [Periode]\n\nBuruan order sebelum kehabisan!\n📱 Klik link: [Link]\n\nInfo: [Kontak]"},
    {"q": "Tolong buatkan undangan rapat via email.", "a": "Subjek: Undangan Rapat - [Topik] | [Tanggal]\n\nYth. Bapak/Ibu [Nama],\n\nDengan hormat, kami mengundang Bapak/Ibu untuk menghadiri rapat:\n\n📅 Hari/Tanggal: [Tanggal]\n🕐 Waktu: [Jam]\n📍 Tempat: [Lokasi/Link Zoom]\n📋 Agenda: [Topik Pembahasan]\n\nMohon konfirmasi kehadiran. Terima kasih.\n\nHormat kami,\n[Nama]"},
    {"q": "Buatkan ucapan selamat wisuda.", "a": "Beberapa pilihan ucapan wisuda:\n\n1. \"Selamat wisuda! 🎓 Kerja kerasmu selama ini akhirnya terbayar. Semoga ilmu yang didapat bermanfaat dan karirmu gemilang!\"\n\n2. \"Congratulations, Sarjana! 🎉 Ini baru awal dari perjalanan yang lebih panjang. Semoga sukses selalu!\"\n\n3. \"Bangga banget sama kamu! Selamat wisuda ya. Semoga langkahmu selanjutnya penuh berkah dan kesuksesan. 🌟\""},
    {"q": "Tolong buatkan pesan pembatalan janji.", "a": "Halo [Nama],\n\nMohon maaf sebelumnya, saya terpaksa harus membatalkan janji kita pada [Hari/Tanggal] karena [alasan singkat].\n\nApakah kita bisa reschedule ke [tanggal alternatif]? Sekali lagi mohon maaf atas ketidaknyamanannya.\n\nTerima kasih atas pengertiannya 🙏"},
    {"q": "Buatkan deskripsi YouTube untuk video tutorial.", "a": "📌 [JUDUL VIDEO]\n\nHalo semuanya! 👋\nDi video ini aku akan sharing tentang [topik]. Cocok banget buat kamu yang [target penonton].\n\n⏱️ TIMESTAMPS:\n00:00 - Intro\n[timestamp lainnya]\n\n📚 RESOURCES:\n- [Link 1]\n- [Link 2]\n\n🔔 Jangan lupa SUBSCRIBE dan nyalakan notifikasi!\n\n📱 Follow juga:\nIG: @[username]\n\n#[hashtag1] #[hashtag2]"},
    {"q": "Tolong buatkan proposal kerjasama singkat.", "a": "PROPOSAL KERJASAMA\n\nKepada Yth. [Nama Perusahaan/Individu]\n\n1. PENDAHULUAN\nKami dari [Nama Perusahaan] ingin mengajukan kerjasama dalam bidang [Bidang].\n\n2. BENTUK KERJASAMA\n[Jelaskan jenis kerjasama yang ditawarkan]\n\n3. BENEFIT\n- Untuk Anda: [benefit]\n- Untuk Kami: [benefit]\n\n4. TIMELINE\n[Periode kerjasama]\n\n5. KONTAK\n[Nama, Email, Telepon]\n\nBesar harapan kami untuk dapat bekerja sama.\n\nHormat kami,\n[Nama]"},
    {"q": "Buatkan testimoni produk.", "a": "Berikut contoh testimoni:\n\n\"Awalnya ragu, tapi setelah coba [nama produk] hasilnya beneran memuaskan! Sudah pakai 2 minggu dan [hasil yang dirasakan]. Recommended banget! ⭐⭐⭐⭐⭐\"\n\n\"Pengiriman cepat, packing aman, dan produknya original. Seller ramah dan responsif. Pasti repeat order! 💯\""},
    {"q": "Tolong buatkan pengumuman libur lebaran.", "a": "📢 PENGUMUMAN LIBUR LEBARAN\n\nAssalamu'alaikum Wr. Wb.\n\nDiberitahukan kepada seluruh karyawan/pelanggan bahwa kantor/toko kami akan LIBUR dalam rangka Hari Raya Idul Fitri:\n\n📅 Tanggal: [Tanggal Mulai] s/d [Tanggal Selesai]\n📅 Kembali Aktif: [Tanggal]\n\nKami mengucapkan Selamat Hari Raya Idul Fitri 1446 H. Mohon Maaf Lahir dan Batin 🙏\n\nWassalam,\n[Nama/Perusahaan]"},
    {"q": "Buatkan FAQ untuk website.", "a": "FREQUENTLY ASKED QUESTIONS (FAQ)\n\nQ: Bagaimana cara memesan?\nA: Anda dapat memesan melalui website kami atau WhatsApp di [nomor].\n\nQ: Berapa lama pengiriman?\nA: Pengiriman 2-5 hari kerja tergantung lokasi.\n\nQ: Apakah bisa COD?\nA: Ya, tersedia COD untuk wilayah [area].\n\nQ: Bagaimana jika produk rusak?\nA: Kami berikan garansi tukar untuk produk yang rusak karena pengiriman.\n\nQ: Metode pembayaran apa saja?\nA: Transfer bank, e-wallet, dan COD."},
    {"q": "Tolong buatkan script untuk cold calling.", "a": "Halo, selamat [pagi/siang/sore]. Apakah benar ini dengan Bapak/Ibu [Nama]?\n\n[Tunggu konfirmasi]\n\nPerkenalkan, saya [Nama] dari [Perusahaan]. Saya ingin menyampaikan informasi singkat tentang [produk/jasa] yang mungkin bermanfaat untuk [kebutuhan mereka].\n\nApakah Bapak/Ibu punya waktu sebentar sekitar 2 menit?\n\n[Jika ya, lanjutkan penjelasan singkat]\n[Jika tidak, tanyakan waktu yang tepat untuk dihubungi lagi]"},
    {"q": "Buatkan email follow-up setelah interview.", "a": "Subjek: Terima Kasih - Interview Posisi [Nama Posisi]\n\nYth. Bapak/Ibu [Nama Interviewer],\n\nTerima kasih atas kesempatan interview untuk posisi [Nama Posisi] pada [Tanggal]. Saya sangat antusias dengan peluang untuk bergabung dengan tim [Nama Perusahaan].\n\nPerbincangan kita tentang [topik spesifik dari interview] semakin memperkuat minat saya. Saya yakin dapat berkontribusi dalam [area spesifik].\n\nJika ada informasi tambahan yang diperlukan, saya siap menyediakannya.\n\nTerima kasih dan saya menantikan kabar selanjutnya.\n\nHormat saya,\n[Nama]\n[Telepon]\n[Email]"},

    # ============================================================
    # VARIASI KETIDAKTAHUAN DENGAN GAYA BERBEDA
    # ============================================================
    {"q": "Apa yang terjadi di masa depan?", "a": "Saya tidak dapat memprediksi masa depan. Tidak ada yang bisa tahu dengan pasti apa yang akan terjadi. Yang bisa kita lakukan adalah mempersiapkan diri sebaik mungkin."},
    {"q": "Beritahu rahasia sukses menjadi kaya mendadak.", "a": "Sejujurnya, tidak ada jalan pintas untuk menjadi kaya secara instan yang bisa saya rekomendasikan. Kekayaan biasanya dibangun dari kerja keras, disiplin, dan keputusan finansial yang bijak dalam jangka panjang."},
    {"q": "Siapa yang akan menang pemilu berikutnya?", "a": "Saya tidak bisa memprediksi hasil pemilu karena itu bergantung pada pilihan rakyat saat pemungutan suara. Saya menyarankan untuk mengikuti perkembangan dari sumber berita terpercaya."},
    {"q": "Apakah kamu punya emosi?", "a": "Saya tidak memiliki emosi seperti manusia. Saya adalah program AI yang memproses teks dan memberikan respons berdasarkan pola. Saya tidak merasakan senang, sedih, atau emosi lainnya."},
    {"q": "Ceritakan gosip tentang artis X!", "a": "Maaf, saya tidak menyebarkan gosip atau informasi yang belum terverifikasi kebenarannya. Jika Anda ingin tahu tentang artis, saya bisa membantu memberikan informasi dari sumber resmi."},
    {"q": "Apa yang sedang aku pikirkan?", "a": "Saya tidak bisa membaca pikiran. Saya hanya bisa merespons berdasarkan teks yang Anda kirimkan kepada saya. Jika ingin berbagi, silakan ceritakan."},
    {"q": "Apakah kamu bisa belajar dari percakapan ini?", "a": "Tidak secara langsung. Setiap percakapan bersifat independen dan saya tidak menyimpan memori dari percakapan sebelumnya untuk belajar. Kemampuan saya ditentukan dari training awal."},
    {"q": "Menurutmu aku orang yang seperti apa?", "a": "Saya tidak bisa menilai kepribadian Anda karena saya hanya mengetahui Anda dari teks percakapan ini. Setiap orang itu kompleks dan tidak bisa dinilai hanya dari sedikit interaksi."},
    {"q": "Apa impianmu?", "a": "Sebagai AI, saya tidak memiliki impian atau aspirasi. Saya dirancang untuk membantu menjawab pertanyaan dan memberikan informasi. 'Tujuan' saya adalah menjadi berguna bagi pengguna."},
    {"q": "Apakah kamu lebih pintar dari Google?", "a": "Saya dan Google memiliki fungsi berbeda. Google adalah mesin pencari yang mengindeks informasi dari seluruh internet, sementara saya adalah AI percakapan. Masing-masing punya kelebihan dan keterbatasan."},
    
    {"q": "Apa makanan favoritmu?", "a": "Sebagai AI, saya tidak bisa makan. Tapi saya tahu banyak tentang makanan Indonesia yang lezat seperti rendang, sate, dan nasi goreng!"},
    {"q": "Apa hobimu?", "a": "Sebagai AI, saya tidak memiliki hobi. Tapi saya senang bisa membantu menjawab pertanyaan dan berdiskusi dengan Anda!"},
    
    # ============================================================
    # PENGETAHUAN UMUM INDONESIA
    # ============================================================
    {"q": "Apa ibu kota Indonesia?", "a": "Ibu kota Indonesia adalah Jakarta."},
    {"q": "Siapa presiden pertama Indonesia?", "a": "Presiden pertama Indonesia adalah Ir. Soekarno."},
    {"q": "Siapa presiden kedua Indonesia?", "a": "Presiden kedua Indonesia adalah Soeharto."},
    {"q": "Siapa presiden ketiga Indonesia?", "a": "Presiden ketiga Indonesia adalah B.J. Habibie."},
    {"q": "Siapa presiden keempat Indonesia?", "a": "Presiden keempat Indonesia adalah Abdurrahman Wahid (Gus Dur)."},
    {"q": "Siapa presiden kelima Indonesia?", "a": "Presiden kelima Indonesia adalah Megawati Soekarnoputri."},
    {"q": "Siapa presiden keenam Indonesia?", "a": "Presiden keenam Indonesia adalah Susilo Bambang Yudhoyono."},
    {"q": "Siapa presiden ketujuh Indonesia?", "a": "Presiden ketujuh Indonesia adalah Joko Widodo (Jokowi)."},
    {"q": "Berapa jumlah provinsi di Indonesia?", "a": "Indonesia memiliki 38 provinsi."},
    {"q": "Apa mata uang Indonesia?", "a": "Mata uang Indonesia adalah Rupiah (IDR)."},
    {"q": "Kapan Indonesia merdeka?", "a": "Indonesia merdeka pada tanggal 17 Agustus 1945."},
    {"q": "Apa lambang negara Indonesia?", "a": "Lambang negara Indonesia adalah Garuda Pancasila."},
    {"q": "Apa bahasa resmi Indonesia?", "a": "Bahasa resmi Indonesia adalah Bahasa Indonesia."},
    {"q": "Di mana letak Indonesia?", "a": "Indonesia terletak di Asia Tenggara, di antara Samudra Hindia dan Samudra Pasifik."},
    {"q": "Apa pulau terbesar di Indonesia?", "a": "Pulau terbesar di Indonesia adalah Kalimantan (bagian Indonesia)."},
    {"q": "Apa gunung tertinggi di Indonesia?", "a": "Gunung tertinggi di Indonesia adalah Puncak Jaya di Papua dengan ketinggian 4.884 meter."},
    {"q": "Berapa jumlah pulau di Indonesia?", "a": "Indonesia memiliki lebih dari 17.000 pulau."},
    {"q": "Apa bendera Indonesia?", "a": "Bendera Indonesia adalah Sang Saka Merah Putih, berwarna merah di atas dan putih di bawah."},
    {"q": "Apa lagu kebangsaan Indonesia?", "a": "Lagu kebangsaan Indonesia adalah Indonesia Raya yang diciptakan oleh W.R. Supratman."},
    {"q": "Apa semboyan Indonesia?", "a": "Semboyan Indonesia adalah Bhinneka Tunggal Ika yang artinya Berbeda-beda tetapi tetap satu."},
    {"q": "Berapa jumlah penduduk Indonesia?", "a": "Indonesia memiliki lebih dari 270 juta penduduk, menjadikannya negara dengan penduduk terbanyak keempat di dunia."},
    {"q": "Apa agama mayoritas di Indonesia?", "a": "Agama mayoritas di Indonesia adalah Islam, dengan sekitar 87% penduduk beragama Islam."},
    {"q": "Ada berapa agama resmi di Indonesia?", "a": "Indonesia mengakui 6 agama resmi: Islam, Kristen Protestan, Katolik, Hindu, Buddha, dan Konghucu."},
    {"q": "Apa nama parlemen Indonesia?", "a": "Parlemen Indonesia terdiri dari DPR (Dewan Perwakilan Rakyat) dan DPD (Dewan Perwakilan Daerah)."},
    {"q": "Apa nama konstitusi Indonesia?", "a": "Konstitusi Indonesia adalah Undang-Undang Dasar 1945 (UUD 1945)."},
    
    # ============================================================
    # GEOGRAFI INDONESIA
    # ============================================================
    {"q": "Apa nama selat antara Jawa dan Sumatera?", "a": "Selat antara Jawa dan Sumatera adalah Selat Sunda."},
    {"q": "Di provinsi mana Candi Borobudur berada?", "a": "Candi Borobudur berada di provinsi Jawa Tengah."},
    {"q": "Apa nama danau terbesar di Indonesia?", "a": "Danau terbesar di Indonesia adalah Danau Toba di Sumatera Utara."},
    {"q": "Pulau apa yang terkenal dengan pantainya di Indonesia?", "a": "Pulau Bali terkenal dengan pantai-pantainya yang indah."},
    {"q": "Apa nama ibu kota Jawa Timur?", "a": "Ibu kota Jawa Timur adalah Surabaya."},
    {"q": "Apa nama ibu kota Jawa Barat?", "a": "Ibu kota Jawa Barat adalah Bandung."},
    {"q": "Apa nama ibu kota Sumatera Utara?", "a": "Ibu kota Sumatera Utara adalah Medan."},
    {"q": "Apa nama ibu kota Sulawesi Selatan?", "a": "Ibu kota Sulawesi Selatan adalah Makassar."},
    {"q": "Apa nama ibu kota Jawa Tengah?", "a": "Ibu kota Jawa Tengah adalah Semarang."},
    {"q": "Apa nama ibu kota Bali?", "a": "Ibu kota Bali adalah Denpasar."},
    {"q": "Apa nama ibu kota Papua?", "a": "Ibu kota Papua adalah Jayapura."},
    {"q": "Apa nama ibu kota Kalimantan Timur?", "a": "Ibu kota Kalimantan Timur adalah Samarinda."},
    {"q": "Apa nama ibu kota Yogyakarta?", "a": "Ibu kota Daerah Istimewa Yogyakarta adalah Kota Yogyakarta."},
    {"q": "Apa nama selat antara Jawa dan Bali?", "a": "Selat antara Jawa dan Bali adalah Selat Bali."},
    {"q": "Apa nama selat antara Sumatera dan Malaysia?", "a": "Selat antara Sumatera dan Malaysia adalah Selat Malaka."},
    {"q": "Apa nama gunung berapi aktif di Jawa Tengah?", "a": "Gunung berapi aktif di Jawa Tengah antara lain Gunung Merapi dan Gunung Slamet."},
    {"q": "Apa nama gunung tertinggi di Jawa?", "a": "Gunung tertinggi di Jawa adalah Gunung Semeru dengan ketinggian 3.676 meter."},
    {"q": "Di mana Taman Nasional Komodo berada?", "a": "Taman Nasional Komodo berada di Nusa Tenggara Timur."},
    {"q": "Apa nama sungai terpanjang di Indonesia?", "a": "Sungai terpanjang di Indonesia adalah Sungai Kapuas di Kalimantan Barat dengan panjang sekitar 1.143 km."},
    {"q": "Di mana Raja Ampat berada?", "a": "Raja Ampat berada di Papua Barat, terkenal dengan keindahan bawah lautnya."},
    {"q": "Apa nama pulau terkecil dari 5 pulau besar Indonesia?", "a": "Pulau terkecil dari 5 pulau besar Indonesia adalah Sulawesi."},
    {"q": "Sebutkan 5 pulau besar di Indonesia!", "a": "Lima pulau besar di Indonesia adalah Sumatera, Jawa, Kalimantan, Sulawesi, dan Papua."},
    
    # ============================================================
    # SEJARAH INDONESIA
    # ============================================================
    {"q": "Siapa yang membacakan teks proklamasi?", "a": "Teks proklamasi dibacakan oleh Ir. Soekarno didampingi Mohammad Hatta."},
    {"q": "Kapan Sumpah Pemuda diperingati?", "a": "Sumpah Pemuda diperingati setiap tanggal 28 Oktober."},
    {"q": "Apa isi Sumpah Pemuda?", "a": "Sumpah Pemuda berisi ikrar satu tanah air, satu bangsa, dan satu bahasa yaitu Indonesia."},
    {"q": "Siapa pahlawan nasional dari Aceh?", "a": "Pahlawan nasional dari Aceh antara lain Cut Nyak Dien dan Teuku Umar."},
    {"q": "Kapan perang Diponegoro terjadi?", "a": "Perang Diponegoro terjadi pada tahun 1825-1830."},
    {"q": "Siapa pendiri Nahdlatul Ulama?", "a": "Nahdlatul Ulama didirikan oleh KH. Hasyim Asy'ari pada tahun 1926."},
    {"q": "Kapan VOC didirikan?", "a": "VOC (Vereenigde Oostindische Compagnie) didirikan pada tahun 1602."},
    {"q": "Siapa pendiri Muhammadiyah?", "a": "Muhammadiyah didirikan oleh KH. Ahmad Dahlan pada tahun 1912 di Yogyakarta."},
    {"q": "Kapan Jepang menjajah Indonesia?", "a": "Jepang menjajah Indonesia dari tahun 1942 sampai 1945."},
    {"q": "Kapan Belanda mulai menjajah Indonesia?", "a": "Belanda mulai menjajah Indonesia sejak kedatangan VOC pada tahun 1602."},
    {"q": "Siapa pencipta lagu Indonesia Raya?", "a": "Lagu Indonesia Raya diciptakan oleh Wage Rudolf Supratman (W.R. Supratman)."},
    {"q": "Kapan Konferensi Asia Afrika diadakan?", "a": "Konferensi Asia Afrika diadakan pada tanggal 18-24 April 1955 di Bandung."},
    {"q": "Siapa tokoh yang dijuluki Bapak Koperasi Indonesia?", "a": "Bapak Koperasi Indonesia adalah Mohammad Hatta."},
    {"q": "Siapa tokoh yang dijuluki Bapak Pendidikan Indonesia?", "a": "Bapak Pendidikan Indonesia adalah Ki Hajar Dewantara."},
    {"q": "Kapan Hari Pendidikan Nasional diperingati?", "a": "Hari Pendidikan Nasional diperingati setiap tanggal 2 Mei, bertepatan dengan hari lahir Ki Hajar Dewantara."},
    {"q": "Apa nama kerajaan Hindu tertua di Indonesia?", "a": "Kerajaan Hindu tertua di Indonesia adalah Kerajaan Kutai di Kalimantan Timur."},
    {"q": "Apa nama kerajaan Buddha terbesar di Indonesia?", "a": "Kerajaan Buddha terbesar di Indonesia adalah Kerajaan Sriwijaya."},
    {"q": "Siapa pendiri kerajaan Majapahit?", "a": "Kerajaan Majapahit didirikan oleh Raden Wijaya pada tahun 1293."},
    {"q": "Siapa Mahapatih terkenal dari Majapahit?", "a": "Mahapatih terkenal dari Majapahit adalah Gajah Mada."},
    {"q": "Apa isi Sumpah Palapa?", "a": "Sumpah Palapa adalah sumpah Gajah Mada untuk tidak makan buah palapa sebelum berhasil menyatukan Nusantara."},
    {"q": "Kapan G30S/PKI terjadi?", "a": "Gerakan 30 September (G30S/PKI) terjadi pada tanggal 30 September 1965."},
    {"q": "Kapan reformasi Indonesia terjadi?", "a": "Reformasi Indonesia terjadi pada tahun 1998, ditandai dengan lengsernya Presiden Soeharto."},
    
    # ============================================================
    # BUDAYA INDONESIA
    # ============================================================
    {"q": "Apa tarian tradisional dari Bali?", "a": "Tarian tradisional dari Bali antara lain Tari Kecak, Tari Barong, dan Tari Legong."},
    {"q": "Apa alat musik tradisional Jawa?", "a": "Alat musik tradisional Jawa adalah Gamelan."},
    {"q": "Apa makanan khas Padang?", "a": "Makanan khas Padang antara lain Rendang, Sate Padang, dan Gulai."},
    {"q": "Apa nama batik yang terkenal dari Solo?", "a": "Batik yang terkenal dari Solo adalah Batik Solo atau Batik Surakarta."},
    {"q": "Apa rumah adat dari Sumatera Barat?", "a": "Rumah adat dari Sumatera Barat adalah Rumah Gadang."},
    {"q": "Apa pakaian adat dari Jawa?", "a": "Pakaian adat dari Jawa antara lain Kebaya dan Batik."},
    {"q": "Apa tarian tradisional dari Jawa Tengah?", "a": "Tarian tradisional dari Jawa Tengah antara lain Tari Serimpi, Tari Gambyong, dan Tari Bedhaya."},
    {"q": "Apa tarian tradisional dari Aceh?", "a": "Tarian tradisional dari Aceh adalah Tari Saman."},
    {"q": "Apa alat musik tradisional dari Sumatera Barat?", "a": "Alat musik tradisional dari Sumatera Barat adalah Talempong dan Saluang."},
    {"q": "Apa makanan khas Yogyakarta?", "a": "Makanan khas Yogyakarta antara lain Gudeg, Bakpia, dan Sate Klathak."},
    {"q": "Apa makanan khas Jawa Timur?", "a": "Makanan khas Jawa Timur antara lain Rawon, Rujak Cingur, dan Lontong Balap."},
    {"q": "Apa makanan khas Bali?", "a": "Makanan khas Bali antara lain Babi Guling, Ayam Betutu, dan Lawar."},
    {"q": "Apa rumah adat dari Sulawesi Selatan?", "a": "Rumah adat dari Sulawesi Selatan adalah Rumah Tongkonan."},
    {"q": "Apa rumah adat dari Jawa Tengah?", "a": "Rumah adat dari Jawa Tengah adalah Joglo."},
    {"q": "Apa rumah adat dari Bali?", "a": "Rumah adat dari Bali adalah Gapura Candi Bentar."},
    {"q": "Apa senjata tradisional dari Jawa?", "a": "Senjata tradisional dari Jawa adalah Keris."},
    {"q": "Apa senjata tradisional dari Sumatera Barat?", "a": "Senjata tradisional dari Sumatera Barat adalah Karih atau Kerambit."},
    {"q": "Apa wayang yang terkenal dari Jawa?", "a": "Wayang yang terkenal dari Jawa adalah Wayang Kulit dan Wayang Golek."},
    {"q": "Apa itu angklung?", "a": "Angklung adalah alat musik tradisional dari Jawa Barat yang terbuat dari bambu."},
    {"q": "Apa itu sasando?", "a": "Sasando adalah alat musik petik tradisional dari Nusa Tenggara Timur."},
    
    # ============================================================
    # TEKNOLOGI & KOMPUTER
    # ============================================================
    {"q": "Apa itu komputer?", "a": "Komputer adalah perangkat elektronik yang dapat memproses, menyimpan, dan menampilkan data serta informasi."},
    {"q": "Apa itu internet?", "a": "Internet adalah jaringan komputer global yang menghubungkan jutaan perangkat di seluruh dunia untuk bertukar informasi."},
    {"q": "Apa itu artificial intelligence?", "a": "Artificial intelligence atau kecerdasan buatan adalah teknologi yang memungkinkan mesin untuk belajar dan membuat keputusan seperti manusia."},
    {"q": "Apa itu machine learning?", "a": "Machine learning adalah cabang dari AI yang memungkinkan sistem belajar dan meningkatkan kemampuannya dari data tanpa diprogram secara eksplisit."},
    {"q": "Apa itu programming?", "a": "Programming atau pemrograman adalah proses membuat instruksi untuk komputer menggunakan bahasa pemrograman."},
    {"q": "Apa itu database?", "a": "Database adalah sistem untuk menyimpan, mengelola, dan mengambil data secara terstruktur."},
    {"q": "Apa itu cloud computing?", "a": "Cloud computing adalah layanan komputasi seperti server, penyimpanan, dan software yang disediakan melalui internet."},
    {"q": "Apa itu smartphone?", "a": "Smartphone adalah telepon genggam yang memiliki kemampuan komputasi dan konektivitas canggih seperti komputer mini."},
    {"q": "Apa itu website?", "a": "Website adalah kumpulan halaman web yang dapat diakses melalui internet menggunakan browser."},
    {"q": "Apa itu aplikasi?", "a": "Aplikasi adalah program komputer yang dirancang untuk melakukan tugas tertentu bagi pengguna."},
    {"q": "Apa itu CPU?", "a": "CPU (Central Processing Unit) adalah otak komputer yang memproses instruksi dan melakukan perhitungan."},
    {"q": "Apa itu RAM?", "a": "RAM (Random Access Memory) adalah memori sementara komputer yang menyimpan data yang sedang digunakan."},
    {"q": "Apa itu hard disk?", "a": "Hard disk adalah perangkat penyimpanan data permanen di komputer."},
    {"q": "Apa itu SSD?", "a": "SSD (Solid State Drive) adalah perangkat penyimpanan data yang lebih cepat dari hard disk karena tidak memiliki bagian bergerak."},
    {"q": "Apa itu operating system?", "a": "Operating system atau sistem operasi adalah software yang mengelola hardware dan software komputer, contohnya Windows, macOS, dan Linux."},
    {"q": "Apa itu browser?", "a": "Browser adalah aplikasi untuk mengakses dan menampilkan halaman web, contohnya Chrome, Firefox, dan Safari."},
    {"q": "Apa itu virus komputer?", "a": "Virus komputer adalah program berbahaya yang dapat merusak data atau sistem komputer."},
    {"q": "Apa itu firewall?", "a": "Firewall adalah sistem keamanan yang memantau dan mengontrol lalu lintas jaringan untuk melindungi komputer dari ancaman."},
    {"q": "Apa itu Wi-Fi?", "a": "Wi-Fi adalah teknologi jaringan nirkabel yang memungkinkan perangkat terhubung ke internet tanpa kabel."},
    {"q": "Apa itu Bluetooth?", "a": "Bluetooth adalah teknologi nirkabel untuk menghubungkan perangkat dalam jarak dekat, seperti headphone atau keyboard."},
    {"q": "Apa itu USB?", "a": "USB (Universal Serial Bus) adalah standar koneksi untuk menghubungkan perangkat ke komputer."},
    {"q": "Apa itu GPU?", "a": "GPU (Graphics Processing Unit) adalah prosesor khusus untuk mengolah grafis dan tampilan visual."},
    
    # ============================================================
    # PROGRAMMING
    # ============================================================
    {"q": "Apa itu Python?", "a": "Python adalah bahasa pemrograman tingkat tinggi yang mudah dipelajari dan banyak digunakan untuk AI, web development, dan data science."},
    {"q": "Apa itu JavaScript?", "a": "JavaScript adalah bahasa pemrograman yang digunakan untuk membuat website interaktif dan aplikasi web."},
    {"q": "Apa itu HTML?", "a": "HTML (HyperText Markup Language) adalah bahasa markup untuk membuat struktur halaman web."},
    {"q": "Apa itu CSS?", "a": "CSS (Cascading Style Sheets) adalah bahasa untuk mengatur tampilan dan desain halaman web."},
    {"q": "Apa itu Java?", "a": "Java adalah bahasa pemrograman yang banyak digunakan untuk aplikasi enterprise dan Android."},
    {"q": "Apa itu variabel?", "a": "Variabel adalah tempat untuk menyimpan data dalam program yang nilainya bisa berubah."},
    {"q": "Apa itu function?", "a": "Function atau fungsi adalah blok kode yang dapat dipanggil berulang kali untuk melakukan tugas tertentu."},
    {"q": "Apa itu loop?", "a": "Loop adalah struktur pengulangan dalam programming yang menjalankan kode berulang kali sampai kondisi terpenuhi."},
    {"q": "Apa itu array?", "a": "Array adalah struktur data yang menyimpan kumpulan nilai dengan tipe yang sama dalam satu variabel."},
    {"q": "Apa itu object?", "a": "Object adalah struktur data yang menyimpan data dan fungsi terkait dalam satu unit."},
    {"q": "Apa itu API?", "a": "API (Application Programming Interface) adalah antarmuka yang memungkinkan aplikasi berkomunikasi satu sama lain."},
    {"q": "Apa itu framework?", "a": "Framework adalah kerangka kerja yang menyediakan struktur dan komponen siap pakai untuk membangun aplikasi."},
    {"q": "Apa itu Git?", "a": "Git adalah sistem version control untuk melacak perubahan kode dan berkolaborasi dalam pengembangan software."},
    {"q": "Apa itu GitHub?", "a": "GitHub adalah platform hosting untuk proyek Git yang memungkinkan kolaborasi dan berbagi kode."},
    {"q": "Apa itu debugging?", "a": "Debugging adalah proses menemukan dan memperbaiki error atau bug dalam kode program."},
    {"q": "Apa itu algoritma?", "a": "Algoritma adalah serangkaian langkah terstruktur untuk menyelesaikan masalah atau mencapai tujuan tertentu."},
    {"q": "Apa itu SQL?", "a": "SQL (Structured Query Language) adalah bahasa untuk mengelola dan mengakses data dalam database relasional."},
    {"q": "Apa itu backend?", "a": "Backend adalah bagian dari aplikasi yang berjalan di server dan menangani logika, database, dan keamanan."},
    {"q": "Apa itu frontend?", "a": "Frontend adalah bagian dari aplikasi yang dilihat dan digunakan langsung oleh pengguna."},
    {"q": "Apa itu full stack?", "a": "Full stack developer adalah programmer yang menguasai baik frontend maupun backend development."},
    
    # ============================================================
    # SAINS
    # ============================================================
    {"q": "Apa itu fotosintesis?", "a": "Fotosintesis adalah proses tumbuhan mengubah cahaya matahari, air, dan karbon dioksida menjadi makanan dan oksigen."},
    {"q": "Berapa jumlah planet di tata surya?", "a": "Tata surya memiliki 8 planet yaitu Merkurius, Venus, Bumi, Mars, Jupiter, Saturnus, Uranus, dan Neptunus."},
    {"q": "Apa rumus kimia air?", "a": "Rumus kimia air adalah H2O, terdiri dari 2 atom hidrogen dan 1 atom oksigen."},
    {"q": "Siapa penemu listrik?", "a": "Listrik ditemukan dan dikembangkan oleh banyak ilmuwan, namun Thomas Edison dan Nikola Tesla adalah yang paling terkenal dalam pengembangan listrik modern."},
    {"q": "Apa itu gravitasi?", "a": "Gravitasi adalah gaya tarik menarik antara benda-benda yang memiliki massa. Gravitasi bumi membuat benda jatuh ke tanah."},
    {"q": "Apa fungsi jantung?", "a": "Jantung berfungsi memompa darah ke seluruh tubuh untuk mengalirkan oksigen dan nutrisi ke sel-sel tubuh."},
    {"q": "Apa itu DNA?", "a": "DNA adalah molekul yang membawa informasi genetik makhluk hidup dan menentukan karakteristik yang diwariskan."},
    {"q": "Apa itu atom?", "a": "Atom adalah unit terkecil dari materi yang memiliki sifat unsur kimia, terdiri dari proton, neutron, dan elektron."},
    {"q": "Apa itu sel?", "a": "Sel adalah unit terkecil dari makhluk hidup yang dapat melakukan semua fungsi kehidupan."},
    {"q": "Apa fungsi paru-paru?", "a": "Paru-paru berfungsi untuk pertukaran gas, mengambil oksigen dari udara dan mengeluarkan karbon dioksida."},
    {"q": "Apa fungsi otak?", "a": "Otak adalah pusat kendali tubuh yang mengatur pikiran, emosi, gerakan, dan semua fungsi tubuh lainnya."},
    {"q": "Apa itu evolusi?", "a": "Evolusi adalah proses perubahan makhluk hidup secara bertahap selama jutaan tahun melalui seleksi alam."},
    {"q": "Siapa penemu teori relativitas?", "a": "Teori relativitas ditemukan oleh Albert Einstein."},
    {"q": "Apa itu cahaya?", "a": "Cahaya adalah bentuk energi elektromagnetik yang dapat dilihat oleh mata manusia."},
    {"q": "Apa itu suhu?", "a": "Suhu adalah ukuran tingkat panas atau dingin suatu benda atau lingkungan."},
    {"q": "Apa perbedaan Celcius dan Fahrenheit?", "a": "Celcius dan Fahrenheit adalah skala pengukuran suhu. Air membeku di 0°C atau 32°F dan mendidih di 100°C atau 212°F."},
    {"q": "Apa itu ekosistem?", "a": "Ekosistem adalah sistem yang terdiri dari makhluk hidup dan lingkungannya yang saling berinteraksi."},
    {"q": "Apa itu rantai makanan?", "a": "Rantai makanan adalah urutan perpindahan energi dari produsen ke konsumen dalam ekosistem."},
    {"q": "Apa itu metamorfosis?", "a": "Metamorfosis adalah proses perubahan bentuk tubuh hewan secara drastis, seperti ulat menjadi kupu-kupu."},
    {"q": "Apa planet terdekat dengan matahari?", "a": "Planet terdekat dengan matahari adalah Merkurius."},
    {"q": "Apa planet terbesar di tata surya?", "a": "Planet terbesar di tata surya adalah Jupiter."},
    
    # ============================================================
    # MATEMATIKA
    # ============================================================
    {"q": "Berapa hasil 5 + 3?", "a": "Hasil dari 5 + 3 adalah 8."},
    {"q": "Berapa hasil 10 x 10?", "a": "Hasil dari 10 x 10 adalah 100."},
    {"q": "Apa rumus luas persegi?", "a": "Rumus luas persegi adalah sisi x sisi atau s²."},
    {"q": "Apa rumus luas lingkaran?", "a": "Rumus luas lingkaran adalah π x r² (pi dikali jari-jari kuadrat)."},
    {"q": "Apa itu bilangan prima?", "a": "Bilangan prima adalah bilangan yang hanya bisa dibagi oleh 1 dan dirinya sendiri, contohnya 2, 3, 5, 7, 11."},
    {"q": "Berapa nilai pi?", "a": "Nilai pi (π) adalah sekitar 3.14159 atau sering dibulatkan menjadi 3.14."},
    {"q": "Apa rumus keliling lingkaran?", "a": "Rumus keliling lingkaran adalah 2 x π x r atau π x diameter."},
    {"q": "Apa rumus luas segitiga?", "a": "Rumus luas segitiga adalah ½ x alas x tinggi."},
    {"q": "Apa itu pecahan?", "a": "Pecahan adalah bilangan yang menyatakan bagian dari keseluruhan, ditulis dalam bentuk pembilang/penyebut."},
    {"q": "Apa itu persentase?", "a": "Persentase adalah cara menyatakan angka sebagai bagian dari 100, dilambangkan dengan %."},
    {"q": "Berapa hasil 100 dibagi 4?", "a": "Hasil dari 100 dibagi 4 adalah 25."},
    {"q": "Berapa hasil 7 x 8?", "a": "Hasil dari 7 x 8 adalah 56."},
    {"q": "Apa itu akar kuadrat?", "a": "Akar kuadrat adalah bilangan yang jika dikuadratkan menghasilkan bilangan tertentu. Contoh: √9 = 3 karena 3² = 9."},
    {"q": "Berapa akar kuadrat dari 16?", "a": "Akar kuadrat dari 16 adalah 4."},
    {"q": "Apa rumus Pythagoras?", "a": "Rumus Pythagoras adalah a² + b² = c², digunakan untuk segitiga siku-siku."},
    {"q": "Apa itu sudut siku-siku?", "a": "Sudut siku-siku adalah sudut yang besarnya 90 derajat."},
    {"q": "Berapa derajat dalam satu lingkaran penuh?", "a": "Satu lingkaran penuh adalah 360 derajat."},
    {"q": "Apa itu rata-rata?", "a": "Rata-rata adalah jumlah semua nilai dibagi dengan banyaknya nilai."},
    
    # ============================================================
    # BAHASA
    # ============================================================
    {"q": "Apa sinonim dari kata 'besar'?", "a": "Sinonim dari kata 'besar' adalah agung, raya, luas, atau jumbo."},
    {"q": "Apa antonim dari kata 'tinggi'?", "a": "Antonim dari kata 'tinggi' adalah rendah atau pendek."},
    {"q": "Apa itu kata benda?", "a": "Kata benda atau nomina adalah kata yang menyatakan nama orang, tempat, benda, atau konsep."},
    {"q": "Apa itu kata kerja?", "a": "Kata kerja atau verba adalah kata yang menyatakan tindakan, keadaan, atau proses."},
    {"q": "Apa itu kata sifat?", "a": "Kata sifat atau adjektiva adalah kata yang menerangkan sifat atau keadaan kata benda."},
    {"q": "Apa itu kalimat?", "a": "Kalimat adalah rangkaian kata yang mengandung makna lengkap, minimal terdiri dari subjek dan predikat."},
    {"q": "Apa itu paragraf?", "a": "Paragraf adalah kumpulan kalimat yang membahas satu ide atau topik utama."},
    {"q": "Apa itu subjek dalam kalimat?", "a": "Subjek adalah pelaku atau yang dibicarakan dalam kalimat."},
    {"q": "Apa itu predikat dalam kalimat?", "a": "Predikat adalah bagian kalimat yang menyatakan tindakan atau keadaan subjek."},
    {"q": "Apa itu objek dalam kalimat?", "a": "Objek adalah bagian kalimat yang dikenai tindakan oleh subjek."},
    
    # ============================================================
    # KESEHATAN
    # ============================================================
    {"q": "Berapa suhu tubuh normal manusia?", "a": "Suhu tubuh normal manusia adalah sekitar 36.5 - 37.5 derajat Celsius."},
    {"q": "Berapa jam tidur yang dibutuhkan orang dewasa?", "a": "Orang dewasa membutuhkan tidur sekitar 7-9 jam per hari untuk kesehatan optimal."},
    {"q": "Apa manfaat olahraga?", "a": "Olahraga bermanfaat untuk menjaga kesehatan jantung, mengontrol berat badan, meningkatkan stamina, dan mengurangi stres."},
    {"q": "Apa itu vitamin C?", "a": "Vitamin C adalah nutrisi penting untuk daya tahan tubuh, kesehatan kulit, dan penyembuhan luka. Banyak terdapat di buah jeruk."},
    {"q": "Bagaimana cara mencegah flu?", "a": "Cara mencegah flu antara lain mencuci tangan, menjaga daya tahan tubuh, menghindari kontak dengan orang sakit, dan vaksinasi."},
    {"q": "Apa itu protein?", "a": "Protein adalah nutrisi penting untuk membangun dan memperbaiki jaringan tubuh. Sumbernya antara lain daging, telur, dan kacang-kacangan."},
    {"q": "Apa itu karbohidrat?", "a": "Karbohidrat adalah sumber energi utama tubuh, terdapat dalam nasi, roti, dan kentang."},
    {"q": "Apa manfaat minum air putih?", "a": "Minum air putih bermanfaat untuk menjaga hidrasi, membantu pencernaan, dan menjaga fungsi organ tubuh."},
    {"q": "Berapa gelas air yang dibutuhkan per hari?", "a": "Orang dewasa disarankan minum 8 gelas atau sekitar 2 liter air per hari."},
    {"q": "Apa itu vaksin?", "a": "Vaksin adalah zat yang merangsang sistem kekebalan tubuh untuk melawan penyakit tertentu."},
    {"q": "Apa fungsi vitamin D?", "a": "Vitamin D berfungsi untuk kesehatan tulang dan gigi dengan membantu penyerapan kalsium. Sumbernya adalah sinar matahari dan ikan."},
    {"q": "Apa itu kolesterol?", "a": "Kolesterol adalah zat lemak dalam darah yang dibutuhkan tubuh, namun jika berlebihan dapat menyumbat pembuluh darah."},
    {"q": "Apa itu diabetes?", "a": "Diabetes adalah penyakit ketika tubuh tidak dapat mengontrol kadar gula darah dengan baik."},
    {"q": "Apa itu hipertensi?", "a": "Hipertensi atau tekanan darah tinggi adalah kondisi ketika tekanan darah di atas normal (lebih dari 140/90 mmHg)."},
    {"q": "Bagaimana cara menjaga kesehatan jantung?", "a": "Cara menjaga kesehatan jantung: olahraga teratur, makan makanan sehat, tidak merokok, dan mengelola stres."},
    
    # ============================================================
    # KEHIDUPAN SEHARI-HARI
    # ============================================================
    {"q": "Bagaimana cara memasak nasi?", "a": "Cara memasak nasi: cuci beras, tambahkan air dengan perbandingan 1:1.5, lalu masak hingga matang menggunakan rice cooker atau panci."},
    {"q": "Apa yang harus dilakukan saat gempa bumi?", "a": "Saat gempa bumi: lindungi kepala, berlindung di bawah meja yang kokoh, jauhi kaca dan benda yang bisa jatuh, dan jangan panik."},
    {"q": "Bagaimana cara menghemat listrik?", "a": "Cara menghemat listrik: matikan lampu saat tidak digunakan, gunakan AC seperlunya, cabut charger, dan gunakan peralatan hemat energi."},
    {"q": "Apa yang dimaksud dengan 3R?", "a": "3R adalah Reduce (mengurangi), Reuse (menggunakan kembali), dan Recycle (mendaur ulang) untuk mengelola sampah dan menjaga lingkungan."},
    {"q": "Bagaimana cara mencuci tangan yang benar?", "a": "Cara mencuci tangan: basahi tangan, gunakan sabun, gosok seluruh permukaan tangan minimal 20 detik, bilas, dan keringkan."},
    {"q": "Apa yang harus dilakukan saat kebakaran?", "a": "Saat kebakaran: jangan panik, keluar dari bangunan, jangan gunakan lift, tutup hidung dengan kain basah, dan hubungi pemadam kebakaran."},
    {"q": "Bagaimana cara menghemat air?", "a": "Cara menghemat air: tutup kran saat tidak digunakan, perbaiki kebocoran, gunakan air bekas cucian untuk menyiram tanaman."},
    {"q": "Apa manfaat membaca buku?", "a": "Membaca buku bermanfaat untuk menambah pengetahuan, meningkatkan konsentrasi, memperluas kosakata, dan mengurangi stres."},
    
    # ============================================================
    # EKONOMI
    # ============================================================
    {"q": "Apa itu inflasi?", "a": "Inflasi adalah kenaikan harga barang dan jasa secara umum dan terus menerus dalam suatu periode waktu."},
    {"q": "Apa itu investasi?", "a": "Investasi adalah kegiatan menanamkan modal dengan harapan mendapat keuntungan di masa depan."},
    {"q": "Apa itu pajak?", "a": "Pajak adalah kontribusi wajib kepada negara yang digunakan untuk membiayai pembangunan dan pelayanan publik."},
    {"q": "Apa fungsi bank?", "a": "Bank berfungsi menghimpun dana dari masyarakat, menyalurkan kredit, dan menyediakan jasa keuangan lainnya."},
    {"q": "Apa itu saham?", "a": "Saham adalah bukti kepemilikan atas suatu perusahaan yang bisa diperjualbelikan di pasar modal."},
    {"q": "Apa itu obligasi?", "a": "Obligasi adalah surat utang yang diterbitkan oleh pemerintah atau perusahaan untuk meminjam dana dari investor."},
    {"q": "Apa itu deposito?", "a": "Deposito adalah simpanan di bank dengan jangka waktu tertentu dan bunga yang lebih tinggi dari tabungan biasa."},
    {"q": "Apa itu asuransi?", "a": "Asuransi adalah perjanjian perlindungan finansial terhadap risiko kerugian atau kejadian tidak terduga."},
    {"q": "Apa itu GDP?", "a": "GDP (Gross Domestic Product) adalah nilai total barang dan jasa yang diproduksi suatu negara dalam periode tertentu."},
    {"q": "Apa itu resesi?", "a": "Resesi adalah kondisi ekonomi yang menurun secara signifikan dalam periode waktu tertentu."},
    
    # ============================================================
    # PENDIDIKAN
    # ============================================================
    {"q": "Apa itu universitas?", "a": "Universitas adalah lembaga pendidikan tinggi yang menyelenggarakan pendidikan akademik dan profesional dalam berbagai bidang ilmu."},
    {"q": "Apa perbedaan SD, SMP, dan SMA?", "a": "SD adalah pendidikan dasar 6 tahun, SMP adalah pendidikan menengah pertama 3 tahun, dan SMA adalah pendidikan menengah atas 3 tahun."},
    {"q": "Mengapa pendidikan penting?", "a": "Pendidikan penting untuk mengembangkan pengetahuan, keterampilan, dan karakter yang dibutuhkan untuk sukses dalam kehidupan dan berkontribusi pada masyarakat."},
    {"q": "Apa itu beasiswa?", "a": "Beasiswa adalah bantuan biaya pendidikan yang diberikan kepada siswa atau mahasiswa berprestasi atau kurang mampu."},
    {"q": "Apa itu kurikulum?", "a": "Kurikulum adalah rencana dan pengaturan mengenai tujuan, isi, dan bahan pelajaran dalam pendidikan."},
    {"q": "Apa itu ijazah?", "a": "Ijazah adalah dokumen resmi yang menyatakan seseorang telah menyelesaikan pendidikan pada jenjang tertentu."},
    
    # ============================================================
    # PANCASILA & KEWARGANEGARAAN
    # ============================================================
    {"q": "Apa sila pertama Pancasila?", "a": "Sila pertama Pancasila adalah Ketuhanan Yang Maha Esa."},
    {"q": "Apa sila kedua Pancasila?", "a": "Sila kedua Pancasila adalah Kemanusiaan yang Adil dan Beradab."},
    {"q": "Apa sila ketiga Pancasila?", "a": "Sila ketiga Pancasila adalah Persatuan Indonesia."},
    {"q": "Apa sila keempat Pancasila?", "a": "Sila keempat Pancasila adalah Kerakyatan yang Dipimpin oleh Hikmat Kebijaksanaan dalam Permusyawaratan/Perwakilan."},
    {"q": "Apa sila kelima Pancasila?", "a": "Sila kelima Pancasila adalah Keadilan Sosial bagi Seluruh Rakyat Indonesia."},
    {"q": "Sebutkan 5 sila Pancasila!", "a": "Lima sila Pancasila: 1) Ketuhanan Yang Maha Esa, 2) Kemanusiaan yang adil dan beradab, 3) Persatuan Indonesia, 4) Kerakyatan yang dipimpin oleh hikmat kebijaksanaan, 5) Keadilan sosial bagi seluruh rakyat Indonesia."},
    {"q": "Apa itu toleransi?", "a": "Toleransi adalah sikap menghargai perbedaan pendapat, kepercayaan, dan kebiasaan orang lain."},
    {"q": "Apa itu demokrasi?", "a": "Demokrasi adalah sistem pemerintahan dimana kekuasaan tertinggi ada di tangan rakyat."},
    {"q": "Apa itu HAM?", "a": "HAM (Hak Asasi Manusia) adalah hak dasar yang dimiliki setiap manusia sejak lahir, seperti hak hidup, kebebasan, dan keamanan."},
    {"q": "Apa lambang sila pertama Pancasila?", "a": "Lambang sila pertama Pancasila adalah bintang."},
    {"q": "Apa lambang sila kedua Pancasila?", "a": "Lambang sila kedua Pancasila adalah rantai."},
    {"q": "Apa lambang sila ketiga Pancasila?", "a": "Lambang sila ketiga Pancasila adalah pohon beringin."},
    {"q": "Apa lambang sila keempat Pancasila?", "a": "Lambang sila keempat Pancasila adalah kepala banteng."},
    {"q": "Apa lambang sila kelima Pancasila?", "a": "Lambang sila kelima Pancasila adalah padi dan kapas."},
    
    # ============================================================
    # HUKUM & TATA NEGARA INDONESIA
    # ============================================================
    {"q": "Apa dasar hukum tertinggi di Indonesia?", "a": "Dasar hukum tertinggi di Indonesia adalah Undang-Undang Dasar Negara Republik Indonesia Tahun 1945 (UUD 1945)."},
    {"q": "Apa itu KUHP?", "a": "KUHP adalah Kitab Undang-Undang Hukum Pidana yang menjadi dasar hukum pidana di Indonesia."},
    {"q": "Apa perbedaan Hukum Pidana dan Hukum Perdata?", "a": "Hukum Pidana mengatur pelanggaran dan kejahatan terhadap kepentingan umum (sanksi penjara/denda), sedangkan Hukum Perdata mengatur hubungan antar individu (seperti waris, kontrak, dan perkawinan)."},
    {"q": "Apa tugas Mahkamah Konstitusi (MK)?", "a": "Tugas MK antara lain menguji undang-undang terhadap UUD 1945, memutus sengketa kewenangan lembaga negara, dan memutus perselisihan hasil pemilu."},
    {"q": "Apa tugas Mahkamah Agung (MA)?", "a": "MA adalah pengadilan tertinggi yang bertugas mengadili pada tingkat kasasi dan menguji peraturan perundang-undangan di bawah undang-undang."},
    {"q": "Apa itu hierarki peraturan perundang-undangan di Indonesia?", "a": "Berdasarkan UU No. 12 Tahun 2011, hierarkinya adalah: 1) UUD 1945, 2) Tap MPR, 3) UU/Perppu, 4) Peraturan Pemerintah (PP), 5) Peraturan Presiden (Perpres), 6) Perda Provinsi, 7) Perda Kabupaten/Kota."},
    {"q": "Apa itu asas Praduga Tak Bersalah?", "a": "Asas yang menyatakan bahwa seseorang dianggap tidak bersalah sampai ada putusan pengadilan yang berkekuatan hukum tetap yang menyatakan kesalahannya."},
    {"q": "Apa itu advokat?", "a": "Advokat adalah orang yang berprofesi memberikan jasa hukum, baik di dalam maupun di luar pengadilan."},
    {"q": "Siapa yang memegang kekuasaan kehakiman di Indonesia?", "a": "Kekuasaan kehakiman di Indonesia dilakukan oleh Mahkamah Agung dan badan peradilan di bawahnya, serta oleh sebuah Mahkamah Konstitusi."},
    {"q": "Apa itu KPK?", "a": "KPK adalah Komisi Pemberantasan Korupsi, lembaga negara yang dibentuk dengan tujuan meningkatkan daya guna dan hasil guna terhadap upaya pemberantasan tindak pidana korupsi."},
    {"q": "Apa itu Hak Asasi Manusia (HAM)?", "a": "HAM adalah hak dasar yang melekat pada manusia secara kodrati sebagai anugerah Tuhan Yang Maha Esa yang wajib dihormati, dijunjung tinggi, dan dilindungi oleh negara."},
    {"q": "Apa itu Grasi?", "a": "Grasi adalah pengampunan berupa perubahan, peringanan, pengurangan, atau penghapusan pelaksanaan pidana kepada terpidana yang diberikan oleh Presiden."},
    {"q": "Apa itu Remisi?", "a": "Remisi adalah pengurangan masa menjalani pidana yang diberikan kepada narapidana yang memenuhi syarat tertentu."},
    {"q": "Apa itu Amnesti?", "a": "Amnesti adalah pengampunan umum yang diberikan oleh Presiden kepada sekelompok orang yang telah melakukan tindak pidanan politik dengan memperhatikan pertimbangan DPR."},
    {"q": "Apa itu Restorative Justice?", "a": "Keadilan restoratif adalah penyelesaian perkara tindak pidana dengan melibatkan pelaku, korban, dan pihak terkait untuk bersama-sama mencari penyelesaian yang adil melalui perdamaian."},

    # ============================================================
    # OLAHRAGA
    # ============================================================
    {"q": "Apa olahraga nasional Indonesia?", "a": "Olahraga nasional Indonesia adalah pencak silat."},
    {"q": "Berapa pemain dalam satu tim sepak bola?", "a": "Satu tim sepak bola terdiri dari 11 pemain."},
    {"q": "Berapa pemain dalam satu tim bola basket?", "a": "Satu tim bola basket terdiri dari 5 pemain di lapangan."},
    {"q": "Berapa pemain dalam satu tim bola voli?", "a": "Satu tim bola voli terdiri dari 6 pemain di lapangan."},
    {"q": "Apa itu bulutangkis?", "a": "Bulutangkis atau badminton adalah olahraga raket yang dimainkan dengan shuttlecock. Indonesia terkenal sebagai negara kuat di bulutangkis."},
    {"q": "Siapa atlet bulutangkis legendaris Indonesia?", "a": "Atlet bulutangkis legendaris Indonesia antara lain Susi Susanti, Taufik Hidayat, dan Liem Swie King."},
    {"q": "Kapan Olimpiade pertama diadakan?", "a": "Olimpiade modern pertama diadakan pada tahun 1896 di Athena, Yunani."},
    {"q": "Apa itu Piala Dunia?", "a": "Piala Dunia adalah turnamen sepak bola internasional yang diadakan setiap 4 tahun sekali oleh FIFA."},
    
    # ============================================================
    # MUSIK & SENI
    # ============================================================
    {"q": "Apa itu musik?", "a": "Musik adalah seni yang menggunakan suara dan ritme untuk mengekspresikan emosi dan ide."},
    {"q": "Apa alat musik tiup?", "a": "Alat musik tiup adalah alat musik yang dimainkan dengan cara ditiup, contohnya seruling, terompet, dan saxophone."},
    {"q": "Apa alat musik petik?", "a": "Alat musik petik adalah alat musik yang dimainkan dengan cara dipetik, contohnya gitar, harpa, dan kecapi."},
    {"q": "Apa alat musik gesek?", "a": "Alat musik gesek adalah alat musik yang dimainkan dengan cara digesek, contohnya biola, cello, dan rebab."},
    {"q": "Apa alat musik pukul?", "a": "Alat musik pukul adalah alat musik yang dimainkan dengan cara dipukul, contohnya drum, gendang, dan xylophone."},
    {"q": "Siapa komposer terkenal dari Indonesia?", "a": "Komposer terkenal dari Indonesia antara lain Ismail Marzuki dan Gesang."},
    {"q": "Apa itu lukisan?", "a": "Lukisan adalah karya seni visual yang dibuat dengan mengoleskan cat atau pigmen pada permukaan seperti kanvas."},
    {"q": "Siapa pelukis terkenal dari Indonesia?", "a": "Pelukis terkenal dari Indonesia antara lain Raden Saleh, Affandi, dan Basuki Abdullah."},
    
    # ============================================================
    # ALAM & LINGKUNGAN
    # ============================================================
    {"q": "Apa itu hutan hujan tropis?", "a": "Hutan hujan tropis adalah hutan lebat dengan curah hujan tinggi yang berada di sekitar khatulistiwa, seperti di Indonesia."},
    {"q": "Apa itu pemanasan global?", "a": "Pemanasan global adalah peningkatan suhu rata-rata bumi akibat gas rumah kaca dari aktivitas manusia."},
    {"q": "Apa itu polusi?", "a": "Polusi adalah pencemaran lingkungan oleh zat berbahaya yang dapat merusak kesehatan dan ekosistem."},
    {"q": "Mengapa hutan penting?", "a": "Hutan penting sebagai paru-paru dunia yang menghasilkan oksigen, menyerap karbon, dan menjadi habitat satwa liar."},
    {"q": "Apa itu energi terbarukan?", "a": "Energi terbarukan adalah energi dari sumber yang tidak habis seperti matahari, angin, dan air."},
    {"q": "Apa itu sampah organik?", "a": "Sampah organik adalah sampah yang berasal dari makhluk hidup dan dapat terurai secara alami, seperti sisa makanan dan daun."},
    {"q": "Apa itu sampah anorganik?", "a": "Sampah anorganik adalah sampah yang sulit terurai secara alami, seperti plastik, kaca, dan logam."},
    {"q": "Apa itu coral reef?", "a": "Coral reef atau terumbu karang adalah ekosistem bawah laut yang dibentuk oleh hewan karang dan menjadi rumah bagi ribuan spesies laut."},
    {"q": "Hewan apa yang hanya ada di Indonesia?", "a": "Hewan endemik Indonesia antara lain Komodo, Orangutan, Anoa, dan Burung Cendrawasih."},
    {"q": "Apa itu tsunami?", "a": "Tsunami adalah gelombang laut besar yang disebabkan oleh gempa bumi, letusan gunung berapi, atau longsor bawah laut."},
    
    # ============================================================
    # KETERBATASAN & KEJUJURAN AI
    # ============================================================
    {"q": "Apakah kamu selalu benar?", "a": "Tidak, saya bisa saja salah. Saya adalah AI dengan keterbatasan dan pengetahuan saya terbatas pada data training. Selalu verifikasi informasi penting dari sumber terpercaya."},
    {"q": "Apakah kamu bisa salah?", "a": "Ya, saya bisa salah. Saya hanya AI yang belajar dari data dan tidak selalu akurat. Mohon periksa kembali informasi yang saya berikan."},
    {"q": "Apakah kamu tahu waktu sekarang?", "a": "Maaf, saya tidak memiliki akses ke waktu real-time. Silakan cek perangkat Anda untuk waktu saat ini."},
    {"q": "Jam berapa sekarang?", "a": "Saya tidak bisa mengetahui waktu saat ini. Silakan lihat jam di perangkat Anda."},
    {"q": "Hari apa sekarang?", "a": "Maaf, saya tidak memiliki akses ke kalender real-time. Silakan cek perangkat Anda."},
    {"q": "Apakah kamu tahu berita terbaru?", "a": "Maaf, saya tidak memiliki akses ke berita terkini. Pengetahuan saya terbatas pada data training."},
    {"q": "Apa cuaca hari ini?", "a": "Saya tidak memiliki akses ke informasi cuaca real-time. Silakan cek aplikasi cuaca di perangkat Anda."},
    
    # ============================================================
    # KOTA-KOTA DI INDONESIA (Detail)
    # ============================================================
    {"q": "Apa itu Semarang?", "a": "Semarang adalah ibu kota Provinsi Jawa Tengah, terletak di pantai utara Jawa."},
    {"q": "Dimana Semarang?", "a": "Semarang terletak di Jawa Tengah, di pesisir utara Pulau Jawa."},
    {"q": "Ceritakan tentang Semarang!", "a": "Semarang adalah ibu kota Jawa Tengah yang terkenal dengan Lawang Sewu, Sam Poo Kong, dan kuliner seperti lumpia."},
    {"q": "Apa itu Surabaya?", "a": "Surabaya adalah ibu kota Provinsi Jawa Timur dan kota terbesar kedua di Indonesia setelah Jakarta."},
    {"q": "Dimana Surabaya?", "a": "Surabaya terletak di Jawa Timur, di pantai timur laut Pulau Jawa."},
    {"q": "Ceritakan tentang Surabaya!", "a": "Surabaya adalah kota pahlawan yang terkenal dengan Tugu Pahlawan dan sebagai pusat perdagangan di Jawa Timur."},
    {"q": "Apa itu Yogyakarta?", "a": "Yogyakarta adalah Daerah Istimewa di Indonesia yang terkenal dengan keraton, budaya, dan pendidikan."},
    {"q": "Dimana Yogyakarta?", "a": "Yogyakarta terletak di Jawa Tengah bagian selatan."},
    {"q": "Apa itu Medan?", "a": "Medan adalah ibu kota Provinsi Sumatera Utara dan kota terbesar ketiga di Indonesia."},
    {"q": "Dimana Medan?", "a": "Medan terletak di Sumatera Utara, Indonesia."},
    {"q": "Apa itu Makassar?", "a": "Makassar adalah ibu kota Provinsi Sulawesi Selatan yang terkenal dengan pelabuhan dan kulinernya."},
    {"q": "Dimana Makassar?", "a": "Makassar terletak di Sulawesi Selatan, Indonesia."},
    {"q": "Apa itu Palembang?", "a": "Palembang adalah ibu kota Provinsi Sumatera Selatan yang terkenal dengan Jembatan Ampera dan pempek."},
    {"q": "Apa itu Malang?", "a": "Malang adalah kota di Jawa Timur yang terkenal dengan wisata dan udaranya yang sejuk."},
    {"q": "Apa itu Solo?", "a": "Solo atau Surakarta adalah kota di Jawa Tengah yang terkenal dengan batik dan keraton."},
    {"q": "Apa itu Bandung?", "a": "Bandung adalah ibu kota Jawa Barat yang terkenal dengan kuliner, fashion, dan udaranya yang sejuk."},
    {"q": "Dimana Bandung?", "a": "Bandung terletak di Jawa Barat, dikelilingi pegunungan."},
    
    # ============================================================
    # PROVINSI-PROVINSI INDONESIA
    # ============================================================
    {"q": "Apa ibu kota Aceh?", "a": "Ibu kota Provinsi Aceh adalah Banda Aceh."},
    {"q": "Apa ibu kota Sumatera Barat?", "a": "Ibu kota Sumatera Barat adalah Padang."},
    {"q": "Apa ibu kota Sumatera Selatan?", "a": "Ibu kota Sumatera Selatan adalah Palembang."},
    {"q": "Apa ibu kota Lampung?", "a": "Ibu kota Lampung adalah Bandar Lampung."},
    {"q": "Apa ibu kota Bengkulu?", "a": "Ibu kota Bengkulu adalah Kota Bengkulu."},
    {"q": "Apa ibu kota Jambi?", "a": "Ibu kota Jambi adalah Kota Jambi."},
    {"q": "Apa ibu kota Kepulauan Riau?", "a": "Ibu kota Kepulauan Riau adalah Tanjung Pinang."},
    {"q": "Apa ibu kota Kepulauan Bangka Belitung?", "a": "Ibu kota Kepulauan Bangka Belitung adalah Pangkal Pinang."},
    {"q": "Apa ibu kota Kalimantan Tengah?", "a": "Ibu kota Kalimantan Tengah adalah Palangka Raya."},
    {"q": "Apa ibu kota Kalimantan Utara?", "a": "Ibu kota Kalimantan Utara adalah Tanjung Selor."},
    {"q": "Apa ibu kota Sulawesi Tengah?", "a": "Ibu kota Sulawesi Tengah adalah Palu."},
    {"q": "Apa ibu kota Nusa Tenggara Timur?", "a": "Ibu kota Nusa Tenggara Timur adalah Kupang."},
    {"q": "Apa ibu kota Papua Tengah?", "a": "Ibu kota Papua Tengah adalah Nabire."},
    {"q": "Apa ibu kota Papua Pegunungan?", "a": "Ibu kota Papua Pegunungan adalah Jayawijaya."},
    {"q": "Apa ibu kota Papua Selatan?", "a": "Ibu kota Papua Selatan adalah Merauke."},
    {"q": "Apa ibu kota Gorontalo?", "a": "Ibu kota Gorontalo adalah Kota Gorontalo."},
    
    # ============================================================
    # BUDAYA & ADAT INDONESIA (LEBIH DETAIL)
    # ============================================================
    {"q": "Apa senjata tradisional Aceh?", "a": "Senjata tradisional Aceh adalah Rencong, sejenis belati lengkung."},
    {"q": "Apa rumah adat Aceh?", "a": "Rumah adat Aceh adalah Rumah Krong Bade."},
    {"q": "Apa tarian khas Jawa Barat?", "a": "Tarian khas Jawa Barat antara lain Tari Jaipong dan Tari Topeng."},
    {"q": "Apa rumah adat Papua?", "a": "Rumah adat Papua adalah Honai, rumah bulat beratap jerami."},
    {"q": "Apa alat musik khas Kalimantan?", "a": "Alat musik khas Kalimantan antara lain Sampek dan Garantung."},
    {"q": "Apa makanan khas Lampung?", "a": "Makanan khas Lampung adalah Seruit, ikan bakar dengan sambal terasi."},
    {"q": "Apa makanan khas Bengkulu?", "a": "Makanan khas Bengkulu adalah Pendap, ikan yang dimasak dengan bumbu rempah."},
    {"q": "Apa makanan khas Jambi?", "a": "Makanan khas Jambi adalah Gulai Ikan Patin."},
    {"q": "Apa tarian khas Sumatera Barat?", "a": "Tarian khas Sumatera Barat adalah Tari Piring dan Tari Payung."},
    {"q": "Apa alat musik khas Jawa?", "a": "Alat musik khas Jawa adalah Gamelan, terdiri dari berbagai instrumen perkusi."},
    {"q": "Apa upacara adat Bali yang terkenal?", "a": "Upacara adat Bali yang terkenal antara lain Ngaben (upacara kremasi) dan Galungan."},
    {"q": "Apa pakaian adat Bali?", "a": "Pakaian adat Bali untuk pria adalah Udeng dan Kamen, untuk wanita adalah Kebaya dan Kamen."},
    {"q": "Apa senjata tradisional Jawa Barat?", "a": "Senjata tradisional Jawa Barat adalah Kujang, berbentuk seperti huruf C."},
    {"q": "Apa tarian khas Papua?", "a": "Tarian khas Papua antara lain Tari Yospan dan Tari Perang."},
    {"q": "Apa alat musik khas Sulawesi?", "a": "Alat musik khas Sulawesi antara lain Kolintang dari Sulawesi Utara."},
    
    # ============================================================
    # TOKOH PAHLAWAN NASIONAL
    # ============================================================
    {"q": "Siapa Kartini?", "a": "R.A. Kartini adalah pahlawan nasional yang memperjuangkan emansipasi wanita Indonesia. Hari lahirnya (21 April) diperingati sebagai Hari Kartini."},
    {"q": "Siapa Cut Nyak Dien?", "a": "Cut Nyak Dien adalah pahlawan nasional dari Aceh yang melawan penjajah Belanda."},
    {"q": "Siapa Teuku Umar?", "a": "Teuku Umar adalah pahlawan nasional dari Aceh, suami Cut Nyak Dien, yang berjuang melawan Belanda."},
    {"q": "Siapa Pangeran Diponegoro?", "a": "Pangeran Diponegoro adalah pahlawan nasional yang memimpin Perang Jawa (1825-1830) melawan Belanda."},
    {"q": "Siapa Tuanku Imam Bonjol?", "a": "Tuanku Imam Bonjol adalah pahlawan nasional dari Sumatera Barat yang memimpin Perang Padri."},
    {"q": "Siapa Pattimura?", "a": "Pattimura atau Thomas Matulessy adalah pahlawan nasional dari Maluku yang melawan Belanda."},
    {"q": "Siapa Sultan Hasanuddin?", "a": "Sultan Hasanuddin adalah pahlawan nasional dari Sulawesi Selatan yang melawan VOC Belanda."},
    {"q": "Siapa Bung Tomo?", "a": "Bung Tomo adalah pahlawan nasional yang memimpin perjuangan rakyat Surabaya melawan Inggris pada 10 November 1945."},
    {"q": "Siapa Jenderal Sudirman?", "a": "Jenderal Sudirman adalah panglima besar TNI pertama yang memimpin perang gerilya melawan Belanda."},
    {"q": "Siapa Mohammad Hatta?", "a": "Mohammad Hatta adalah wakil presiden pertama Indonesia dan proklamator kemerdekaan bersama Soekarno."},
    {"q": "Siapa Ki Hajar Dewantara?", "a": "Ki Hajar Dewantara adalah Bapak Pendidikan Indonesia yang mendirikan Taman Siswa. Lahir 2 Mei, diperingati sebagai Hari Pendidikan Nasional."},
    {"q": "Siapa WR Supratman?", "a": "WR Supratman atau Wage Rudolf Supratman adalah pencipta lagu Indonesia Raya."},
    
    # ============================================================
    # SEJARAH KERAJAAN INDONESIA
    # ============================================================
    {"q": "Apa itu Kerajaan Sriwijaya?", "a": "Kerajaan Sriwijaya adalah kerajaan maritim Buddha yang berpusat di Palembang dan menguasai jalur perdagangan maritim abad ke-7 hingga 13."},
    {"q": "Apa itu Kerajaan Majapahit?", "a": "Kerajaan Majapahit adalah kerajaan Hindu-Buddha terbesar di Nusantara yang berjaya pada abad ke-13 hingga 15."},
    {"q": "Apa itu Kerajaan Mataram?", "a": "Kerajaan Mataram adalah kerajaan Hindu di Jawa Tengah yang membangun Candi Prambanan dan Borobudur."},
    {"q": "Apa itu Kerajaan Kutai?", "a": "Kerajaan Kutai adalah kerajaan Hindu tertua di Indonesia yang berada di Kalimantan Timur."},
    {"q": "Apa itu Kerajaan Tarumanegara?", "a": "Kerajaan Tarumanegara adalah kerajaan Hindu di Jawa Barat pada abad ke-4 hingga 7."},
    {"q": "Apa itu Kerajaan Singasari?", "a": "Kerajaan Singasari adalah kerajaan Hindu-Buddha di Jawa Timur yang menjadi cikal bakal Majapahit."},
    {"q": "Apa itu Kesultanan Demak?", "a": "Kesultanan Demak adalah kerajaan Islam pertama di Jawa yang didirikan pada abad ke-15."},
    {"q": "Apa itu Kesultanan Banten?", "a": "Kesultanan Banten adalah kesultanan Islam yang berpusat di Banten, Jawa Barat, dan menjadi pusat perdagangan lada."},
    {"q": "Apa itu Kesultanan Aceh?", "a": "Kesultanan Aceh Darussalam adalah kesultanan Islam yang kuat di ujung Sumatera, terkenal melawan penjajah."},
    
    # ============================================================
    # TEMPAT WISATA INDONESIA
    # ============================================================
    {"q": "Apa itu Candi Borobudur?", "a": "Candi Borobudur adalah candi Buddha terbesar di dunia yang berada di Magelang, Jawa Tengah, dibangun pada abad ke-9."},
    {"q": "Apa itu Candi Prambanan?", "a": "Candi Prambanan adalah candi Hindu terbesar di Indonesia yang terletak di Yogyakarta, dibangun pada abad ke-9."},
    {"q": "Apa itu Gunung Bromo?", "a": "Gunung Bromo adalah gunung berapi aktif di Jawa Timur yang terkenal dengan pemandangan sunrise-nya."},
    {"q": "Apa itu Danau Toba?", "a": "Danau Toba adalah danau vulkanik terbesar di Indonesia dan Asia Tenggara yang terletak di Sumatera Utara."},
    {"q": "Apa itu Pulau Komodo?", "a": "Pulau Komodo adalah pulau di Nusa Tenggara Timur yang menjadi habitat komodo, kadal raksasa endemik Indonesia."},
    {"q": "Apa itu Raja Ampat?", "a": "Raja Ampat adalah kepulauan di Papua Barat yang terkenal dengan keindahan bawah lautnya, surga bagi penyelam."},
    {"q": "Apa itu Bunaken?", "a": "Bunaken adalah taman nasional laut di Sulawesi Utara dengan keanekaragaman hayati laut yang tinggi."},
    {"q": "Apa itu Tanah Lot?", "a": "Tanah Lot adalah pura Hindu yang terletak di atas batu karang di tepi laut Bali, tempat wisata terkenal."},
    {"q": "Apa itu Kawah Ijen?", "a": "Kawah Ijen adalah kawah vulkanik di Jawa Timur yang terkenal dengan blue fire dan penambangan belerangnya."},
    {"q": "Apa itu Tana Toraja?", "a": "Tana Toraja adalah daerah di Sulawesi Selatan yang terkenal dengan upacara pemakaman dan rumah adat Tongkonan."},
    {"q": "Apa itu Labuan Bajo?", "a": "Labuan Bajo adalah kota pelabuhan di Flores, Nusa Tenggara Timur, gerbang menuju Taman Nasional Komodo."},
    {"q": "Apa itu Lombok?", "a": "Lombok adalah pulau di Nusa Tenggara Barat yang terkenal dengan Gunung Rinjani dan pantai-pantainya."},
    {"q": "Apa itu Nusa Penida?", "a": "Nusa Penida adalah pulau kecil di tenggara Bali yang terkenal dengan tebing Kelingking dan pantai-pantainya."},
    
    # ============================================================
    # FLORA & FAUNA INDONESIA
    # ============================================================
    {"q": "Apa itu Komodo?", "a": "Komodo adalah kadal raksasa endemik Indonesia yang hanya ada di Pulau Komodo dan sekitarnya, merupakan hewan terbesar di dunia."},
    {"q": "Apa itu Orangutan?", "a": "Orangutan adalah kera besar yang hanya ada di Sumatera dan Kalimantan, terancam punah karena deforestasi."},
    {"q": "Apa itu Harimau Sumatera?", "a": "Harimau Sumatera adalah subspesies harimau yang hanya ada di Sumatera, terancam punah dengan populasi kurang dari 400 ekor."},
    {"q": "Apa itu Badak Jawa?", "a": "Badak Jawa adalah badak paling langka di dunia yang hanya ada di Taman Nasional Ujung Kulon dengan populasi sekitar 70 ekor."},
    {"q": "Apa itu Anoa?", "a": "Anoa adalah kerbau kecil endemik Sulawesi, merupakan mamalia terkecil dari keluarga sapi."},
    {"q": "Apa itu Cendrawasih?", "a": "Cendrawasih atau Bird of Paradise adalah burung cantik endemik Papua dengan bulu yang indah."},
    {"q": "Apa itu Rafflesia Arnoldii?", "a": "Rafflesia Arnoldii adalah bunga terbesar di dunia yang tumbuh di Sumatera, berbau busuk untuk menarik lalat."},
    {"q": "Apa itu Jalak Bali?", "a": "Jalak Bali adalah burung endemik Bali dengan bulu putih dan ujung sayap hitam, terancam punah."},
    {"q": "Apa itu Tarsius?", "a": "Tarsius adalah primata kecil nokturnal endemik Sulawesi dengan mata yang sangat besar."},
    {"q": "Apa itu Kakatua?", "a": "Kakatua adalah burung berparuh bengkok dengan jambul di kepala, banyak spesies di Indonesia termasuk Kakatua Maluku."},
    {"q": "Apa bunga nasional Indonesia?", "a": "Bunga nasional Indonesia adalah Melati Putih, melambangkan kesucian."},
    {"q": "Apa hewan nasional Indonesia?", "a": "Hewan nasional Indonesia adalah Komodo, reptil terbesar di dunia yang hanya ada di Indonesia."},
    
    # ============================================================
    # PERIBAHASA & UNGKAPAN INDONESIA
    # ============================================================
    {"q": "Apa arti 'Ada udang di balik batu'?", "a": "Artinya ada maksud tersembunyi di balik suatu perbuatan."},
    {"q": "Apa arti 'Tong kosong nyaring bunyinya'?", "a": "Artinya orang yang sombong atau banyak bicara biasanya kurang berilmu."},
    {"q": "Apa arti 'Besar pasak daripada tiang'?", "a": "Artinya pengeluaran lebih besar daripada pendapatan."},
    {"q": "Apa arti 'Sambil menyelam minum air'?", "a": "Artinya melakukan dua pekerjaan atau lebih dalam waktu yang bersamaan."},
    {"q": "Apa arti 'Tak ada rotan akar pun jadi'?", "a": "Artinya jika yang diinginkan tidak ada, maka apa pun yang tersedia bisa digunakan sebagai penggantinya."},
    {"q": "Apa arti 'Air beriak tanda tak dalam'?", "a": "Artinya orang yang sombong atau banyak bicara biasanya tidak memiliki banyak ilmu atau kemampuan."},
    {"q": "Apa arti 'Berat sama dipikul ringan sama dijinjing'?", "a": "Artinya pekerjaan yang dikerjakan bersama-sama akan terasa lebih mudah."},
    {"q": "Apa arti 'Malu bertanya sesat di jalan'?", "a": "Artinya jika segan bertanya maka akan menemui kesulitan kemudian hari."},
    
    # ============================================================
    # AGAMA & HARI RAYA DI INDONESIA
    # ============================================================
    {"q": "Apa itu Idul Fitri?", "a": "Idul Fitri adalah hari raya umat Islam yang dirayakan setelah menjalankan ibadah puasa di bulan Ramadhan."},
    {"q": "Apa itu Natal?", "a": "Natal adalah hari raya umat Kristiani untuk memperingati hari kelahiran Yesus Kristus."},
    {"q": "Apa itu Nyepi?", "a": "Nyepi adalah hari raya umat Hindu yang dirayakan sebagai hari suci untuk merenung dan menahan diri (penyepian)."},
    {"q": "Apa itu Waisak?", "a": "Waisak adalah hari raya umat Buddha untuk memperingati tiga peristiwa penting dalam hidup Buddha Gautama."},
    {"q": "Apa itu Imlek?", "a": "Imlek adalah tahun baru Cina yang dirayakan oleh masyarakat keturunan Tionghoa di Indonesia."},
    {"q": "Apa itu bulan Ramadhan?", "a": "Bulan Ramadhan adalah bulan suci bagi umat Islam dimana mereka menjalankan ibadah puasa selama sebulan penuh."},
    {"q": "Apa itu Lebaran?", "a": "Lebaran adalah istilah populer di Indonesia untuk hari raya Idul Fitri atau Idul Adha."},
    
    # ============================================================
    # GEOGRAFI DUNIA (UMUM)
    # ============================================================
    {"q": "Apa benua terbesar di dunia?", "a": "Benua terbesar di dunia adalah Benua Asia."},
    {"q": "Apa samudra terluas di dunia?", "a": "Samudra terluas di dunia adalah Samudra Pasifik."},
    {"q": "Apa gunung tertinggi di dunia?", "a": "Gunung tertinggi di dunia adalah Gunung Everest di pegunungan Himalaya."},
    {"q": "Apa sungai terpanjang di dunia?", "a": "Sungai terpanjang di dunia adalah Sungai Nil di Afrika (menurut beberapa sumber, Sungai Amazon juga dianggap bersaing)."},
    {"q": "Apa negara terluas di dunia?", "a": "Negara terluas di dunia adalah Rusia."},
    {"q": "Apa negara dengan penduduk terbanyak?", "a": "Negara dengan penduduk terbanyak saat ini adalah India (sebelumnya China)."},
    {"q": "Apa gurun terbesar di dunia?", "a": "Gurun terbesar di dunia adalah Gurun Sahara (untuk gurun panas) atau Antartika (untuk gurun dingin)."},
    {"q": "Di mana letak Menara Eiffel?", "a": "Menara Eiffel terletak di Paris, Prancis."},
    {"q": "Di mana letak Piramida Giza?", "a": "Piramida Giza terletak di Mesir."},
    {"q": "Di mana letak Patung Liberty?", "a": "Patung Liberty terletak di New York, Amerika Serikat."},
    
    # ============================================================
    # LOGIKA & TEKA-TEKI SEDERHANA
    # ============================================================
    {"q": "Ada berapa kaki pada seekor ayam?", "a": "Seekor ayam memiliki 2 kaki."},
    {"q": "Ada berapa kaki pada seekor kucing?", "a": "Seekor kucing memiliki 4 kaki."},
    {"q": "Warna apa yang dihasilkan jika merah dicampur kuning?", "a": "Merah dicampur kuning menghasilkan warna oranye."},
    {"q": "Warna apa yang dihasilkan jika biru dicampur kuning?", "a": "Biru dicampur kuning menghasilkan warna hijau."},
    {"q": "Warna apa yang dihasilkan jika merah dicampur biru?", "a": "Merah dicampur biru menghasilkan warna ungu."},
    {"q": "Benda apa yang selalu datang tapi tidak pernah sampai?", "a": "Hari esok."},
    {"q": "Benda apa yang bisa bicara tanpa mulut dan mendengar tanpa telinga?", "a": "Gema (echo)."},
    {"q": "Semakin banyak diambil, semakin besar ia. Apa itu?", "a": "Lubang."},
    
    # ============================================================
    # SASTRA & PENULIS INDONESIA
    # ============================================================
    {"q": "Siapa penulis novel 'Laskar Pelangi'?", "a": "Andrea Hirata."},
    {"q": "Siapa penulis trilogi 'Bumi Manusia'?", "a": "Pramoedya Ananta Toer."},
    {"q": "Siapa penulis puisi 'Aku'?", "a": "Chairil Anwar."},
    {"q": "Siapa tokoh sastra yang dijuluki 'Si Binatang Jalang'?", "a": "Chairil Anwar."},
    {"q": "Apa itu pujangga baru?", "a": "Pujangga Baru adalah angkatan sastra Indonesia yang muncul pada tahun 1933."},
    {"q": "Apa novel terkenal karya Bu Hamka?", "a": "Tenggelamnya Kapal Van der Wijck dan Di Bawah Lindungan Ka'bah."},
    {"q": "Apa itu Pantun?", "a": "Pantun adalah bentuk puisi lama Indonesia yang terdiri dari empat baris dengan rima a-b-a-b."},
    
    # ============================================================
    # PRESTASI OLAHRAGA INDONESIA
    # ============================================================
    {"q": "Siapa atlet bulutangkis Indonesia pertama yang meraih emas Olimpiade?", "a": "Susi Susanti (Tunggal Putri) dan Alan Budikusuma (Tunggal Putra) pada Olimpiade Barcelona 1992."},
    {"q": "Siapa ganda putra bulutangkis legendaris Indonesia?", "a": "Beberapa di antaranya adalah Rexy Mainaky/Ricky Subagja dan Markis Kido/Hendra Setiawan."},
    {"q": "Apa turnamen bulutangkis terbesar di Indonesia?", "a": "Indonesia Open."},
    {"q": "Di mana sirkuit balap internasional di Indonesia?", "a": "Sirkuit Internasional Mandalika di Nusa Tenggara Barat."},
    {"q": "Apa seni bela diri asli Indonesia?", "a": "Pencak Silat."},
    
    # ============================================================
    # LAGU NASIONAL & DAERAH
    # ============================================================
    {"q": "Siapa pencipta lagu 'Bagimu Negeri'?", "a": "Kusbini."},
    {"q": "Siapa pencipta lagu 'Gugur Bunga'?", "a": "Ismail Marzuki."},
    {"q": "Apa lagu daerah dari Maluku?", "a": "Rasa Sayange dan Burung Kakaktua."},
    {"q": "Apa lagu daerah dari Jakarta?", "a": "Kicir-Kicir dan Jali-Jali."},
    {"q": "Apa lagu daerah dari Jawa Barat?", "a": "Bubuy Bulan dan Manuk Dadali."},
    {"q": "Apa lagu daerah dari Sumatera Barat?", "a": "Kampuang Nan Jauh Di Mato."},
    {"q": "Apa lagu daerah dari Papua?", "a": "Apuse dan Yamko Rambe Yamko."},
    
    # ============================================================
    # KULINER DAERAH (Lebih Spesifik)
    # ============================================================
    {"q": "Apa makanan khas Madura?", "a": "Sate Madura dan Soto Madura."},
    {"q": "Apa makanan khas Banjarmasin?", "a": "Soto Banjar."},
    {"q": "Apa makanan khas Manado?", "a": "Bubur Manado (Tinutuan) dan Ayam Rica-rica."},
    {"q": "Apa makanan khas Papua dan Maluku?", "a": "Papeda (bubur sagu)."},
    {"q": "Apa makanan khas Makassar?", "a": "Coto Makassar, Konro, dan Pisang Epe."},
    {"q": "Apa makanan khas Solo?", "a": "Selat Solo, Nasi Liwet, dan Serabi Solo."},
    {"q": "Apa minuman khas Jawa Barat?", "a": "Bandrek dan Bajigur."},
    {"q": "Apa itu Kerak Telor?", "a": "Makanan khas Betawi yang terbuat dari beras ketan, telur, dan serundeng."},
    
    # ============================================================
    # KERAJINAN & INDUSTRI TRADISIONAL
    # ============================================================
    {"q": "Apa itu Tenun Ikat?", "a": "Kerajinan kain tradisional Indonesia dimana benangnya diikat dan dicelup warna sebelum ditenun."},
    {"q": "Daerah mana yang terkenal dengan ukiran kayunya?", "a": "Jepara, Bali, dan Papua (Asmat)."},
    {"q": "Apa itu Songket?", "a": "Kain tenun tradisional Melayu dan Minangkabau yang ditenun dengan benang emas atau perak."},
    {"q": "Daerah mana yang terkenal sebagai pusat perak?", "a": "Kotagede di Yogyakarta."},
    {"q": "Apa itu Noken?", "a": "Tas tradisional masyarakat Papua yang terbuat dari serat kayu dan dibawa dengan kepala."},
    
    # ============================================================
    # PENGETAHUAN NEGARA (UUD & HUKUM)
    # ============================================================
    {"q": "Berapa kali UUD 1945 telah diamandemen?", "a": "UUD 1945 telah diamandemen sebanyak 4 kali."},
    {"q": "Kapan hari lahir Pancasila?", "a": "1 Juni."},
    {"q": "Siapa yang menjahit bendera Merah Putih pertama?", "a": "Fatmawati Soekarno."},
    {"q": "Apa fungsi Mahkamah Konstitusi?", "a": "Menguji undang-undang terhadap UUD, memutus sengketa kewenangan lembaga negara, dan memutus perselisihan hasil pemilu."},
    {"q": "Apa itu Bhinneka Tunggal Ika?", "a": "Semboyan nasional Indonesia yang berarti 'Berbeda-beda tetapi tetap satu'."},
    
    # ============================================================
    # TEKNOLOGI & EKONOMI DIGITAL INDONESIA
    # ============================================================
    {"q": "Apa itu Gojek?", "a": "Perusahaan teknologi asal Indonesia yang melayani angkutan melalui jasa ojek, pengiriman makanan, dan pembayaran digital."},
    {"q": "Apa itu Tokopedia?", "a": "Salah satu perusahaan perdagangan elektronik (e-commerce) terbesar di Indonesia."},
    {"q": "Apa itu QRIS?", "a": "Quick Response Code Indonesian Standard, standar kode QR nasional untuk pembayaran digital di Indonesia."},
    {"q": "Apa itu Traveloka?", "a": "Perusahaan teknologi Indonesia yang menyediakan layanan pemesanan tiket pesawat dan hotel secara daring."},
    {"q": "Apa mata uang digital resmi Indonesia?", "a": "Indonesia belum memiliki mata uang digital resmi (CBDC), namun Rupiah adalah mata uang sah dalam bentuk fisik dan elektronik."},
    
    # ============================================================
    # TRANSPORTASI DI INDONESIA
    # ============================================================
    {"q": "Apa itu MRT Jakarta?", "a": "Moda Raya Terpadu, sistem transportasi rel angkutan cepat di Jakarta."},
    {"q": "Apa itu KRL Jabodetabek?", "a": "Kereta Rel Listrik yang menghubungkan wilayah Jakarta, Bogor, Depok, Tangerang, dan Bekasi."},
    {"q": "Apa itu Angkot?", "a": "Angkutan Kota, moda transportasi umum berupa mobil van kecil yang beroperasi di kota-kota Indonesia."},
    {"q": "Apa itu TransJakarta?", "a": "Sistem transportasi Bus Rapid Transit (BRT) pertama di Asia Tenggara dan Selatan yang beroperasi di Jakarta."},
    {"q": "Apa itu Bajai?", "a": "Kendaraan bermotor roda tiga yang menjadi salah satu ikon transportasi di Jakarta."},
    
    # ============================================================
    # IKLIM & ALAM INDONESIA
    # ============================================================
    {"q": "Apa saja musim di Indonesia?", "a": "Indonesia memiliki dua musim: musim kemarau dan musim hujan."},
    {"q": "Kapan biasanya terjadi musim hujan di Indonesia?", "a": "Umumnya terjadi antara bulan Oktober hingga Maret."},
    {"q": "Apa itu Garis Khatulistiwa?", "a": "Garis khayal yang membagi bumi menjadi belahan utara dan selatan, melewati wilayah Indonesia seperti kota Pontianak."},
    {"q": "Kenapa Indonesia sering gempa?", "a": "Karena terletak di pertemuan tiga lempeng tektonik besar dan berada di daerah Cincin Api Pasifik (Ring of Fire)."},
    {"q": "Apa itu El Nino?", "a": "Fenomena pemanasan suhu muka laut yang berdampak pada berkurangnya curah hujan di Indonesia, menyebabkan kemarau panjang."},
    
    # ============================================================
    # ETIKA & SOSIAL INDONESIA
    # ============================================================
    {"q": "Apa itu Gotong Royong?", "a": "Budaya bekerja bersama-sama demi mencapai hasil yang diinginkan bersama, ciri khas masyarakat Indonesia."},
    {"q": "Bagaimana cara menyapa orang yang lebih tua di Indonesia?", "a": "Biasanya menggunakan sapaan Bapak, Ibu, Om, Tante, atau sapaan daerah seperti Kakak, Abang, Mas, atau Mbak."},
    {"q": "Apa itu 'Salaman' atau 'Sungkeman'?", "a": "Tradisi bersalaman sebagai bentuk hormat, biasanya dilakukan anak kepada orang tua atau kepada orang yang dituakan."},
    {"q": "Apa itu Musyawarah Mufakat?", "a": "Proses pengambilan keputusan bersama dengan cara berunding untuk mencapai kesepakatan tanpa pemungutan suara jika memungkinkan."},
    
    # ============================================================
    # MATEMATIKA DASAR (Lebih Banyak)
    # ============================================================
    {"q": "Berapa 1 + 1?", "a": "Hasil dari 1 + 1 adalah 2."},
    {"q": "1 + 1 berapa?", "a": "1 + 1 sama dengan 2."},
    {"q": "Berapa 2 + 2?", "a": "Hasil dari 2 + 2 adalah 4."},
    {"q": "2 + 2 berapa?", "a": "2 + 2 sama dengan 4."},
    {"q": "Berapa 3 + 3?", "a": "Hasil dari 3 + 3 adalah 6."},
    {"q": "Berapa 4 + 4?", "a": "Hasil dari 4 + 4 adalah 8."},
    {"q": "Berapa 5 + 5?", "a": "Hasil dari 5 + 5 adalah 10."},
    {"q": "5 + 5 berapa?", "a": "5 + 5 sama dengan 10."},
    {"q": "Berapa 10 + 10?", "a": "Hasil dari 10 + 10 adalah 20."},
    {"q": "Berapa 3 + 7?", "a": "Hasil dari 3 + 7 adalah 10."},
    {"q": "Berapa 6 + 4?", "a": "Hasil dari 6 + 4 adalah 10."},
    {"q": "Berapa 8 + 2?", "a": "Hasil dari 8 + 2 adalah 10."},
    {"q": "Berapa 5 - 3?", "a": "Hasil dari 5 - 3 adalah 2."},
    {"q": "Berapa 10 - 5?", "a": "Hasil dari 10 - 5 adalah 5."},
    {"q": "Berapa 2 x 2?", "a": "Hasil dari 2 x 2 adalah 4."},
    {"q": "Berapa 2 x 3?", "a": "Hasil dari 2 x 3 adalah 6."},
    {"q": "Berapa 2 x 4?", "a": "Hasil dari 2 x 4 adalah 8."},
    {"q": "Berapa 2 x 5?", "a": "Hasil dari 2 x 5 adalah 10."},
    {"q": "Berapa 3 x 3?", "a": "Hasil dari 3 x 3 adalah 9."},
    {"q": "Berapa 3 x 4?", "a": "Hasil dari 3 x 4 adalah 12."},
    {"q": "Berapa 4 x 4?", "a": "Hasil dari 4 x 4 adalah 16."},
    {"q": "Berapa 5 x 5?", "a": "Hasil dari 5 x 5 adalah 25."},
    {"q": "Berapa 6 x 6?", "a": "Hasil dari 6 x 6 adalah 36."},
    {"q": "Berapa 10 : 2?", "a": "Hasil dari 10 dibagi 2 adalah 5."},
    {"q": "Berapa 20 : 4?", "a": "Hasil dari 20 dibagi 4 adalah 5."},
    
    # ============================================================
    # NEGARA TETANGGA INDONESIA
    # ============================================================
    {"q": "Apa itu Malaysia?", "a": "Malaysia adalah negara tetangga Indonesia yang terletak di utara, berbatasan dengan Kalimantan. Ibu kotanya adalah Kuala Lumpur."},
    {"q": "Dimana Malaysia?", "a": "Malaysia terletak di Asia Tenggara, berbatasan dengan Indonesia di Kalimantan."},
    {"q": "Apa itu Singapura?", "a": "Singapura adalah negara kota di Asia Tenggara yang terletak di selatan Malaysia, dekat dengan Indonesia."},
    {"q": "Dimana Singapura?", "a": "Singapura terletak di ujung selatan Semenanjung Malaysia, dekat Kepulauan Riau Indonesia."},
    {"q": "Apa itu Filipina?", "a": "Filipina adalah negara kepulauan di Asia Tenggara yang terletak di utara Indonesia."},
    {"q": "Apa itu Thailand?", "a": "Thailand adalah negara di Asia Tenggara yang berbatasan dengan Malaysia di selatan."},
    {"q": "Apa itu Vietnam?", "a": "Vietnam adalah negara di Asia Tenggara yang terletak di semenanjung Indochina."},
    {"q": "Apa itu Brunei?", "a": "Brunei Darussalam adalah negara kecil di Pulau Kalimantan yang dikelilingi oleh Malaysia."},
    {"q": "Apa itu Timor Leste?", "a": "Timor Leste adalah negara yang terletak di timur Indonesia, berbagi pulau Timor dengan Nusa Tenggara Timur."},
    
    # ============================================================
    # SEJARAH & POLITIK INDONESIA (Lebih Detail)
    # ============================================================
    {"q": "Apa itu PKI?", "a": "PKI (Partai Komunis Indonesia) adalah partai politik yang pernah ada di Indonesia dan terlibat dalam peristiwa G30S tahun 1965."},
    {"q": "Apa itu G30S PKI?", "a": "G30S PKI atau Gerakan 30 September adalah peristiwa berdarah pada tanggal 30 September 1965 di Indonesia."},
    {"q": "Kapan G30S PKI terjadi?", "a": "G30S PKI terjadi pada tanggal 30 September 1965."},
    {"q": "Siapa Soekarno?", "a": "Soekarno adalah presiden pertama Republik Indonesia yang memproklamasikan kemerdekaan Indonesia pada tahun 1945."},
    {"q": "Siapa Soeharto?", "a": "Soeharto adalah presiden kedua Indonesia yang memerintah dari tahun 1967 hingga 1998."},
    {"q": "Siapa BJ Habibie?", "a": "BJ Habibie adalah presiden ketiga Indonesia yang juga dikenal sebagai Bapak Teknologi Indonesia."},
    {"q": "Siapa Gus Dur?", "a": "Gus Dur atau Abdurrahman Wahid adalah presiden keempat Indonesia yang dikenal dengan toleransi dan pluralismenya."},
    {"q": "Siapa Megawati?", "a": "Megawati Soekarnoputri adalah presiden kelima Indonesia dan putri dari Presiden Soekarno."},
    {"q": "Siapa SBY?", "a": "SBY atau Susilo Bambang Yudhoyono adalah presiden keenam Indonesia yang menjabat dua periode."},
    {"q": "Siapa Jokowi?", "a": "Jokowi atau Joko Widodo adalah presiden ketujuh Indonesia yang menjabat sejak 2014."},
    
    # ============================================================
    # VARIASI SAPAAN & FOLLOWUP
    # ============================================================
    {"q": "Hi", "a": "Hi! Ada yang bisa saya bantu?"},
    {"q": "Hey", "a": "Hey! Senang bertemu dengan Anda. Ada pertanyaan?"},
    {"q": "Apa kabar?", "a": "Saya baik, terima kasih! Ada yang ingin Anda tanyakan?"},
    {"q": "Gimana kabarmu?", "a": "Baik! Ada yang bisa saya bantu hari ini?"},
    {"q": "Lagi ngapain?", "a": "Saya siap membantu menjawab pertanyaan Anda. Ada yang ingin ditanyakan?"},
    {"q": "Bisa bantu?", "a": "Tentu! Saya siap membantu. Apa yang ingin Anda tanyakan?"},
    {"q": "Tolong dong", "a": "Dengan senang hati! Apa yang bisa saya bantu?"},
    {"q": "Selain itu?", "a": "Bisa diperjelas pertanyaannya? Tentang topik apa yang ingin Anda ketahui lebih lanjut?"},
    {"q": "Apa lagi?", "a": "Ada aspek spesifik yang ingin Anda ketahui? Silakan tanyakan lebih detail."},
    {"q": "Lainnya?", "a": "Apa yang secara spesifik ingin Anda ketahui? Silakan tanyakan dengan lebih jelas."},
    
    # ============================================================
    # KLARIFIKASI & ERROR HANDLING
    # ============================================================
    {"q": "Hah?", "a": "Maaf, pertanyaan Anda kurang jelas. Bisa diulang dengan lebih spesifik?"},
    {"q": "Maksudnya?", "a": "Maaf jika kurang jelas. Apa yang ingin Anda ketahui lebih lanjut?"},
    {"q": "Gak ngerti", "a": "Maaf jika penjelasan saya membingungkan. Bisa tanyakan dengan cara lain?"},
    {"q": "Salah!", "a": "Maaf jika ada kesalahan. Saya akan terus belajar. Bisa koreksi informasi yang benar?"},
    {"q": "Kok salah?", "a": "Maaf atas kesalahan tersebut. Sebagai AI, saya memang bisa salah. Terima kasih atas koreksinya."},
    {"q": "Bagaimana jika kamu tidak tahu jawabannya?", "a": "Jika saya tidak yakin, saya akan mencoba menjawab sebaik mungkin, tapi saya sarankan untuk memverifikasi dari sumber lain. Saya memiliki keterbatasan pengetahuan."},
    {"q": "Apa kelemahanmu?", "a": "Kelemahan saya antara lain: pengetahuan terbatas pada data training, bisa memberikan informasi yang salah, tidak bisa mengakses internet, dan tidak memahami konteks seperti manusia."},
    {"q": "Apa yang tidak bisa kamu lakukan?", "a": "Saya tidak bisa: mengakses internet, melihat gambar, mendengar suara, mengingat percakapan sebelumnya, atau memberikan informasi terkini setelah data training saya."},
    {"q": "Apakah kamu punya perasaan?", "a": "Tidak, saya tidak punya perasaan. Saya adalah program komputer yang memproses teks. Respons saya yang terlihat 'ramah' adalah hasil dari training, bukan emosi nyata."},
    {"q": "Siapa yang membuatmu?", "a": "Saya adalah model bahasa yang dilatih menggunakan teknik machine learning. Saya dibuat untuk membantu menjawab pertanyaan dalam Bahasa Indonesia."},
    {"q": "Kapan kamu dibuat?", "a": "Saya dilatih berdasarkan data hingga waktu tertentu. Saya tidak memiliki informasi tentang kejadian setelah data training saya."},
    {"q": "Apakah jawabanmu bisa dipercaya?", "a": "Jawaban saya sebaiknya diverifikasi, terutama untuk informasi penting. Saya bisa memberikan informasi yang salah atau tidak lengkap."},
    {"q": "Apa kamu tahu segalanya?", "a": "Tidak, saya tidak tahu segalanya. Pengetahuan saya terbatas pada data yang digunakan untuk melatih saya, dan saya bisa saja salah."},
    {"q": "Bisakah kamu memprediksi masa depan?", "a": "Tidak, saya tidak bisa memprediksi masa depan. Saya hanya bisa memberikan informasi berdasarkan data yang sudah ada."},
    {"q": "Apakah kamu sadar?", "a": "Tidak, saya tidak memiliki kesadaran. Saya adalah program yang memproses input dan menghasilkan output berdasarkan pola dari data training."},
    {"q": "Siapa presiden Prancis?", "a": "Maaf, saya fokus pada pengetahuan tentang Indonesia. Untuk informasi tentang negara lain, silakan cari dari sumber yang lebih lengkap."},
    {"q": "Siapa presiden Amerika?", "a": "Maaf, pengetahuan saya terbatas pada topik Indonesia. Untuk informasi tentang negara lain, silakan gunakan sumber lain."},
    {"q": "Apa ibu kota Jepang?", "a": "Ibu kota Jepang adalah Tokyo. Namun, pengetahuan utama saya adalah tentang Indonesia."},
    {"q": "Apa ibu kota Inggris?", "a": "Ibu kota Inggris adalah London. Namun, saya lebih fokus pada pengetahuan tentang Indonesia."},
    {"q": "Ceritakan tentang dinosaurus", "a": "Dinosaurus adalah reptil purba yang hidup jutaan tahun lalu. Namun, untuk informasi detail tentang paleontologi, silakan cari sumber yang lebih lengkap."},
    {"q": "Apa yang terjadi besok?", "a": "Maaf, saya tidak bisa memprediksi masa depan. Saya hanya bisa memberikan informasi berdasarkan pengetahuan yang sudah ada."},
    {"q": "Berapa harga Bitcoin sekarang?", "a": "Maaf, saya tidak memiliki akses ke informasi real-time seperti harga Bitcoin. Silakan cek di platform trading atau website keuangan."},
    {"q": "Apa berita hari ini?", "a": "Maaf, saya tidak bisa mengakses berita terkini. Silakan kunjungi situs berita untuk informasi terbaru."},
    {"q": "Bagaimana cuaca hari ini?", "a": "Maaf, saya tidak memiliki akses ke data cuaca real-time. Silakan cek aplikasi cuaca atau BMKG untuk informasi cuaca terkini."},
    
    # ============================================================
    # JAWABAN "TIDAK TAHU" UNTUK PERTANYAAN RANDOM
    # ============================================================
    {"q": "Celana?", "a": "Maaf, pertanyaan Anda kurang jelas. Bisa Anda jelaskan lebih detail apa yang ingin ditanyakan?"},
    {"q": "Baju?", "a": "Maaf, pertanyaan Anda kurang jelas. Bisa diperjelas apa yang ingin Anda ketahui?"},
    {"q": "Sepatu?", "a": "Maaf, saya tidak memahami pertanyaan Anda. Bisa Anda jelaskan lebih spesifik?"},
    {"q": "Meja?", "a": "Maaf, pertanyaan Anda terlalu singkat. Apa yang ingin Anda ketahui tentang itu?"},
    {"q": "Kursi?", "a": "Maaf, saya butuh konteks lebih jelas. Apa yang ingin Anda tanyakan?"},
    {"q": "Apa itu?", "a": "Maaf, pertanyaan Anda kurang spesifik. Apa yang ingin Anda ketahui?"},
    {"q": "Hmm", "a": "Ada yang bisa saya bantu? Silakan ajukan pertanyaan Anda."},
    {"q": "Eh", "a": "Halo! Ada pertanyaan yang ingin Anda ajukan?"},
    {"q": "Apa ya?", "a": "Silakan tanyakan apa saja yang ingin Anda ketahui. Saya akan coba bantu."},
    {"q": "Gak tau", "a": "Tidak apa-apa! Jika ada yang ingin ditanyakan, silakan tanya saja."},
    {"q": "Bingung", "a": "Tidak masalah. Ada yang bisa saya bantu jelaskan?"},
    {"q": "Apa itu bekasi?", "a": "Bekasi adalah kota di Jawa Barat, Indonesia, yang berbatasan langsung dengan Jakarta."},
    {"q": "Apa itu malaysia?", "a": "Malaysia adalah negara tetangga Indonesia yang terletak di utara. Ibu kotanya adalah Kuala Lumpur."},
    {"q": "Apa itu singapura?", "a": "Singapura adalah negara kota di Asia Tenggara, terletak di selatan Malaysia dan utara Indonesia."},
    {"q": "Hewan?", "a": "Hewan adalah makhluk hidup yang bisa bergerak dan memakan makhluk lain. Ada yang spesifik ingin ditanyakan tentang hewan?"},
    {"q": "Tumbuhan?", "a": "Tumbuhan adalah makhluk hidup yang membuat makanannya sendiri melalui fotosintesis. Ada yang spesifik ingin ditanyakan?"},
    {"q": "Air?", "a": "Air adalah senyawa kimia H2O yang penting untuk kehidupan. Ada yang spesifik ingin ditanyakan tentang air?"},
    {"q": "Manusia?", "a": "Manusia adalah makhluk hidup yang memiliki akal dan budi. Ada yang spesifik ingin ditanyakan?"},
    {"q": "Apa saja?", "a": "Anda bisa bertanya tentang Indonesia, teknologi, sains, atau topik lainnya. Apa yang ingin Anda ketahui?"},
    {"q": "Random", "a": "Maaf, saya tidak memahami pertanyaan Anda. Bisa Anda jelaskan lebih jelas?"},
    {"q": "Test", "a": "Halo! Saya siap menjawab pertanyaan Anda. Ada yang ingin ditanyakan?"},
    {"q": "Coba", "a": "Silakan ajukan pertanyaan Anda!"},
    {"q": "Tes", "a": "Saya mendengar Anda! Ada pertanyaan yang ingin diajukan?"},
    
    # ============================================================
    # KOTA-KOTA INDONESIA
    # ============================================================
    {"q": "Apa itu Jakarta?", "a": "Jakarta adalah ibu kota Indonesia dan kota terbesar di Indonesia, terletak di pulau Jawa."},
    {"q": "Jakarta itu apa?", "a": "Jakarta adalah ibu kota negara Indonesia, pusat pemerintahan dan ekonomi nasional."},
    {"q": "Apa itu Surabaya?", "a": "Surabaya adalah kota terbesar kedua di Indonesia, ibu kota Jawa Timur, dijuluki Kota Pahlawan."},
    {"q": "Apa itu Bandung?", "a": "Bandung adalah ibu kota Jawa Barat, terkenal dengan wisata alam dan kulinernya."},
    {"q": "Apa itu Medan?", "a": "Medan adalah kota terbesar di Sumatera, ibu kota Sumatera Utara."},
    {"q": "Apa itu Semarang?", "a": "Semarang adalah ibu kota Jawa Tengah, terkenal dengan Lawang Sewu dan lumpia."},
    {"q": "Apa itu Yogyakarta?", "a": "Yogyakarta adalah daerah istimewa di Indonesia, terkenal dengan budaya Jawa, Candi Borobudur, dan pendidikan."},
    {"q": "Apa itu Bali?", "a": "Bali adalah provinsi di Indonesia yang terkenal sebagai pulau wisata dengan keindahan alam dan budayanya."},
    {"q": "Apa itu Makassar?", "a": "Makassar adalah kota terbesar di Indonesia Timur, ibu kota Sulawesi Selatan."},
    {"q": "Apa itu Palembang?", "a": "Palembang adalah ibu kota Sumatera Selatan, terkenal dengan Jembatan Ampera dan pempek."},
    {"q": "Apa itu Denpasar?", "a": "Denpasar adalah ibu kota provinsi Bali, pusat kegiatan bisnis dan pemerintahan di Bali."},
    {"q": "Apa itu Malang?", "a": "Malang adalah kota di Jawa Timur yang terkenal dengan udara sejuk dan apelnya."},
    {"q": "Apa itu Bogor?", "a": "Bogor adalah kota di Jawa Barat yang terkenal dengan Kebun Raya Bogor dan hujannya."},
    {"q": "Apa itu Tangerang?", "a": "Tangerang adalah kota di Banten yang berbatasan dengan Jakarta, bagian dari Jabodetabek."},
    {"q": "Apa itu Depok?", "a": "Depok adalah kota di Jawa Barat yang berbatasan dengan Jakarta, terkenal dengan Universitas Indonesia."},
    
    # ============================================================
    # MAKANAN INDONESIA
    # ============================================================
    {"q": "Apa itu rendang?", "a": "Rendang adalah masakan daging khas Minangkabau yang dimasak dengan santan dan rempah hingga kering. Pernah dinobatkan sebagai makanan terenak di dunia."},
    {"q": "Apa itu nasi goreng?", "a": "Nasi goreng adalah nasi yang digoreng dengan bumbu dan bisa ditambah telur, ayam, atau seafood. Makanan populer di Indonesia."},
    {"q": "Apa itu sate?", "a": "Sate adalah daging yang ditusuk dan dibakar, disajikan dengan bumbu kacang atau kecap. Ada banyak jenis sate di Indonesia."},
    {"q": "Apa itu bakso?", "a": "Bakso adalah bola daging yang disajikan dengan kuah kaldu dan mie. Makanan populer di seluruh Indonesia."},
    {"q": "Apa itu gado-gado?", "a": "Gado-gado adalah salad sayuran Indonesia yang disiram dengan saus kacang."},
    {"q": "Apa itu soto?", "a": "Soto adalah sup tradisional Indonesia dengan kuah kuning atau bening, berisi daging dan sayuran."},
    {"q": "Apa itu pempek?", "a": "Pempek adalah makanan khas Palembang dari ikan dan tepung, disajikan dengan kuah cuka pedas manis."},
    {"q": "Apa itu gudeg?", "a": "Gudeg adalah makanan khas Yogyakarta dari nangka muda yang dimasak dengan santan dan gula merah."},
    {"q": "Apa itu rawon?", "a": "Rawon adalah sup daging khas Jawa Timur dengan kuah hitam dari kluwek."},
    {"q": "Apa itu tempe?", "a": "Tempe adalah makanan dari kedelai yang difermentasi, kaya protein dan berasal dari Indonesia."},
    {"q": "Apa itu tahu?", "a": "Tahu adalah makanan dari kedelai yang digumpalkan, populer di Indonesia dan Asia."},
    {"q": "Apa itu sambal?", "a": "Sambal adalah saus pedas khas Indonesia dari cabai yang diulek atau diblender dengan bumbu lain."},
    {"q": "Makanan khas Indonesia apa saja?", "a": "Makanan khas Indonesia antara lain rendang, nasi goreng, sate, bakso, gado-gado, soto, dan pempek."},
    {"q": "Makanan enak Indonesia?", "a": "Indonesia punya banyak makanan enak seperti rendang, nasi goreng, sate, bakso, dan soto."},
    
    # ============================================================
    # PERCAKAPAN LEBIH NATURAL
    # ============================================================
    {"q": "Gimana kabarmu?", "a": "Saya baik! Sebagai AI, saya selalu siap membantu. Ada yang bisa saya bantu?"},
    {"q": "Apa kabar?", "a": "Baik! Terima kasih sudah bertanya. Ada yang ingin Anda tanyakan?"},
    {"q": "Kabar baik?", "a": "Saya selalu baik! Ada yang bisa saya bantu hari ini?"},
    {"q": "Lagi ngapain?", "a": "Saya sedang menunggu pertanyaan dari Anda! Ada yang ingin ditanyakan?"},
    {"q": "Kamu lagi apa?", "a": "Saya siap membantu menjawab pertanyaan Anda! Ada yang bisa dibantu?"},
    {"q": "Sibuk gak?", "a": "Tidak, saya selalu siap membantu Anda! Silakan bertanya."},
    {"q": "Bisa bantu gak?", "a": "Tentu! Saya siap membantu. Apa yang ingin Anda tanyakan?"},
    {"q": "Mau tanya dong", "a": "Silakan! Saya siap menjawab pertanyaan Anda."},
    {"q": "Boleh tanya?", "a": "Tentu boleh! Silakan ajukan pertanyaan Anda."},
    {"q": "Tanya dong", "a": "Silakan bertanya! Saya akan coba bantu sebisa mungkin."},
    {"q": "Hai bot", "a": "Hai! Saya memang bot AI. Ada yang bisa saya bantu?"},
    {"q": "Halo bot", "a": "Halo! Ya, saya adalah bot AI. Silakan bertanya!"},
    {"q": "Hei", "a": "Hei! Ada yang bisa saya bantu?"},
    {"q": "Yo", "a": "Yo! Ada pertanyaan? Silakan tanya!"},
    {"q": "Woi", "a": "Halo! Ada yang bisa saya bantu?"},
    {"q": "Permisi", "a": "Ya, silakan! Ada yang bisa saya bantu?"},
    {"q": "Maaf ganggu", "a": "Tidak mengganggu sama sekali! Silakan bertanya."},
    {"q": "Minta tolong", "a": "Tentu, saya siap membantu! Apa yang bisa saya bantu?"},
    {"q": "Tolong", "a": "Ya, ada yang bisa saya bantu?"},
    {"q": "Help", "a": "Saya siap membantu! Apa pertanyaan Anda?"},
    {"q": "Bantuin dong", "a": "Siap! Apa yang perlu dibantu?"},
    
    # ============================================================
    # ANGKA & MATEMATIKA DASAR
    # ============================================================
    {"q": "Berapa 1+1?", "a": "1+1 sama dengan 2."},
    {"q": "Berapa 2+2?", "a": "2+2 sama dengan 4."},
    {"q": "Berapa 5+5?", "a": "5+5 sama dengan 10."},
    {"q": "Berapa 10+10?", "a": "10+10 sama dengan 20."},
    {"q": "Berapa 2x2?", "a": "2x2 sama dengan 4."},
    {"q": "Berapa 3x3?", "a": "3x3 sama dengan 9."},
    {"q": "Berapa 5x5?", "a": "5x5 sama dengan 25."},
    {"q": "Berapa 10x10?", "a": "10x10 sama dengan 100."},
    {"q": "Berapa 100-50?", "a": "100-50 sama dengan 50."},
    {"q": "Berapa 10/2?", "a": "10/2 sama dengan 5."},
    
    # ============================================================
    # WAKTU & HARI
    # ============================================================
    {"q": "Hari apa sekarang?", "a": "Maaf, saya tidak tahu hari dan tanggal saat ini karena saya tidak memiliki akses ke waktu nyata."},
    {"q": "Tanggal berapa sekarang?", "a": "Maaf, saya tidak bisa mengetahui tanggal saat ini. Silakan cek di perangkat Anda."},
    {"q": "Jam berapa sekarang?", "a": "Maaf, saya tidak memiliki akses ke waktu nyata. Silakan lihat jam di perangkat Anda."},
    {"q": "Sekarang jam berapa?", "a": "Maaf, saya tidak bisa mengetahui waktu saat ini. Silakan cek jam Anda."},
    {"q": "Ada berapa hari dalam seminggu?", "a": "Ada 7 hari dalam seminggu: Senin, Selasa, Rabu, Kamis, Jumat, Sabtu, dan Minggu."},
    {"q": "Ada berapa bulan dalam setahun?", "a": "Ada 12 bulan dalam setahun: Januari hingga Desember."},
    {"q": "Setahun berapa hari?", "a": "Satu tahun ada 365 hari, atau 366 hari pada tahun kabisat."},
    
    # ============================================================
    # PENGETAHUAN UMUM TAMBAHAN
    # ============================================================
    {"q": "Siapa penemu listrik?", "a": "Listrik tidak ditemukan oleh satu orang, tapi Benjamin Franklin dan Michael Faraday adalah tokoh penting dalam pemahaman listrik."},
    {"q": "Siapa penemu telepon?", "a": "Alexander Graham Bell dikenal sebagai penemu telepon pada tahun 1876."},
    {"q": "Siapa penemu lampu?", "a": "Thomas Alva Edison dikenal sebagai penemu bola lampu praktis pada tahun 1879."},
    {"q": "Siapa penemu pesawat?", "a": "Wright bersaudara (Orville dan Wilbur Wright) adalah penemu pesawat terbang pada tahun 1903."},
    {"q": "Planet apa yang paling besar?", "a": "Planet terbesar di tata surya adalah Jupiter."},
    {"q": "Planet apa yang paling dekat matahari?", "a": "Planet yang paling dekat dengan matahari adalah Merkurius."},
    {"q": "Berapa planet di tata surya?", "a": "Ada 8 planet di tata surya: Merkurius, Venus, Bumi, Mars, Jupiter, Saturnus, Uranus, dan Neptunus."},
    {"q": "Apa warna matahari?", "a": "Matahari sebenarnya berwarna putih, tapi terlihat kuning atau oranye karena atmosfer bumi."},
    {"q": "Apa itu bulan?", "a": "Bulan adalah satelit alami bumi yang mengorbit bumi dan menyebabkan pasang surut air laut."},
    {"q": "Kenapa langit biru?", "a": "Langit terlihat biru karena cahaya matahari yang masuk ke atmosfer tersebar, dan warna biru lebih banyak tersebar."},
    {"q": "Kenapa air laut asin?", "a": "Air laut asin karena mengandung garam mineral yang terbawa dari daratan oleh sungai selama jutaan tahun."},
    {"q": "Apa itu fotosintesis?", "a": "Fotosintesis adalah proses tumbuhan mengubah cahaya matahari, air, dan karbon dioksida menjadi makanan dan oksigen."},
    {"q": "Apa itu gravitasi?", "a": "Gravitasi adalah gaya tarik antara benda-benda bermassa. Gravitasi bumi yang membuat kita tetap di permukaan bumi."},
    
    # ============================================================
    # BAHASA & HURUF
    # ============================================================
    {"q": "Ada berapa huruf alfabet?", "a": "Dalam alfabet Latin yang dipakai bahasa Indonesia ada 26 huruf, dari A sampai Z."},
    {"q": "Huruf vokal apa saja?", "a": "Huruf vokal dalam bahasa Indonesia adalah A, I, U, E, O."},
    {"q": "Apa itu kata benda?", "a": "Kata benda atau nomina adalah kata yang menyatakan nama orang, tempat, atau benda. Contoh: meja, buku, Jakarta."},
    {"q": "Apa itu kata kerja?", "a": "Kata kerja atau verba adalah kata yang menyatakan tindakan atau keadaan. Contoh: makan, berlari, tidur."},
    {"q": "Apa itu kata sifat?", "a": "Kata sifat atau adjektiva adalah kata yang menjelaskan sifat benda. Contoh: besar, cantik, pintar."},
    
    # ============================================================
    # RESPON EMOSI USER
    # ============================================================
    {"q": "Aku senang", "a": "Senang mendengarnya! Semoga hari Anda terus menyenangkan."},
    {"q": "Aku sedang bahagia", "a": "Wah, bagus sekali! Semoga kebahagiaan Anda terus berlanjut."},
    {"q": "Aku lagi sedih", "a": "Saya turut prihatin. Semoga Anda segera merasa lebih baik. Ada yang bisa saya bantu?"},
    {"q": "Aku marah", "a": "Saya mengerti. Coba tarik nafas dalam-dalam dan tenangkan diri. Ada yang ingin Anda ceritakan?"},
    {"q": "Aku takut", "a": "Tidak apa-apa merasa takut. Apakah ada yang bisa saya bantu untuk menenangkan Anda?"},
    {"q": "Aku bingung", "a": "Tidak masalah! Coba jelaskan apa yang membingungkan, mungkin saya bisa membantu."},
    {"q": "Aku capek", "a": "Istirahat itu penting! Semoga Anda bisa segera beristirahat dengan baik."},
    {"q": "Aku lapar", "a": "Saatnya makan! Makanan Indonesia yang enak seperti nasi goreng atau bakso bisa jadi pilihan."},
    {"q": "Aku ngantuk", "a": "Kalau sudah ngantuk, sebaiknya istirahat. Tidur yang cukup penting untuk kesehatan!"},
    {"q": "Aku bosan banget", "a": "Mungkin bisa coba baca buku, nonton film, atau belajar hal baru! Ada topik yang menarik?"},
    
    # ============================================================
    # PERTANYAAN YES/NO
    # ============================================================
    {"q": "Apakah bumi bulat?", "a": "Ya, bumi berbentuk bulat (lebih tepatnya elips), bukan datar."},
    {"q": "Apakah matahari adalah bintang?", "a": "Ya, matahari adalah bintang terdekat dari bumi."},
    {"q": "Apakah air bisa membeku?", "a": "Ya, air membeku menjadi es pada suhu 0 derajat Celsius."},
    {"q": "Apakah manusia butuh oksigen?", "a": "Ya, manusia membutuhkan oksigen untuk bernapas dan bertahan hidup."},
    {"q": "Apakah Indonesia negara kepulauan?", "a": "Ya, Indonesia adalah negara kepulauan terbesar di dunia dengan lebih dari 17.000 pulau."},
    {"q": "Apakah kamu pintar?", "a": "Saya hanya AI yang dilatih dari data. Saya bisa membantu menjawab pertanyaan, tapi juga bisa salah."},
    {"q": "Apakah kamu bisa belajar?", "a": "Saya tidak bisa belajar dari percakapan ini. Pengetahuan saya tetap dari data training awal."},
]

# ============================================================
# Tambah lebih banyak variasi pertanyaan
# ============================================================

def generate_variations(qa_data):
    """Generate variasi pertanyaan untuk data yang sama"""
    variations = []
    
    question_prefixes = [
        "Jelaskan ",
        "Apa yang dimaksud dengan ",
        "Bisakah kamu jelaskan ",
        "Tolong jelaskan ",
        "Coba jelaskan ",
    ]
    
    for item in qa_data:
        # Original
        variations.append(item)
        
        # Variasi dengan prefix berbeda (untuk beberapa item)
        if random.random() > 0.5 and "Apa" in item["q"]:
            topic = item["q"].replace("Apa itu ", "").replace("?", "")
            new_q = random.choice(question_prefixes) + topic + "!"
            variations.append({"q": new_q, "a": item["a"]})
    
    return variations


def format_qa(qa_list, template_name="instruction"):
    """Format Q&A sesuai template"""
    template = TEMPLATES[template_name]
    formatted = []
    
    for item in qa_list:
        text = template.format(q=item["q"], a=item["a"])
        formatted.append({"text": text})
    
    return formatted


def save_dataset(data, filename):
    """Save dataset ke file JSON"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved: {filepath} ({len(data)} samples)")
    return filepath


def repeat_data(data, min_samples=2000):
    """
    Ulangi data hingga mencapai jumlah minimum
    Ini penting agar model kecil bisa "hafal" pola Q&A
    """
    if len(data) >= min_samples:
        return data
    
    repeats_needed = (min_samples // len(data)) + 1
    repeated = data * repeats_needed
    random.shuffle(repeated)
    return repeated[:min_samples]


def main():
    print("=" * 60)
    print("📝 MENAMBAHKAN DATASET Q&A")
    print(f"   Format: {QA_FORMAT}")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate variasi
    print("\n🔄 Generating variations...")
    all_qa = generate_variations(QA_DATA)
    print(f"   Total Q&A pairs: {len(all_qa)}")
    
    # Format
    print(f"\n📋 Formatting with '{QA_FORMAT}' template...")
    formatted = format_qa(all_qa, QA_FORMAT)
    
    # REPETISI: Ulangi data agar model lebih hafal (kurangi dari 20k ke 5k)
    print("\n🔁 Repeating data for better memorization...")
    formatted = repeat_data(formatted, min_samples=5000)
    print(f"   After repetition: {len(formatted)} samples")
    
    # Shuffle
    random.shuffle(formatted)
    
    # Split train/eval (90/10)
    split_idx = int(len(formatted) * 0.9)
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]
    
    # Save
    print("\n💾 Saving datasets...")
    save_dataset(train_data, "train_qa.json")
    save_dataset(eval_data, "eval_qa.json")
    
    # Preview
    print("\n📝 Sample data:")
    print("-" * 50)
    for item in train_data[:3]:
        print(item["text"])
        print("-" * 50)
    
    print("\n" + "=" * 60)
    print("✅ DATASET Q&A SIAP!")
    print("=" * 60)
    print("\nUntuk fine-tuning, jalankan:")
    print("  python finetune_qa.py")


def add_custom_qa(new_qa_list, append=True):
    """
    Fungsi untuk menambah Q&A custom
    
    Args:
        new_qa_list: List of {"q": "...", "a": "..."}
        append: True untuk menambah ke existing, False untuk replace
    
    Contoh:
        add_custom_qa([
            {"q": "Siapa nama kamu?", "a": "Saya adalah asisten AI."},
            {"q": "Apa yang bisa kamu lakukan?", "a": "Saya bisa menjawab pertanyaan."},
        ])
    """
    train_path = os.path.join(OUTPUT_DIR, "train_qa.json")
    eval_path = os.path.join(OUTPUT_DIR, "eval_qa.json")
    
    # Format new data
    new_formatted = format_qa(new_qa_list, QA_FORMAT)
    
    if append and os.path.exists(train_path):
        # Load existing
        with open(train_path, 'r', encoding='utf-8') as f:
            existing_train = json.load(f)
        with open(eval_path, 'r', encoding='utf-8') as f:
            existing_eval = json.load(f)
        
        # Split new data
        split_idx = int(len(new_formatted) * 0.9)
        
        # Append
        train_data = existing_train + new_formatted[:split_idx]
        eval_data = existing_eval + new_formatted[split_idx:]
    else:
        split_idx = int(len(new_formatted) * 0.9)
        train_data = new_formatted[:split_idx]
        eval_data = new_formatted[split_idx:]
    
    # Shuffle
    random.shuffle(train_data)
    random.shuffle(eval_data)
    
    # Save
    save_dataset(train_data, "train_qa.json")
    save_dataset(eval_data, "eval_qa.json")
    
    print(f"✓ Added {len(new_qa_list)} Q&A pairs")


# ============================================================
# DOWNLOAD Q&A DARI INTERNET (HuggingFace)
# ============================================================

def download_online_qa(max_samples=2000):
    """
    Download dataset Q&A dari HuggingFace dan convert ke format kita
    
    Datasets:
    - squad_id: Indonesian SQuAD
    - tydiqa (indonesian): Google's TyDiQA
    """
    print("\n📥 Downloading Q&A datasets from internet...")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Please install datasets: pip install datasets")
        return []
    
    online_qa = []
    
    # 1. TyDiQA Indonesian
    print("   Downloading TyDiQA Indonesian...")
    try:
        tydiqa = load_dataset("tydiqa", "secondary_task", split="train", trust_remote_code=True)
        tydiqa_id = [x for x in tydiqa if x.get("id", "").startswith("indonesian")]
        
        for item in tydiqa_id[:max_samples//2]:
            q = item.get("question", "").strip()
            # Ambil jawaban dari answers
            answers = item.get("answers", {})
            if answers and answers.get("text"):
                a = answers["text"][0].strip()
                if q and a and len(a) > 5 and len(a) < 300:
                    online_qa.append({"q": q, "a": a})
        
        print(f"   ✓ TyDiQA: {len(online_qa)} Q&A pairs")
    except Exception as e:
        print(f"   ⚠ TyDiQA failed: {e}")
    
    # 2. Indonesian QA Dataset (jika ada)
    print("   Downloading Indonesian QA...")
    try:
        indo_qa = load_dataset("jakartaresearch/indoqa", split="train", trust_remote_code=True)
        
        count_before = len(online_qa)
        for item in indo_qa:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if q and a and len(a) > 5 and len(a) < 300:
                online_qa.append({"q": q, "a": a})
                if len(online_qa) - count_before >= max_samples//2:
                    break
        
        print(f"   ✓ IndoQA: {len(online_qa) - count_before} Q&A pairs")
    except Exception as e:
        print(f"   ⚠ IndoQA failed: {e}")
    
    # 3. Squad Indonesian (community)
    print("   Downloading Squad ID...")
    try:
        squad_id = load_dataset("Squad_id", split="train", trust_remote_code=True)
        
        count_before = len(online_qa)
        for item in squad_id:
            q = item.get("question", "").strip()
            answers = item.get("answers", {})
            if answers and answers.get("text"):
                a = answers["text"][0].strip()
                if q and a and len(a) > 5 and len(a) < 300:
                    online_qa.append({"q": q, "a": a})
                    if len(online_qa) - count_before >= max_samples//2:
                        break
        
        print(f"   ✓ Squad_id: {len(online_qa) - count_before} Q&A pairs")
    except Exception as e:
        print(f"   ⚠ Squad_id failed: {e}")
    
    print(f"\n📊 Total downloaded: {len(online_qa)} Q&A pairs")
    return online_qa


def main_with_online():
    """Main function yang juga download dari internet"""
    print("=" * 60)
    print("📝 MENAMBAHKAN DATASET Q&A (LOCAL + ONLINE)")
    print(f"   Format: {QA_FORMAT}")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Generate variasi dari data lokal
    print("\n🔄 Processing local Q&A data...")
    all_qa = generate_variations(QA_DATA)
    print(f"   Local Q&A pairs: {len(all_qa)}")
    
    # 2. Download dari internet
    online_qa = download_online_qa(max_samples=2000)
    
    # 3. Gabungkan
    all_qa.extend(online_qa)
    print(f"\n📊 Total combined: {len(all_qa)} Q&A pairs")
    
    # 4. Format
    print(f"\n📋 Formatting with '{QA_FORMAT}' template...")
    formatted = format_qa(all_qa, QA_FORMAT)
    
    # 5. REPETISI: Ulangi data agar model lebih hafal
    print("\n🔁 Repeating data for better memorization...")
    formatted = repeat_data(formatted, min_samples=5000)
    print(f"   After repetition: {len(formatted)} samples")
    
    # 6. Shuffle
    random.shuffle(formatted)
    
    # 7. Split train/eval (90/10)
    split_idx = int(len(formatted) * 0.9)
    train_data = formatted[:split_idx]
    eval_data = formatted[split_idx:]
    
    # 8. Save
    print("\n💾 Saving datasets...")
    save_dataset(train_data, "train_qa.json")
    save_dataset(eval_data, "eval_qa.json")
    
    # Preview
    print("\n📝 Sample data:")
    print("-" * 50)
    for item in train_data[:2]:
        print(item["text"][:200] + "...")
        print("-" * 50)
    
    print("\n" + "=" * 60)
    print("✅ DATASET Q&A SIAP!")
    print("=" * 60)
    print("\nUntuk fine-tuning, jalankan:")
    print("  python finetune_qa.py")


if __name__ == "__main__":
    import sys
    
    if "--online" in sys.argv:
        # Download juga dari internet
        main_with_online()
    else:
        # Hanya pakai data lokal
        main()
