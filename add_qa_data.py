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
    # CONVERSATIONAL / CHATBOT
    # ============================================================
    {"q": "Halo", "a": "Halo! Ada yang bisa saya bantu?"},
    {"q": "Hai", "a": "Hai! Senang bertemu dengan Anda. Ada yang ingin ditanyakan?"},
    {"q": "Halo, apa kabar?", "a": "Halo! Saya baik, terima kasih. Ada yang bisa saya bantu hari ini?"},
    {"q": "Selamat pagi", "a": "Selamat pagi! Semoga hari Anda menyenangkan. Ada yang bisa saya bantu?"},
    {"q": "Selamat siang", "a": "Selamat siang! Ada yang bisa saya bantu?"},
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
    print("\nUntuk training, jalankan:")
    print("  python train_tiny_llm.py")
    print("\nPastikan path dataset di train_tiny_llm.py:")
    print('  train_path = "./dataset/train_qa.json"')
    print('  eval_path = "./dataset/eval_qa.json"')


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


if __name__ == "__main__":
    main()
