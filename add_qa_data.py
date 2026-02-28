"""
Script untuk Menambahkan Dataset Q&A
====================================
Cara mudah menambah data tanya jawab untuk training

Penggunaan:
    python add_qa_data.py
"""import json
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
    # TEKNOLOGI & AI — PENJELASAN MENDALAM
    # ============================================================
    {"q": "Bagaimana cara kerja kecerdasan buatan (AI)?", "a": "Sistem AI bekerja dengan cara mempelajari pola dari data yang sangat besar. Untuk memahami ini, kita perlu membayangkan otak manusia: ketika kita melihat ribuan gambar kucing, otak kita belajar mengenali ciri khas kucing. AI melakukan hal serupa melalui lapisan-lapisan matematis yang disebut neural network. Setiap lapisan mendeteksi fitur yang semakin kompleks—lapisan awal mengenali tepi dan warna, lapisan tengah menggabungkannya menjadi bentuk, dan lapisan akhir menghasilkan keputusan. Keuntungannya adalah AI bisa memproses jutaan data dengan konsisten tanpa lelah. Kerugiannya adalah AI tidak memiliki pemahaman sejati—ia mengenali pola, bukan makna."},
    {"q": "Apa perbedaan antara machine learning dan deep learning?", "a": "Untuk memahami perbedaannya, bayangkan machine learning sebagai teknik mengajarkan komputer menggunakan fitur yang sudah kita tentukan sendiri, misalnya 'jika spam maka mengandung kata tertentu'. Deep learning adalah evolusinya—sistem tidak perlu diberi tahu fitur apa yang harus dicari. Sistem ini menggunakan neural network berlapis (deep neural network) yang secara otomatis menemukan fitur sendiri dari data mentah. Jadi deep learning adalah bagian dari machine learning, tetapi menggunakan pendekatan yang jauh lebih otomatis dan cocok untuk data kompleks seperti gambar, suara, dan teks."},
    {"q": "Mengapa model AI seperti ChatGPT bisa menjawab pertanyaan dengan baik?", "a": "Model seperti ChatGPT dilatih menggunakan teknik yang disebut transformer dengan data teks dalam jumlah sangat besar—miliaran halaman buku, artikel, dan percakapan. Sistem ini bekerja dengan cara memprediksi kata selanjutnya yang paling mungkin muncul berdasarkan konteks kalimat sebelumnya. Setelah pre-training, model disesuaikan lebih lanjut menggunakan teknik RLHF (Reinforcement Learning from Human Feedback), di mana penilai manusia mengevaluasi jawaban sehingga model belajar memberi respons yang berguna dan aman. Hasilnya adalah model yang seolah 'mengerti', padahal sesungguhnya ia sangat mahir dalam pencocokan pola linguistik."},
    {"q": "Apa itu overfitting dalam machine learning dan mengapa itu masalah?", "a": "Overfitting terjadi ketika model belajar terlalu detail dari data latih sehingga ia 'menghafal' data tersebut alih-alih memahami polanya. Analogi yang tepat adalah seorang siswa yang hanya menghafal soal latihan tanpa memahami konsep—ia gagal ketika menghadapi soal baru. Dalam ML, model yang overfit akan mendapat akurasi sangat tinggi pada data latih tetapi buruk pada data baru (data uji). Untuk mengatasinya, kita bisa menggunakan teknik seperti regularisasi, dropout, atau memperbanyak data latih. Keseimbangan antara bias dan variansi adalah kunci—model yang terlalu sederhana underfitting, model yang terlalu kompleks bisa overfitting."},
    {"q": "Bagaimana internet bekerja dari saat kita mengetik URL hingga halaman tampil?", "a": "Proses ini melibatkan beberapa tahap. Pertama, browser menerjemahkan nama domain (misalnya google.com) menjadi alamat IP melalui sistem DNS (Domain Name System)—seperti buku telepon internet. Kedua, browser membuat koneksi ke server menggunakan protokol TCP/IP—data dipecah menjadi paket kecil, dikirim melalui berbagai rute jaringan, lalu disusun ulang di tujuan. Ketiga, browser mengirim permintaan HTTP/HTTPS ke server, server memproses dan mengirim kembali file HTML, CSS, dan JavaScript. Terakhir, browser menguraikan semua file itu dan merender tampilan visual yang kita lihat. Seluruh proses ini biasanya terjadi dalam hitungan milidetik."},
    {"q": "Apa itu enkripsi dan mengapa penting untuk keamanan data?", "a": "Enkripsi adalah proses mengubah data agar tidak bisa dibaca tanpa kunci khusus—seperti mengunci pesan dalam brankas yang hanya bisa dibuka oleh orang yang punya kuncinya. Sistem ini bekerja dengan cara menggunakan algoritma matematika kompleks seperti AES atau RSA untuk mengacak data. Ketika Anda berbelanja online, kartu kredit Anda dienkripsi sebelum dikirim ke server—bahkan jika peretas mencegat data tersebut, mereka hanya mendapat teks acak yang tidak berguna. Tanpa enkripsi, semua komunikasi internet kita—email, pesan, transaksi bank—bisa dibaca oleh siapa saja yang berada di jalur jaringan yang sama."},
    {"q": "Apa itu neural network dan bagaimana cara kerjanya?", "a": "Neural network terinspirasi dari struktur otak manusia. Sistem ini terdiri dari node-node (neuron buatan) yang tersusun dalam lapisan: lapisan input menerima data, lapisan tersembunyi (hidden layers) melakukan transformasi matematis, dan lapisan output menghasilkan prediksi. Setiap koneksi antar neuron memiliki bobot (weight). Proses pembelajaran terjadi melalui backpropagation—ketika prediksi salah, kesalahan dihitung dan dikirim balik ke jaringan untuk menyesuaikan bobot. Proses ini diulang jutaan kali hingga model menghasilkan prediksi yang akurat. Semakin banyak lapisan, semakin 'dalam' jaringan tersebut—itulah asal usul istilah 'deep learning'."},
    {"q": "Mengapa privasi data menjadi isu penting di era digital?", "a": "Di era digital, setiap aktivitas kita meninggalkan jejak data—aplikasi apa yang kita buka, di mana kita berada, apa yang kita beli, bahkan berapa lama kita menatap suatu konten. Keuntungan dari pengumpulan data ini adalah personalisasi layanan yang lebih baik. Namun kerugiannya sangat serius: data bisa dijual ke pihak ketiga tanpa sepengetahuan kita, digunakan untuk manipulasi psikologis melalui iklan bertarget, atau dicuri oleh peretas dan digunakan untuk penipuan identitas. Regulasi seperti GDPR di Eropa hadir karena kesadaran bahwa data pribadi adalah aset bernilai yang harus dilindungi, bukan sekadar byte informasi."},
    {"q": "Apa itu natural language processing (NLP) dan apa aplikasinya?", "a": "Natural Language Processing adalah cabang AI yang memungkinkan komputer memahami, menginterpretasikan, dan menghasilkan bahasa manusia. Untuk memahami tantangannya, bayangkan betapa ambigusnya bahasa kita—'Bank bisa jadi tepi sungai atau lembaga keuangan' dan sarkasme seperti 'Bagus sekali pekerjaanmu!' bisa berarti pujian atau kritikan. NLP bekerja dengan cara mengubah teks menjadi representasi numerik, menganalisis struktur gramatikal, dan memahami konteks semantik. Aplikasinya sangat luas: asisten virtual seperti Siri, terjemahan otomatis, analisis sentimen media sosial, chatbot layanan pelanggan, hingga sistem yang merangkum dokumen panjang secara otomatis."},
    {"q": "Bagaimana cara kerja algoritma rekomendasi di platform seperti YouTube atau Netflix?", "a": "Algoritma rekomendasi bekerja dengan cara menganalisis dua jenis data: perilaku Anda sendiri (video yang ditonton, durasi menonton, video yang di-skip) dan perilaku pengguna lain yang memiliki selera serupa. Teknik utama yang digunakan adalah collaborative filtering—'pengguna yang menyukai video A dan B, juga cenderung menyukai video C'—dikombinasikan dengan content-based filtering yang menganalisis konten video itu sendiri. Model ini terus diperbarui secara real-time. Keuntungannya adalah menemukan konten yang relevan di tengah jutaan pilihan. Namun ada risiko filter bubble—Anda hanya melihat konten yang memperkuat pandangan yang sudah ada, mempersempit perspektif."},
    {"q": "Apa perbedaan antara HTTP dan HTTPS, dan mengapa HTTPS lebih aman?", "a": "HTTP (HyperText Transfer Protocol) adalah protokol komunikasi antara browser dan server, tetapi data dikirim dalam teks biasa yang bisa dibaca oleh siapa pun di jaringan yang sama. HTTPS menambahkan lapisan keamanan dengan enkripsi TLS/SSL—semua data dienkripsi sebelum dikirim. Sistem ini bekerja dengan cara menggunakan kriptografi asimetris: server memiliki kunci publik dan kunci privat. Browser menggunakan kunci publik untuk mengenkripsi data, hanya server yang bisa mendekripsinya menggunakan kunci privat. Analogi sederhananya adalah HTTP seperti kartu pos yang bisa dibaca semua orang, sementara HTTPS seperti surat dalam amplop tersegel. Itulah mengapa URL dengan gembok (🔒) sangat penting terutama saat memasukkan data sensitif."},
    {"q": "Apa itu cloud computing dan apa keuntungannya dibanding server lokal?", "a": "Cloud computing adalah model penyediaan layanan komputasi—server, penyimpanan, database, software—melalui internet (awan) oleh penyedia seperti AWS, Google Cloud, atau Azure. Sistem ini bekerja dengan cara memvirtualisasi sumber daya fisik sehingga bisa dibagi dan diskalakan dinamis. Dibanding server lokal, keuntungannya adalah: skalabilitas instan (bisa menambah kapasitas dalam menit, bukan minggu), model bayar sesuai pemakaian sehingga efisien biaya, tidak perlu tim IT besar untuk pemeliharaan fisik, dan keandalan tinggi dengan redundansi data di banyak lokasi. Kerugiannya adalah ketergantungan pada koneksi internet dan kontrol yang lebih terbatas atas infrastruktur serta potensi risiko keamanan data pihak ketiga."},
    {"q": "Apa itu ransomware dan bagaimana cara melindungi diri darinya?", "a": "Ransomware adalah jenis malware yang mengenkripsi semua file korban dan meminta tebusan (ransom) untuk kunci dekripsinya. Sistemnya bekerja dengan cara masuk melalui email phishing, lampiran berbahaya, atau kerentanan software yang belum diperbarui. Setelah aktif, ia mengenkripsi file secara sistematis dalam hitungan menit. Serangan ransomware WannaCry pada 2017, misalnya, menginfeksi lebih dari 200.000 komputer di 150 negara dan merugikan miliaran dolar. Untuk melindungi diri, strategi terbaik adalah: backup data secara berkala ke lokasi terpisah (offline), selalu perbarui sistem operasi dan software, jangan membuka lampiran email dari pengirim tidak dikenal, dan gunakan antivirus yang terpercaya."},
    {"q": "Bagaimana cara kerja GPS dalam menentukan lokasi kita?", "a": "GPS (Global Positioning System) bekerja dengan cara menggunakan trilaterasi sinyal dari minimal 4 satelit yang mengorbit bumi pada ketinggian sekitar 20.000 km. Setiap satelit terus-menerus memancarkan sinyal waktu yang sangat presisi. Perangkat GPS mengukur berapa lama waktu yang dibutuhkan sinyal tersebut untuk tiba—karena sinyal bergerak dengan kecepatan cahaya (300.000 km/detik), selisih waktu kecil pun mewakili jarak yang signifikan. Dari jarak ke masing-masing satelit, sistem menghitung perpotongan koordinat untuk menentukan posisi tiga dimensi (lintang, bujur, dan ketinggian). Akurasi modern bisa mencapai beberapa sentimeter menggunakan teknik differential GPS."},
    {"q": "Apa itu blockchain dan mengapa teknologi ini dianggap revolusioner?", "a": "Blockchain adalah sistem penyimpanan data yang terdesentralisasi—tidak ada satu entitas pun yang mengontrolnya. Sistem ini bekerja dengan cara menyimpan data dalam 'blok' yang terhubung secara kriptografis membentuk 'rantai'. Setiap blok berisi data transaksi, timestamp, dan hash (sidik jari digital) dari blok sebelumnya. Jika seseorang mencoba mengubah data di satu blok, hash-nya berubah dan semua blok sesudahnya menjadi tidak valid—ini yang membuat blockchain sangat sulit dimanipulasi. Dianggap revolusioner karena menghilangkan kebutuhan perantara terpercaya (seperti bank) untuk memverifikasi transaksi. Namun tantangannya adalah konsumsi energi yang sangat besar dan skalabilitas yang terbatas."},
    {"q": "Apa itu Internet of Things (IoT) dan bagaimana dampaknya pada kehidupan sehari-hari?", "a": "Internet of Things adalah konsep menghubungkan berbagai perangkat fisik—kulkas, lampu, thermostat, mobil, sensor industri—ke internet sehingga bisa berkomunikasi dan dikendalikan dari jarak jauh. Sistem ini bekerja dengan cara menanamkan sensor, prosesor kecil, dan modul konektivitas pada perangkat tersebut. Dampak positifnya sangat luas: rumah pintar yang mengoptimalkan konsumsi energi secara otomatis, kota pintar dengan manajemen lalu lintas cerdas, pemantauan kesehatan real-time melalui wearable, dan efisiensi industri melalui pemeliharaan prediktif mesin. Namun risiko keamanannya signifikan—setiap perangkat IoT adalah titik masuk potensial bagi peretas, dan banyak perangkat IoT murah memiliki keamanan yang lemah."},
    {"q": "Bagaimana cara kerja pengenalan wajah (face recognition) pada smartphone?", "a": "Pengenalan wajah modern bekerja dengan cara menggunakan deep learning, khususnya Convolutional Neural Network (CNN). Pertama, kamera menangkap gambar wajah dan mendeteksi titik-titik kunci (landmark) seperti jarak antara mata, bentuk hidung, dan kontur rahang—hingga 68 titik atau lebih. Pola geometris unik ini diubah menjadi representasi numerik yang disebut face embedding—seperti 'sidik jari digital wajah'. Saat verifikasi, wajah baru diubah menjadi embedding dan dibandingkan dengan embedding yang tersimpan menggunakan ukuran jarak matematis. Keakuratannya sangat tinggi, bahkan melampaui kemampuan manusia dalam kondisi ideal. Namun rentan terhadap bias—model yang dilatih dengan data tidak beragam bisa berperforma buruk untuk kelompok etnis tertentu."},
    {"q": "Apa itu big data dan mengapa ukuran data menjadi penting?", "a": "Big data merujuk pada kumpulan data yang terlalu besar, terlalu cepat berubah, atau terlalu beragam untuk diproses dengan alat database konvensional. Konsep ini sering dijelaskan dengan 3V: Volume (ukuran data, bisa petabyte), Velocity (kecepatan data masuk, seperti jutaan tweet per menit), dan Variety (keragaman format: teks, gambar, video, sensor). Untuk memahami mengapa ini penting, ambil contoh: sebuah rumah sakit besar menghasilkan terabyte data pasien setiap hari. Dengan menganalisis big data ini menggunakan ML, pola penyakit bisa terdeteksi lebih awal, dan pengobatan bisa dipersonalisasi. Platform seperti Apache Hadoop dan Spark memungkinkan pemrosesan data sebesar ini secara terdistribusi di ribuan server sekaligus."},
    {"q": "Apa itu transformasi digital dan mengapa perusahaan harus melakukannya?", "a": "Transformasi digital bukan sekadar mengadopsi teknologi baru, melainkan perubahan fundamental dalam cara organisasi beroperasi dan memberikan nilai kepada pelanggan menggunakan teknologi digital. Untuk memahami urgensinya, kita perlu melihat efek disrupsi: Kodak bangkrut bukan karena tidak tahu tentang kamera digital—mereka yang menemukannya—tetapi karena tidak mau mengubah model bisnisnya. Keuntungannya meliputi efisiensi operasional yang signifikan, customer experience yang lebih personal, kemampuan pengambilan keputusan berbasis data, dan fleksibilitas bisnis dalam merespons perubahan pasar. Kerugiannya adalah biaya transformasi yang besar, resistensi budaya dari karyawan, dan risiko keamanan siber yang meningkat seiring digitalisasi."},
    {"q": "Apa perbedaan antara AI generatif dan AI tradisional?", "a": "AI tradisional umumnya dirancang untuk tugas spesifik—klasifikasi gambar, deteksi spam, rekomendasi produk—menggunakan model yang dilatih untuk satu tujuan. AI generatif, sebaliknya, mampu menciptakan konten baru yang belum pernah ada sebelumnya: teks, gambar, musik, video, bahkan kode program. Sistem ini bekerja dengan cara mempelajari distribusi probabilitas dari data latih, kemudian mengambil sampel dari distribusi tersebut untuk menghasilkan output baru yang 'mirip' namun unik. Model seperti GPT untuk teks dan Stable Diffusion untuk gambar adalah contohnya. Implikasinya sangat besar: pekerjaan kreatif yang dulu hanya bisa dilakukan manusia kini bisa diotomasi, memunculkan pertanyaan etika serius tentang hak cipta dan autentisitas konten."},
    {"q": "Mengapa keamanan siber (cybersecurity) semakin kritis di era modern?", "a": "Keamanan siber semakin kritis karena hampir semua aspek kehidupan modern bergantung pada sistem digital—perbankan, infrastruktur listrik, rumah sakit, transportasi. Permukaan serangan (attack surface) terus meluas seiring meningkatnya konektivitas IoT dan kerja jarak jauh. Serangan siber modern sangat canggih: phishing yang menggunakan AI untuk membuat email palsu yang meyakinkan, zero-day exploits yang mengeksploitasi kerentanan yang belum diketahui vendor, dan serangan supply chain yang menarget perangkat lunak yang digunakan jutaan organisasi sekaligus. Keuntungan investasi di cybersecurity adalah perlindungan data dan reputasi. Kerugian mengabaikannya bisa berupa kerugian finansial ratusan miliar rupiah dan hilangnya kepercayaan publik yang sulit dipulihkan."},
    {"q": "Apa itu etika AI dan mengapa ini menjadi isu global?", "a": "Etika AI membahas pertanyaan mendasar tentang bagaimana sistem AI seharusnya berperilaku dan dampaknya terhadap masyarakat. Isu utama meliputi: bias algoritmik — ketika model AI mereplikasi atau memperkuat diskriminasi yang ada di data latih (misalnya sistem rekrutmen yang tidak adil terhadap perempuan); transparansi — model 'black box' seperti deep learning sulit dijelaskan keputusannya; privasi — pengumpulan data masif untuk melatih AI; dan penggantian pekerjaan. Untuk memahami skalanya: ketika AI digunakan dalam sistem peradilan untuk memprediksi kemungkinan residivis, kesalahan sistemis bisa merugikan ribuan orang. Regulasi internasional seperti EU AI Act mencoba mengatur penggunaan AI berisiko tinggi, namun tantangannya adalah bahwa teknologi bergerak jauh lebih cepat dari regulasi."},
    {"q": "Bagaimana cara kerja kompresi data dan mengapa ini penting?", "a": "Kompresi data bekerja dengan cara menemukan dan menghilangkan redundansi dalam data untuk memperkecil ukurannya. Ada dua jenis utama: kompresi lossless yang mempertahankan semua data asli (seperti ZIP atau PNG)-sistem ini bekerja dengan cara mengganti pola berulang dengan representasi yang lebih pendek, misalnya rangkaian 'AAAAAAA' diganti dengan kode '7A'; dan kompresi lossy yang membuang data yang tidak terlalu berpengaruh pada persepsi manusia (seperti JPEG untuk gambar atau MP3 untuk audio)—mata manusia kurang peka terhadap detail frekuensi tinggi, sehingga data tersebut bisa dibuang. Pentingnya kompresi: tanpa kompresi video, streaming Netflix akan membutuhkan bandwidth 50x lebih besar, membuat internet seperti yang kita kenal hari ini tidak mungkin ada."},
    {"q": "Apa itu quantum computing dan bagaimana bedanya dengan komputer biasa?", "a": "Komputer biasa menggunakan bit yang bernilai 0 atau 1. Quantum computing menggunakan qubit yang bisa berada dalam 0, 1, dan superposisi keduanya secara bersamaan—berkat prinsip mekanika kuantum. Ini seperti perbedaan antara senter yang bisa menyala atau mati, dengan lampu yang bisa menyala sebagian pada berbagai intensitas sekaligus. Keunggulannya adalah kemampuan memproses banyak kemungkinan secara paralel, sehingga masalah tertentu yang membutuhkan komputer klasik jutaan tahun bisa diselesaikan dalam hitungan menit. Aplikasi potensialnya meliputi kriptografi (dan juga pemecahan enkripsi yang ada), simulasi molekul untuk penemuan obat, dan optimisasi logistik. Tantangannya adalah qubit sangat rapuh—kesalahan terjadi karena gangguan lingkungan sekecil apapun, itulah mengapa quantum computer membutuhkan suhu mendekati nol mutlak."},
    {"q": "Apa itu API dan mengapa menjadi fondasi ekosistem digital modern?", "a": "API (Application Programming Interface) adalah kontrak yang mendefinisikan bagaimana dua aplikasi berkomunikasi satu sama lain—seperti pramusaji di restoran: Anda tidak perlu masuk ke dapur, cukup sampaikan pesanan ke pramusaji yang meneruskannya ke koki dan membawa hasilnya kembali. Sistem ini bekerja dengan cara memberikan endpoint URL yang menerima permintaan (request) dalam format tertentu dan mengembalikan data (response), biasanya dalam format JSON. API menjadi fondasi ekosistem digital karena memungkinkan integrasi tanpa batas: aplikasi ojek online menggunakan API Google Maps untuk peta, API payment gateway untuk pembayaran, dan API SMS untuk notifikasi. Tanpa API, setiap perusahaan harus membangun semuanya dari nol. Dengan API, ekosistem digital menjadi seperti LEGO—komponen yang bisa saling disambungkan."},
    {"q": "Bagaimana algoritma sorting bekerja dan mengapa efisiensinya penting?", "a": "Algoritma sorting mengurutkan data—angka, nama, atau objek apapun. Untuk memahami mengapa efisiensi itu krusial, bayangkan mengurutkan 1 juta nama. Bubble sort, yang membandingkan dua elemen bersebelahan secara berulang, membutuhkan hingga 1 triliun perbandingan. Quick sort, yang membagi data berdasarkan pivot secara rekursif, hanya membutuhkan sekitar 20 juta perbandingan—perbedaan 50.000 kali. Efisiensi ini diukur dengan notasi Big O: O(n²) untuk bubble sort vs O(n log n) untuk quick sort. Di dunia nyata, ini berarti perbedaan antara pencarian yang selesai dalam 0.1 detik versus 83 menit pada dataset yang sama. Itulah mengapa Google, database, dan sistem operasi menggunakan algoritma yang dioptimalkan—pilihan algoritma secara langsung menentukan pengalaman pengguna."},

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
    
    # ============================================================
    # KETERBATASAN & KEJUJURAN AI
    # ============================================================
    {"q": "Apakah kamu selalu benar?", "a": "Tidak, saya bisa saja salah. Saya adalah AI dengan keterbatasan dan pengetahuan saya terbatas pada data training. Selalu verifikasi informasi penting dari sumber terpercaya."},
    {"q": "Apakah kamu bisa salah?", "a": "Ya, saya bisa salah. Saya hanya AI yang belajar dari data dan tidak selalu akurat. Mohon periksa kembali informasi yang saya berikan."},
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

    # ============================================================
    # MATEMATIKA MENENGAH - SOAL CERITA & PENALARAN BERTAHAP
    # ============================================================
    {
        "q": "Pak Budi membeli 5 kg apel dengan harga Rp18.000 per kg dan 3 kg jeruk dengan harga Rp12.000 per kg. Ia membayar dengan uang Rp200.000. Berapa kembalian yang ia terima?",
        "a": "Diketahui:\n- Apel: 5 kg × Rp18.000 = Rp90.000\n- Jeruk: 3 kg × Rp12.000 = Rp36.000\n- Uang yang dibayarkan: Rp200.000\n\nDitanya: Berapa kembalian?\n\nPenyelesaian:\nTotal belanja = Rp90.000 + Rp36.000 = Rp126.000\nKembalian = Rp200.000 - Rp126.000 = Rp74.000\n\nJadi, jawabannya adalah Pak Budi menerima kembalian sebesar Rp74.000."
    },
    {
        "q": "Sebuah mobil menempuh jarak 240 km dalam waktu 3 jam. Jika kecepatan mobil dinaikkan 20 km/jam, berapa waktu yang diperlukan untuk menempuh jarak yang sama?",
        "a": "Diketahui:\n- Jarak: 240 km\n- Waktu awal: 3 jam\n- Kecepatan awal: 240 ÷ 3 = 80 km/jam\n- Kecepatan baru: 80 + 20 = 100 km/jam\n\nDitanya: Waktu dengan kecepatan baru?\n\nPenyelesaian:\nWaktu = Jarak ÷ Kecepatan = 240 ÷ 100 = 2,4 jam = 2 jam 24 menit\n\nJadi, jawabannya adalah waktu yang diperlukan adalah 2 jam 24 menit."
    },
    {
        "q": "Nilai ulangan matematika 6 siswa adalah 70, 85, 90, 75, 80, dan 60. Berapa rata-rata nilai mereka? Apakah rata-rata tersebut di atas 75?",
        "a": "Diketahui:\n- Nilai: 70, 85, 90, 75, 80, 60\n- Jumlah siswa: 6\n\nDitanya: Rata-rata nilai dan apakah di atas 75?\n\nPenyelesaian:\nJumlah nilai = 70 + 85 + 90 + 75 + 80 + 60 = 460\nRata-rata = 460 ÷ 6 = 76,67\n\nKarena 76,67 > 75, maka rata-rata berada di atas 75.\n\nJadi, jawabannya adalah rata-rata nilai adalah 76,67 dan ya, nilainya di atas 75."
    },
    {
        "q": "Seorang pedagang membeli 50 buah mangga dengan harga Rp3.000 per buah. Ia menjual 40 buah dengan harga Rp5.000 per buah, dan sisanya rusak. Berapa keuntungan atau kerugian pedagang tersebut?",
        "a": "Diketahui:\n- Modal: 50 × Rp3.000 = Rp150.000\n- Penjualan: 40 × Rp5.000 = Rp200.000\n- 10 buah rusak (tidak terjual)\n\nDitanya: Untung atau rugi?\n\nPenyelesaian:\nKeuntungan = Hasil jual - Modal = Rp200.000 - Rp150.000 = Rp50.000\n\nJadi, jawabannya adalah pedagang mendapat keuntungan sebesar Rp50.000."
    },
    {
        "q": "Sebuah kolam renang dapat diisi oleh pipa A dalam 6 jam dan pipa B dalam 4 jam. Jika kedua pipa dibuka bersamaan, berapa jam kolam akan penuh?",
        "a": "Diketahui:\n- Pipa A mengisi 1/6 kolam per jam\n- Pipa B mengisi 1/4 kolam per jam\n\nDitanya: Waktu pengisian jika kedua pipa dibuka bersama?\n\nPenyelesaian:\nKecepatan gabungan = 1/6 + 1/4 = 2/12 + 3/12 = 5/12 kolam per jam\nWaktu = 1 ÷ (5/12) = 12/5 = 2,4 jam = 2 jam 24 menit\n\nJadi, jawabannya adalah kolam akan penuh dalam 2 jam 24 menit."
    },
    {
        "q": "Toko A menjual baju seharga Rp120.000 dengan diskon 25%. Toko B menjual baju yang sama seharga Rp100.000 dengan diskon 10%. Manakah yang lebih murah?",
        "a": "Diketahui:\n- Toko A: Rp120.000 diskon 25%\n- Toko B: Rp100.000 diskon 10%\n\nDitanya: Mana yang lebih murah?\n\nPenyelesaian:\nHarga Toko A setelah diskon = Rp120.000 × (1 - 0,25) = Rp120.000 × 0,75 = Rp90.000\nHarga Toko B setelah diskon = Rp100.000 × (1 - 0,10) = Rp100.000 × 0,90 = Rp90.000\n\nKedua harga sama setelah diskon.\n\nJadi, jawabannya adalah kedua toko sama-sama menjual dengan harga Rp90.000 setelah diskon."
    },
    {
        "q": "Sebuah kereta berangkat pukul 07.30 dan tiba pukul 11.15. Jarak yang ditempuh 285 km. Berapa kecepatan rata-rata kereta tersebut?",
        "a": "Diketahui:\n- Berangkat: 07.30, tiba: 11.15\n- Jarak: 285 km\n\nDitanya: Kecepatan rata-rata?\n\nPenyelesaian:\nLama perjalanan = 11.15 - 07.30 = 3 jam 45 menit = 3,75 jam\nKecepatan = Jarak ÷ Waktu = 285 ÷ 3,75 = 76 km/jam\n\nJadi, jawabannya adalah kecepatan rata-rata kereta adalah 76 km/jam."
    },
    {
        "q": "Ibu menabung Rp500.000 di bank dengan bunga sederhana 8% per tahun. Berapa total uang ibu setelah 3 tahun?",
        "a": "Diketahui:\n- Modal awal (P): Rp500.000\n- Suku bunga (r): 8% per tahun\n- Waktu (t): 3 tahun\n\nDitanya: Total uang setelah 3 tahun?\n\nPenyelesaian:\nBunga = P × r × t = Rp500.000 × 0,08 × 3 = Rp120.000\nTotal = Modal + Bunga = Rp500.000 + Rp120.000 = Rp620.000\n\nJadi, jawabannya adalah total uang ibu setelah 3 tahun adalah Rp620.000."
    },
    {
        "q": "Sebuah persegi panjang memiliki keliling 54 cm. Jika panjangnya dua kali lebarnya, berapakah luas persegi panjang tersebut?",
        "a": "Diketahui:\n- Keliling = 54 cm\n- Panjang (p) = 2 × lebar (l)\n\nDitanya: Luas persegi panjang?\n\nPenyelesaian:\nKeliling = 2 × (p + l)\n54 = 2 × (2l + l)\n54 = 2 × 3l\n54 = 6l\nl = 9 cm, sehingga p = 18 cm\n\nLuas = p × l = 18 × 9 = 162 cm²\n\nJadi, jawabannya adalah luas persegi panjang tersebut adalah 162 cm²."
    },
    {
        "q": "Ayah berjalan ke pasar dengan kecepatan 4 km/jam, lalu pulang naik ojek dengan kecepatan 20 km/jam. Jika jarak rumah ke pasar 2 km, berapa total waktu perjalanan ayah?",
        "a": "Diketahui:\n- Jarak = 2 km\n- Kecepatan jalan kaki = 4 km/jam\n- Kecepatan ojek = 20 km/jam\n\nDitanya: Total waktu perjalanan?\n\nPenyelesaian:\nWaktu pergi (jalan kaki) = 2 ÷ 4 = 0,5 jam = 30 menit\nWaktu pulang (ojek) = 2 ÷ 20 = 0,1 jam = 6 menit\nTotal waktu = 30 + 6 = 36 menit\n\nJadi, jawabannya adalah total waktu perjalanan ayah adalah 36 menit."
    },
    {
        "q": "Dalam sebuah kelas ada 30 siswa. Sebanyak 18 siswa suka matematika, 14 siswa suka IPA, dan 8 siswa suka keduanya. Berapa siswa yang tidak suka keduanya?",
        "a": "Diketahui:\n- Total siswa: 30\n- Suka matematika: 18\n- Suka IPA: 14\n- Suka keduanya: 8\n\nDitanya: Siswa yang tidak suka keduanya?\n\nPenyelesaian (menggunakan diagram Venn):\nSiswa yang suka salah satu atau keduanya = 18 + 14 - 8 = 24\nSiswa yang tidak suka keduanya = 30 - 24 = 6\n\nJadi, jawabannya adalah 6 siswa tidak suka keduanya."
    },
    {
        "q": "Harga sebuah sepeda motor turun 15% menjadi Rp17.000.000. Berapa harga sepeda motor sebelum turun?",
        "a": "Diketahui:\n- Harga setelah turun: Rp17.000.000\n- Persentase penurunan: 15%\n\nDitanya: Harga sebelum turun?\n\nPenyelesaian:\nHarga setelah turun = 85% dari harga awal\nRp17.000.000 = 85% × Harga awal\nHarga awal = Rp17.000.000 ÷ 0,85 = Rp20.000.000\n\nJadi, jawabannya adalah harga sepeda motor sebelum turun adalah Rp20.000.000."
    },
    {
        "q": "Seorang pekerja dibayar Rp120.000 per hari. Ia bekerja 22 hari dalam sebulan, namun dipotong pajak 5%. Berapa gaji bersih yang ia terima?",
        "a": "Diketahui:\n- Upah per hari: Rp120.000\n- Hari kerja: 22 hari\n- Pajak: 5%\n\nDitanya: Gaji bersih?\n\nPenyelesaian:\nGaji kotor = Rp120.000 × 22 = Rp2.640.000\nPotongan pajak = 5% × Rp2.640.000 = Rp132.000\nGaji bersih = Rp2.640.000 - Rp132.000 = Rp2.508.000\n\nJadi, jawabannya adalah gaji bersih yang diterima adalah Rp2.508.000."
    },
    {
        "q": "Dua kota A dan B berjarak 300 km. Mobil I berangkat dari A dan mobil II berangkat dari B secara bersamaan menuju satu sama lain. Kecepatan mobil I 60 km/jam dan mobil II 90 km/jam. Berapa lama sampai keduanya bertemu?",
        "a": "Diketahui:\n- Jarak A ke B: 300 km\n- Kecepatan mobil I: 60 km/jam\n- Kecepatan mobil II: 90 km/jam\n- Berangkat bersamaan dari arah berlawanan\n\nDitanya: Waktu hingga bertemu?\n\nPenyelesaian:\nKecepatan pendekatan = 60 + 90 = 150 km/jam\nWaktu bertemu = 300 ÷ 150 = 2 jam\n\nJadi, jawabannya adalah kedua mobil akan bertemu setelah 2 jam."
    },
    {
        "q": "Sebuah toko memberikan diskon kedua sebesar 10% setelah diskon pertama 20% dari harga asli Rp500.000. Berapa harga akhir barang tersebut?",
        "a": "Diketahui:\n- Harga asli: Rp500.000\n- Diskon pertama: 20%\n- Diskon kedua: 10% dari harga setelah diskon pertama\n\nDitanya: Harga akhir?\n\nPenyelesaian:\nSetelah diskon pertama = Rp500.000 × (1 - 0,20) = Rp400.000\nSetelah diskon kedua = Rp400.000 × (1 - 0,10) = Rp360.000\n\nJadi, jawabannya adalah harga akhir barang tersebut adalah Rp360.000."
    },
    {
        "q": "Sebuah tangki air berbentuk balok berukuran panjang 2 m, lebar 1,5 m, dan tinggi 1 m. Jika tangki terisi 3/4 penuh, berapa liter air di dalamnya? (1 m³ = 1000 liter)",
        "a": "Diketahui:\n- Panjang: 2 m, Lebar: 1,5 m, Tinggi: 1 m\n- Terisi 3/4 penuh\n- 1 m³ = 1000 liter\n\nDitanya: Jumlah air dalam liter?\n\nPenyelesaian:\nVolume tangki = 2 × 1,5 × 1 = 3 m³\nVolume air = 3/4 × 3 = 2,25 m³\nDalam liter = 2,25 × 1000 = 2.250 liter\n\nJadi, jawabannya adalah tangki berisi 2.250 liter air."
    },
    {
        "q": "Rini menyelesaikan 2/5 pekerjaan rumah pada hari pertama dan 1/3 sisanya pada hari kedua. Berapa bagian pekerjaan yang belum selesai?",
        "a": "Diketahui:\n- Hari pertama: 2/5 dari total pekerjaan\n- Hari kedua: 1/3 dari sisa pekerjaan\n\nDitanya: Bagian pekerjaan yang belum selesai?\n\nPenyelesaian:\nSisa setelah hari pertama = 1 - 2/5 = 3/5\nYang dikerjakan hari kedua = 1/3 × 3/5 = 1/5\nTotal selesai = 2/5 + 1/5 = 3/5\nBelum selesai = 1 - 3/5 = 2/5\n\nJadi, jawabannya adalah 2/5 pekerjaan belum selesai."
    },
    {
        "q": "Suatu tempat wisata dikunjungi 1.200 orang selama 5 hari berturut-turut dengan rata-rata 240 orang per hari. Pada hari ke-6, jumlah pengunjung 300 orang. Berapa rata-rata pengunjung selama 6 hari?",
        "a": "Diketahui:\n- Total pengunjung 5 hari: 1.200 orang\n- Hari ke-6: 300 orang\n\nDitanya: Rata-rata selama 6 hari?\n\nPenyelesaian:\nTotal pengunjung 6 hari = 1.200 + 300 = 1.500 orang\nRata-rata = 1.500 ÷ 6 = 250 orang per hari\n\nJadi, jawabannya adalah rata-rata pengunjung selama 6 hari adalah 250 orang per hari."
    },
    {
        "q": "Sebuah pipa bocor mengeluarkan 50 ml air per menit. Berapa liter air yang terbuang sia-sia dalam satu hari?",
        "a": "Diketahui:\n- Kebocoran: 50 ml per menit\n- Waktu: 1 hari = 24 jam = 1.440 menit\n\nDitanya: Total air yang terbuang dalam liter?\n\nPenyelesaian:\nTotal air = 50 ml × 1.440 = 72.000 ml\nDalam liter = 72.000 ÷ 1.000 = 72 liter\n\nJadi, jawabannya adalah 72 liter air terbuang sia-sia setiap harinya."
    },
    {
        "q": "Di sebuah sekolah, perbandingan siswa laki-laki dan perempuan adalah 3 : 5. Jika jumlah siswa perempuan 200 orang, berapa total seluruh siswa?",
        "a": "Diketahui:\n- Perbandingan laki-laki : perempuan = 3 : 5\n- Jumlah perempuan: 200 orang\n\nDitanya: Total seluruh siswa?\n\nPenyelesaian:\nSatu bagian = 200 ÷ 5 = 40 orang\nJumlah laki-laki = 3 × 40 = 120 orang\nTotal siswa = 120 + 200 = 320 orang\n\nJadi, jawabannya adalah total seluruh siswa adalah 320 orang."
    },
    {
        "q": "Seorang penjahit dapat membuat 3 baju dalam 2 hari. Berapa baju yang bisa dibuat dalam 30 hari? Dan berapa hari untuk membuat 45 baju?",
        "a": "Diketahui:\n- 3 baju dalam 2 hari\n- Target 1: berapa baju dalam 30 hari\n- Target 2: berapa hari untuk 45 baju\n\nDitanya: Jumlah baju dan hari yang diperlukan?\n\nPenyelesaian:\nLaju kerja = 3/2 baju per hari = 1,5 baju per hari\n\nBaju dalam 30 hari = 1,5 × 30 = 45 baju\nHari untuk 45 baju = 45 ÷ 1,5 = 30 hari\n\nJadi, jawabannya adalah penjahit bisa membuat 45 baju dalam 30 hari, dan untuk membuat 45 baju dibutuhkan 30 hari."
    },
    {
        "q": "Hana memiliki uang Rp350.000. Ia menghabiskan 40% untuk buku, 25% untuk alat tulis, dan sisanya ia tabung. Berapa rupiah yang ditabung Hana?",
        "a": "Diketahui:\n- Total uang: Rp350.000\n- Untuk buku: 40%\n- Untuk alat tulis: 25%\n- Sisanya ditabung\n\nDitanya: Uang yang ditabung?\n\nPenyelesaian:\nPersentase ditabung = 100% - 40% - 25% = 35%\nUang yang ditabung = 35% × Rp350.000 = 0,35 × Rp350.000 = Rp122.500\n\nJadi, jawabannya adalah Hana menabung sebesar Rp122.500."
    },
    {
        "q": "Sebuah proyek dijadwalkan selesai dalam 20 hari dengan 15 pekerja. Setelah 8 hari, 5 pekerja sakit dan tidak bisa bekerja. Berapa hari tambahan yang diperlukan untuk menyelesaikan proyek?",
        "a": "Diketahui:\n- Rencana: 20 hari × 15 pekerja = 300 hari-orang (total pekerjaan)\n- Sudah dikerjakan: 8 hari × 15 pekerja = 120 hari-orang\n- Sisa pekerjaan: 300 - 120 = 180 hari-orang\n- Pekerja tersisa: 15 - 5 = 10 pekerja\n\nDitanya: Berapa hari untuk menyelesaikan sisa pekerjaan?\n\nPenyelesaian:\nHari yang diperlukan = 180 ÷ 10 = 18 hari\nRencana sisa hari = 20 - 8 = 12 hari\nTambahan hari = 18 - 12 = 6 hari\n\nJadi, jawabannya adalah proyek membutuhkan 6 hari tambahan dari jadwal semula."
    },
    {
        "q": "Nilai median dari data berikut: 45, 72, 38, 91, 56, 64, 38. Tentukan juga nilai modusnya.",
        "a": "Diketahui:\n- Data: 45, 72, 38, 91, 56, 64, 38\n- Jumlah data: 7\n\nDitanya: Median dan modus?\n\nPenyelesaian:\nUrutkan data: 38, 38, 45, 56, 64, 72, 91\nMedian = nilai tengah = data ke-4 = 56\nModus = nilai yang paling sering muncul = 38 (muncul 2 kali)\n\nJadi, jawabannya adalah median data tersebut adalah 56 dan modusnya adalah 38."
    },
    {
        "q": "Sebuah becak melaju sejauh 1,8 km dalam 12 menit. Sebuah sepeda melaju dengan kecepatan 15 km/jam. Manakah yang lebih cepat dan berapa selisih kecepatannya?",
        "a": "Diketahui:\n- Becak: 1,8 km dalam 12 menit\n- Sepeda: 15 km/jam\n\nDitanya: Mana yang lebih cepat dan berapa selisihnya?\n\nPenyelesaian:\nKecepatan becak = 1,8 km ÷ (12/60) jam = 1,8 ÷ 0,2 = 9 km/jam\nKecepatan sepeda = 15 km/jam\n\nSepeda lebih cepat.\nSelisih kecepatan = 15 - 9 = 6 km/jam\n\nJadi, jawabannya adalah sepeda lebih cepat dengan selisih kecepatan 6 km/jam dibandingkan becak."
    },

    # ============================================================
    # KESEHATAN & KEDOKTERAN (PENJELASAN MEKANISME)
    # ============================================================
    {"q": "Mengapa kita harus mencuci tangan sebelum makan?", "a": "Alasan mengapa mencuci tangan sebelum makan sangat penting adalah karena tangan kita bersentuhan dengan ratusan permukaan sepanjang hari dan menjadi perantara mikroorganisme patogen seperti bakteri E. coli, Salmonella, dan virus Norovirus. Secara fisiologis, ketika kita makan dengan tangan yang terkontaminasi, patogen tersebut masuk ke saluran pencernaan dan dapat menyebabkan infeksi. Sabun bekerja dengan merusak membran lipid bakteri dan virus, sementara aliran air membilas partikel fisik. WHO memperkirakan bahwa kebiasaan cuci tangan yang benar dapat mengurangi diare hingga 30% dan infeksi saluran pernapasan hingga 20%."},
    {"q": "Mengapa orang yang kurang tidur mudah sakit?", "a": "Secara fisiologis, selama tidur tubuh memproduksi sitokin, yaitu protein yang berperan sebagai sinyal komunikasi sistem imun. Sitokin tertentu diperlukan untuk melawan infeksi dan peradangan. Penelitian menunjukkan bahwa kurang tidur (kurang dari 6 jam) menurunkan produksi sel T dan antibodi, sehingga respons imun melemah. Selain itu, tidur adalah waktu tubuh memperbaiki kerusakan sel dan melakukan regenerasi. Kurang tidur juga meningkatkan kadar kortisol (hormon stres) yang secara langsung menekan fungsi sistem imun, membuat tubuh lebih rentan terhadap infeksi virus dan bakteri."},
    {"q": "Kenapa merokok bisa menyebabkan kanker paru-paru?", "a": "Alasan mengapa rokok menyebabkan kanker paru-paru adalah karena asap rokok mengandung lebih dari 70 zat karsinogen, termasuk benzopiren, nitrosamin, dan formaldehida. Secara molekuler, zat-zat ini berikatan dengan DNA sel epitel saluran napas dan menyebabkan mutasi genetik, khususnya pada gen penekan tumor seperti p53 dan gen proto-onkogen seperti RAS. Mutasi yang terakumulasi merusak mekanisme kontrol siklus sel normal, sehingga sel mulai membelah tak terkendali. Penelitian menunjukkan bahwa perokok memiliki risiko 15-30 kali lebih tinggi terkena kanker paru dibandingkan bukan perokok, dan risiko ini berkorelasi langsung dengan jumlah rokok dan lama merokok."},
    {"q": "Mengapa diabetes tipe 2 bisa dicegah dengan pola makan sehat?", "a": "Alasan mengapa pola makan sehat mencegah diabetes tipe 2 berkaitan dengan mekanisme resistensi insulin. Konsumsi karbohidrat sederhana dan lemak trans secara berlebihan menyebabkan lonjakan glukosa darah berulang, memaksa pankreas memproduksi insulin dalam jumlah besar. Seiring waktu, sel-sel tubuh terutama sel otot dan hati menjadi kurang sensitif terhadap insulin, kondisi yang disebut resistensi insulin. Makanan berserat tinggi memperlambat penyerapan glukosa, menstabilkan kadar gula darah, dan meningkatkan sensitivitas insulin. Penelitian menunjukkan bahwa kehilangan berat badan 5-7% saja sudah dapat mengurangi risiko diabetes hingga 58% pada populasi berisiko tinggi."},
    {"q": "Mengapa olahraga teratur baik untuk kesehatan jantung?", "a": "Secara fisiologis, olahraga aerobik seperti lari atau bersepeda meningkatkan beban kerja jantung sehingga memicu adaptasi struktural: otot jantung menjadi lebih tebal dan kuat (hipertrofi fisiologis), volume sekuncup meningkat, dan denyut nadi istirahat menurun. Olahraga juga merangsang produksi enzim lipoprotein lipase yang meningkatkan kadar HDL (kolesterol baik) dan menurunkan LDL serta trigliserida. Selain itu, olahraga meningkatkan elastisitas pembuluh darah dan produksi nitrat oksida, yang melebarkan arteri dan menurunkan tekanan darah. Penelitian menunjukkan bahwa 150 menit olahraga intensitas sedang per minggu dapat menurunkan risiko penyakit jantung koroner hingga 35%."},
    {"q": "Mengapa tubuh memerlukan serat dalam makanan?", "a": "Alasan mengapa serat sangat penting bagi tubuh adalah karena serat tidak dapat dicerna oleh enzim pencernaan manusia, namun berperan krusial dalam dua cara utama. Pertama, serat larut (seperti pektin dan beta-glukan) membentuk gel di usus yang memperlambat penyerapan glukosa dan kolesterol, membantu mengontrol kadar gula darah dan menurunkan LDL. Kedua, serat tidak larut menambah bulk pada feses dan mempercepat transit usus, mencegah sembelit dan mengurangi paparan racun terhadap dinding kolon. Secara tambahan, serat adalah prebiotik yang memberi makan bakteri baik usus (microbiome), yang memproduksi asam lemak rantai pendek penting untuk kesehatan sel usus dan sistem imun."},
    {"q": "Apa yang terjadi dalam tubuh saat kita mengalami stres?", "a": "Secara fisiologis, ketika otak mendeteksi ancaman, hipotalamus mengaktifkan sumbu HPA (hypothalamic-pituitary-adrenal) dan sistem saraf simpatis. Kelenjar adrenal melepaskan adrenalin (epinefrin) dalam hitungan detik, menyebabkan denyut jantung meningkat, bronkus melebar, dan aliran darah dialihkan ke otot. Bersamaan, kortisol dilepaskan untuk memobilisasi cadangan energi dengan memecah glikogen dan lemak. Respons ini bermanfaat jangka pendek ('fight or flight'), namun stres kronis menyebabkan kadar kortisol tinggi terus-menerus, yang melemahkan imunitas, meningkatkan tekanan darah, mengganggu tidur, dan merusak hippocampus di otak yang berperan dalam memori."},
    {"q": "Mengapa penting untuk mendapatkan vaksinasi?", "a": "Alasan mengapa vaksinasi sangat penting berkaitan dengan prinsip imunitas adaptif. Vaksin memperkenalkan antigen (fragmen protein atau patogen yang dilemahkan) ke tubuh tanpa menyebabkan penyakit. Sistem imun merespons dengan memproduksi antibodi spesifik dan sel memori (sel B dan sel T memori). Jika di kemudian hari terpapar patogen asli, sel memori ini memungkinkan respons imun yang jauh lebih cepat dan kuat sebelum infeksi berkembang. Penelitian menunjukkan vaksin seperti campak memiliki efektivitas 97% dalam mencegah penyakit. Selain proteksi individual, vaksinasi massal menciptakan herd immunity, melindungi orang-orang yang tidak bisa divaksin seperti bayi dan orang dengan imunodefisiensi."},
    {"q": "Mengapa terlalu banyak garam berbahaya bagi tekanan darah?", "a": "Secara fisiologis, natrium (Na+) dalam garam berperan dalam mengatur keseimbangan cairan tubuh. Ketika asupan natrium berlebih, konsentrasi natrium dalam darah meningkat, sehingga otak memberi sinyal rasa haus dan ginjal menahan lebih banyak air untuk mengencerkan natrium. Volume darah meningkat, menyebabkan tekanan yang lebih besar pada dinding pembuluh darah, inilah hipertensi. Jangka panjang, tekanan tinggi merusak dinding arteri, memicu pembentukan plak aterosklerosis, dan meningkatkan beban kerja jantung. WHO merekomendasikan asupan garam tidak lebih dari 5 gram per hari. Penelitian menunjukkan mengurangi asupan garam dapat menurunkan tekanan darah sistolik rata-rata 5 mmHg."},
    {"q": "Apa yang terjadi pada tubuh ketika kita berpuasa?", "a": "Secara metabolik, tubuh melewati beberapa fase saat berpuasa. Pada 6-24 jam pertama, kadar insulin menurun dan glukagon meningkat, memicu hati untuk memecah glikogen menjadi glukosa. Setelah cadangan glikogen habis (sekitar 24 jam), tubuh beralih ke glukoneogenesis (pembentukan glukosa dari asam amino) dan ketogenesis (pembentukan keton dari asam lemak). Keton menjadi sumber energi alternatif untuk otak. Penelitian menunjukkan puasa intermiten meningkatkan sensitivitas insulin, menurunkan inflamasi, dan memicu autofagi, yaitu proses pembersihan sel dari protein rusak yang dikaitkan dengan pencegahan penyakit degeneratif."},
    {"q": "Mengapa kurang minum air putih berbahaya bagi tubuh?", "a": "Alasan mengapa dehidrasi berbahaya adalah karena air terlibat dalam hampir setiap proses biokimia tubuh. Air berfungsi sebagai pelarut reaksi enzimatik, media transportasi nutrisi dan oksigen melalui darah, pengatur suhu tubuh melalui keringat, dan pelumas sendi. Ketika tubuh kekurangan cairan hanya 2%, performa kognitif dan fisik sudah menurun secara signifikan. Pada tingkat 5-10%, terjadi gangguan serius seperti sakit kepala, kebingungan, dan penurunan tekanan darah. Secara khusus, dehidrasi memekatkan urin dan meningkatkan risiko batu ginjal serta infeksi saluran kemih. Dewasa umumnya membutuhkan sekitar 2-3 liter cairan per hari, tergantung aktivitas dan suhu lingkungan."},
    {"q": "Mengapa lemak tidak selamanya jahat bagi tubuh?", "a": "Penelitian menunjukkan bahwa tidak semua lemak berdampak negatif dan tubuh sesungguhnya membutuhkan lemak untuk berbagai fungsi vital. Lemak tak jenuh tunggal (seperti dalam minyak zaitun dan alpukat) dan lemak tak jenuh ganda omega-3 (dalam ikan berlemak) menurunkan LDL sambil mempertahankan atau meningkatkan HDL, mengurangi inflamasi, dan mendukung kesehatan jantung. Secara struktural, lemak merupakan komponen utama membran sel, selubung mielin saraf, dan bahan baku hormon steroid seperti estrogen dan testosteron. Lemak juga melarutkan vitamin A, D, E, K yang larut lemak. Yang berbahaya adalah lemak trans (dalam makanan olahan) dan lemak jenuh berlebihan yang meningkatkan LDL dan peradangan."},
    {"q": "Bagaimana cara pertolongan pertama yang benar saat seseorang tersedak?", "a": "Alasan mengapa teknik Heimlich efektif dalam mengatasi tersedak berkaitan dengan prinsip tekanan udara. Ketika benda asing menyumbat saluran napas, metode yang tepat adalah melakukan abdominal thrust (Heimlich maneuver). Cara kerjanya: berdiri di belakang korban, kepalkan satu tangan dan letakkan sedikit di atas pusar, genggam dengan tangan lain, lalu tekan ke dalam dan ke atas dengan gerakan cepat. Tekanan mendadak ini memampatkan rongga dada, meningkatkan tekanan udara di paru-paru jauh melebihi tekanan atmosfer, sehingga menciptakan aliran udara paksa yang dapat melontarkan benda asing ke luar. Pada bayi di bawah 1 tahun, teknik yang digunakan berbeda yaitu back blows dan chest thrusts karena anatomi yang lebih rentan."},
    {"q": "Mengapa depresi dianggap penyakit medis serius, bukan sekadar kesedihan biasa?", "a": "Alasan mengapa depresi adalah kondisi medis nyata berkaitan dengan perubahan biologis yang terukur di otak. Penelitian dengan pencitraan otak menunjukkan bahwa depresi melibatkan penurunan kadar neurotransmiter seperti serotonin, dopamin, dan norepinefrin di sinapsis, serta perubahan struktural seperti berkurangnya volume hippocampus akibat stres oksidatif. Secara neuroendokrin, penderita depresi sering menunjukkan disregulasi sumbu HPA dengan kadar kortisol yang abnormal. Inflamasi sistemik yang ditandai dengan peningkatan sitokin pro-inflamasi juga konsisten ditemukan. Tanpa pengobatan, depresi berat dapat menyebabkan neurodegenerasi progresif. Ini berbeda mendasar dari kesedihan normal yang tidak mengubah fungsi otak secara struktural."},
    {"q": "Mengapa indeks massa tubuh (IMT) yang tinggi meningkatkan risiko penyakit?", "a": "Secara fisiologis, jaringan lemak terutama lemak visceral di sekitar organ perut bukan hanya penyimpan energi pasif, melainkan organ endokrin aktif. Lemak visceral menyekresikan adipokin pro-inflamasi seperti TNF-alpha dan interleukin-6, serta resistin yang mengganggu sinyal insulin. Kelebihan lemak visceral juga memicu inflamasi kronis tingkat rendah yang merusak pembuluh darah, meningkatkan risiko aterosklerosis. Secara mekanik, obesitas meningkatkan beban pada sendi, jantung, dan sistem pernapasan. Penelitian menunjukkan bahwa IMT di atas 30 kg/m persegi berkaitan dengan peningkatan risiko diabetes tipe 2 sebesar 7 kali lipat, penyakit jantung 2 kali lipat, dan beberapa jenis kanker karena kadar estrogen berlebih dari jaringan lemak."},
    {"q": "Mengapa konsumsi alkohol berlebihan merusak hati?", "a": "Secara biokimia, hati adalah organ utama yang memetabolisme alkohol (etanol). Alkohol diubah oleh enzim alkohol dehidrogenase (ADH) menjadi asetaldehida, senyawa sangat toksik yang merusak protein dan DNA sel hati. Asetaldehida kemudian diubah menjadi asetat oleh asetaldehida dehidrogenase. Proses metabolisme ini menghasilkan NADH berlebih yang menggeser metabolisme lemak, menyebabkan akumulasi lemak di sel hati (steatosis hepatik atau fatty liver). Konsumsi kronik memicu inflamasi (hepatitis alkoholik) dan akhirnya fibrosis, yaitu penggantian sel hati fungsional dengan jaringan parut (sirosis). Ketika lebih dari 75% sel hati rusak, fungsi vital seperti sintesis protein, detoksifikasi, dan produksi faktor pembekuan darah terganggu secara serius."},
    {"q": "Mengapa olahraga dapat membantu mengatasi kecemasan dan depresi?", "a": "Penelitian menunjukkan bahwa olahraga memiliki efek antidepresan dan ansiolitik yang setara dengan obat-obatan ringan melalui beberapa mekanisme. Secara neurokimia, olahraga aerobik meningkatkan pelepasan endorfin dan endocannabinoid, menghasilkan euforia dan perasaan tenang. Olahraga juga meningkatkan kadar BDNF (brain-derived neurotrophic factor), protein yang merangsang pertumbuhan neuron baru di hippocampus, area otak yang rentan mengecil pada depresi. Secara hormonal, olahraga menurunkan kortisol dan adrenalin, mengurangi respons stres. Aspek psikologis seperti pencapaian tujuan dan peningkatan kepercayaan diri turut berkontribusi. WHO merekomendasikan 150 menit aktivitas fisik sedang per minggu sebagai salah satu strategi kesehatan mental."},
    {"q": "Apa yang menyebabkan tekanan darah tinggi dan bagaimana cara mencegahnya?", "a": "Alasan mengapa hipertensi berkembang melibatkan beberapa mekanisme. Secara fisiologis, tekanan darah ditentukan oleh curah jantung (cardiac output) dan resistensi vaskular perifer. Hipertensi esensial (90% kasus) terjadi karena kombinasi faktor: kekakuan arteri akibat penuaan dan aterosklerosis, aktivasi berlebihan sistem renin-angiotensin-aldosteron yang menahan natrium dan air, serta disfungsi endotel yang mengurangi produksi nitrat oksida (vasodilator alami). Faktor risiko yang bisa dimodifikasi meliputi obesitas, konsumsi garam berlebih, kurang olahraga, stres kronis, dan merokok. Pencegahannya melalui diet DASH (kaya buah, sayuran, whole grain, rendah garam), olahraga teratur, manajemen stres, dan pembatasan alkohol yang terbukti secara klinis menurunkan tekanan darah secara bermakna."},
    {"q": "Mengapa penting bagi ibu hamil untuk mengonsumsi asam folat?", "a": "Alasan mengapa asam folat kritis selama kehamilan berkaitan dengan perannya dalam sintesis DNA dan pembelahan sel. Folat adalah kofaktor esensial untuk menghasilkan nukleotida purin dan timidin yang dibutuhkan untuk replikasi DNA. Selama 4 minggu pertama kehamilan (sering sebelum ibu menyadari kehamilannya), neural tube yang merupakan cikal bakal otak dan sumsum tulang belakang berkembang dan menutup. Kekurangan folat mengganggu penutupan neural tube, menyebabkan defek lahir serius seperti spina bifida (sumsum tulang belakang terbuka) dan anensefali (otak tidak berkembang sempurna). Penelitian menunjukkan suplementasi 400-800 mcg asam folat per hari sebelum dan selama awal kehamilan dapat mengurangi risiko neural tube defect hingga 70%."},
    {"q": "Mengapa paparan sinar matahari pagi bermanfaat bagi kesehatan?", "a": "Secara fotobiologis, sinar ultraviolet B (UVB) dari matahari memicu produksi vitamin D3 di kulit. Sinar UVB mengubah 7-dehidrokolesterol di kulit menjadi pre-vitamin D3, yang kemudian diaktifkan oleh hati dan ginjal menjadi kalsitriol (vitamin D aktif). Kalsitriol mengatur penyerapan kalsium dan fosfor di usus, vital untuk kesehatan tulang dan pencegahan osteoporosis. Penelitian juga menunjukkan vitamin D berperan dalam modulasi imun, regulasi pertumbuhan sel, dan pencegahan beberapa kanker. Selain itu, cahaya matahari pagi yang kaya spektrum biru merangsang suprachiasmatic nucleus di hipotalamus, menyinkronkan ritme sirkadian yang mengatur siklus tidur-bangun dan meningkatkan kualitas tidur malam. Waktu optimal adalah 10-30 menit antara pukul 09.00-10.00 tanpa tabir surya."},
    {"q": "Bagaimana antibiotik bekerja dan mengapa tidak boleh digunakan sembarangan?", "a": "Secara mekanistik, antibiotik bekerja dengan menarget struktur spesifik bakteri yang tidak dimiliki sel manusia. Penisilin menghambat sintesis dinding sel bakteri (peptidoglikan) sehingga bakteri pecah akibat tekanan osmotik. Aminoglikosida menghambat ribosom 30S bakteri, mencegah sintesis protein. Fluorokuinolon menghambat enzim topoisomerase bakteri, mencegah replikasi DNA. Penggunaan sembarangan berbahaya karena antibiotik tidak efektif untuk infeksi virus seperti flu dan COVID-19, namun justru merusak microbiome usus yang sehat. Penggunaan tidak tuntas atau berlebihan juga menciptakan tekanan seleksi yang memunculkan strain bakteri resisten. WHO menyatakan resistensi antibiotik sebagai ancaman kesehatan global terbesar yang mengancam kemampuan mengobati pneumonia, TBC, dan infeksi pasca-operasi."},
    {"q": "Mengapa pola tidur yang teratur penting untuk kesehatan metabolisme?", "a": "Secara kronobiologis, tubuh memiliki jam biologis internal (sirkadian) yang mengatur hampir semua proses metabolisme. Selama tidur, terutama pada fase slow-wave sleep, hormon pertumbuhan (GH) diproduksi secara pulsatil untuk memicu perbaikan jaringan dan metabolisme lemak. Hormon leptin (supresor nafsu makan) meningkat sementara ghrelin (stimulator nafsu makan) menurun saat tidur cukup. Kurang tidur membalik pola ini, meningkatkan ghrelin dan menurunkan leptin, menyebabkan peningkatan nafsu makan terutama untuk makanan tinggi kalori. Penelitian menunjukkan tidur kurang dari 6 jam per malam selama 2 minggu sudah menurunkan sensitivitas insulin setara dengan efek obesitas ringan. Tidur juga merupakan waktu utama tubuh membersihkan toksin otak melalui sistem glimfatik."},
    {"q": "Apa manfaat meditasi dan mindfulness bagi kesehatan mental dan fisik?", "a": "Penelitian menggunakan neuroimaging menunjukkan bahwa praktik meditasi secara konsisten menghasilkan perubahan struktural pada otak. Korteks prefrontal yang terkait regulasi emosi dan pengambilan keputusan menebal, sementara amigdala (pusat respons ketakutan dan stres) mengecil secara fungsional. Secara fisiologis, meditasi mengaktifkan sistem saraf parasimpatis ('rest and digest'), menurunkan kadar kortisol, tekanan darah, dan denyut jantung. Mekanisme ini juga mengurangi inflamasi sistemik yang diukur dengan penurunan penanda seperti CRP dan IL-6. Sebuah meta-analisis melibatkan 3.500 peserta menunjukkan meditasi mindfulness sama efektifnya dengan antidepresan untuk mencegah kekambuhan depresi. Bahkan 8 minggu program meditasi terbukti meningkatkan kepadatan materi abu-abu di hippocampus."},
    {"q": "Mengapa anak-anak perlu mendapat gizi seimbang untuk tumbuh kembang optimal?", "a": "Secara ilmiah, masa anak-anak terutama 1000 hari pertama kehidupan dari konsepsi hingga usia 2 tahun adalah jendela kritis perkembangan otak, yang tumbuh dengan kecepatan tertinggi mencapai 80% dari ukuran dewasa. Protein menyediakan asam amino untuk sintesis neurotransmiter dan struktur sel otak. Asam lemak omega-3 (khususnya DHA) merupakan komponen struktural utama membran neuron dan selubung mielin yang mempengaruhi kecepatan konduksi sinyal saraf. Zat besi kritis untuk mielinisasi dan sintesis neurotransmiter; defisiensi besi berkorelasi dengan gangguan kognitif permanen. Penelitian dari Harvard menunjukkan malnutrisi di 1000 hari pertama dapat menurunkan IQ rata-rata 10-15 poin dan berdampak permanen pada kapasitas belajar meski gizi kemudian diperbaiki."},
    {"q": "Mengapa mencegah penyakit lebih baik daripada mengobati?", "a": "Alasan mengapa pencegahan lebih unggul dari pengobatan dapat dilihat dari beberapa dimensi ilmiah dan ekonomi. Secara biologis, banyak penyakit kronis seperti diabetes tipe 2, penyakit jantung koroner, dan hipertensi berkembang selama bertahun-tahun sebelum gejala muncul, dan saat terdeteksi seringkali sudah terjadi kerusakan organ yang tidak sepenuhnya reversibel. Vaksinasi mencegah 2-3 juta kematian per tahun secara global dengan biaya jauh lebih rendah dari pengobatan penyakit itu sendiri. Penelitian pharmaco-economic menunjukkan setiap 1 dolar yang diinvestasikan dalam pencegahan dapat menghemat 5-6 dolar biaya pengobatan. Secara fisiologis, tubuh dalam kondisi sehat jauh lebih efisien menjaga homeostasis daripada memulihkannya setelah rusak. Selain itu, banyak kerusakan kronis seperti sirosis hati, nefropati diabetik, atau kerusakan saraf tidak dapat diperbaiki sepenuhnya dengan pengobatan modern mana pun."},
    {"q": "Bagaimana cara tubuh melawan infeksi bakteri secara alami?", "a": "Secara imunologis, tubuh memiliki pertahanan berlapis terhadap bakteri. Lini pertama adalah pertahanan fisik dan kimiawi: kulit yang utuh, lendir di saluran napas yang menjebak patogen, asam lambung yang membunuh sebagian besar bakteri yang tertelan, dan flora normal usus yang bersaing dengan bakteri patogen. Jika bakteri menembus lini pertama, sistem imun bawaan (innate immunity) merespons dalam menit hingga jam melalui fagosit seperti neutrofil dan makrofag yang menelan dan menghancurkan bakteri, serta sistem komplemen yang melubangi membran bakteri. Jika infeksi berlanjut, sistem imun adaptif aktif dalam 4-7 hari: sel B memproduksi antibodi spesifik yang menandai bakteri untuk dihancurkan, sementara sel T sitotoksik membunuh sel yang terinfeksi secara langsung. Demam yang dihasilkan selama infeksi juga berperan aktif, karena suhu tinggi menghambat pertumbuhan bakteri dan mempercepat respons imun."},

    # ============================================================
    # BAHASA INDONESIA & LINGUISTIK ANALITIS
    # ============================================================
    {"q": "Mengapa bahasa Indonesia menggunakan awalan 'me-' pada kata kerja aktif?", "a": "Aturan ini ada karena dalam tata bahasa Indonesia, awalan 'me-' berfungsi sebagai penanda verba aktif transitif maupun intransitif. Secara historis, awalan ini berasal dari rumpun bahasa Melayu yang membedakan peran gramatikal kata melalui afiksasi, bukan melalui urutan kata seperti dalam bahasa Inggris. Dengan adanya 'me-', pembaca langsung mengetahui bahwa subjek adalah pelaku tindakan. Misalnya, 'menulis' menunjukkan subjek sedang melakukan tindakan tulis, sementara bentuk tanpa awalan seperti 'tulis' cenderung digunakan dalam kalimat perintah atau pasif informal."},
    {"q": "Apa perbedaan antara 'di-' sebagai awalan (prefiks) dan 'di' sebagai kata depan (preposisi)?", "a": "Perbedaan ini penting secara tata bahasa. 'Di-' sebagai prefiks melekat langsung pada kata kerja tanpa spasi dan menandai bentuk pasif, misalnya 'dibaca', 'ditulis', 'dimakan'. Sementara 'di' sebagai preposisi menunjukkan tempat atau lokasi dan selalu diikuti spasi sebelum kata berikutnya, misalnya 'di rumah', 'di sekolah', 'di sini'. Aturan ini ada karena keduanya memiliki fungsi gramatikal yang berbeda: satu menunjukkan hubungan antarkata dalam frasa verbal, satu lagi menunjukkan hubungan lokatif. Kesalahan penulisan ini sangat umum karena keduanya berbunyi sama dalam percakapan lisan."},
    {"q": "Mengapa kata 'mempermasalahkan' memiliki bentuk yang panjang dengan banyak imbuhan?", "a": "Kata 'mempermasalahkan' terbentuk dari proses afiksasi berlapis: me- + per- + masalah + -kan. Ini mencerminkan produktivitas morfologis bahasa Indonesia yang tinggi. Awalan 'me-' menandai verba aktif, 'per-' berfungsi sebagai konfiks kausatif yang bermakna 'menjadikan sesuatu sebagai', dan sufiks '-kan' menambah makna ketransitifan atau benefaktif. Jadi secara analitis, 'mempermasalahkan' berarti 'menjadikan sesuatu sebagai masalah'. Aturan ini ada karena bahasa Indonesia lebih memilih menambah makna melalui imbuhan daripada menciptakan kata baru, sehingga satu kata dasar bisa menghasilkan puluhan bentuk turunan."},
    {"q": "Bagaimana cara membedakan penggunaan kata 'dan', 'serta', dan 'dengan' sebagai konjungsi?", "a": "Untuk memilih kata yang tepat, kita perlu memahami nuansa masing-masing. 'Dan' adalah konjungsi koordinatif paling netral yang menghubungkan elemen setara, misalnya 'ibu dan ayah'. 'Serta' memiliki nuansa yang lebih formal dan sering menandai tambahan yang erat kaitannya atau serentak dengan elemen sebelumnya, misalnya 'membawa pena serta buku'. 'Dengan' biasanya menunjukkan cara atau penyertaan, bukan sekadar penambahan, misalnya 'datang dengan membawa hadiah'. Kesalahan terjadi ketika 'dengan' digunakan sebagai pengganti 'dan' dalam penghubung subjek majemuk, padahal 'dengan' tidak setara secara gramatikal melainkan menunjukkan relasi."},
    {"q": "Mengapa kata 'apotek' sering salah ditulis menjadi 'apotik'?", "a": "Makna kata ini tidak berubah dalam kedua ejaan, namun 'apotek' adalah satu-satunya bentuk baku menurut KBBI. Kesalahan ini terjadi karena adanya pengaruh pelafalan. Dalam bahasa lisan, bunyi /e/ pada suku kata akhir terbuka sering diucapkan seperti /i/ oleh penutur Jawa dan beberapa daerah lain akibat pengaruh dialek. Fenomena ini disebut variasi fonologis regional. Secara etimologis, kata ini berasal dari bahasa Belanda 'apotheek' yang diserap ke dalam bahasa Indonesia dengan penyesuaian ejaan resmi. Aturan ini ada agar terdapat standarisasi penulisan yang seragam di seluruh Indonesia, terlepas dari variasi dialek."},
    {"q": "Apa yang dimaksud dengan 'eufemisme' dalam bahasa Indonesia dan mengapa digunakan?", "a": "Eufemisme adalah penggunaan ungkapan yang lebih halus atau tidak langsung untuk menggantikan kata yang dianggap kasar, tabu, atau menyakitkan. Misalnya, 'meninggal dunia' untuk menggantikan 'mati', 'tunanetra' untuk 'buta', atau 'pekerja seks komersial' untuk kata yang lebih vulgar. Aturan ini ada karena bahasa mencerminkan nilai-nilai sosial dan budaya suatu masyarakat. Dalam konteks budaya Indonesia yang mengedepankan kesopanan, eufemisme berfungsi menjaga harmoni sosial dan perasaan orang yang dibicarakan. Secara pragmatis, pilihan kata mencerminkan sikap dan empati pembicara."},
    {"q": "Mengapa kalimat pasif lebih sering digunakan dalam teks formal bahasa Indonesia daripada dalam bahasa Inggris?", "a": "Aturan ini ada karena perbedaan preferensi stilistika antara kedua bahasa. Dalam bahasa Indonesia formal dan akademik, kalimat pasif seperti 'telah dilakukan penelitian' sering digunakan untuk memberi kesan objektivitas dan menghindari penyebutan pelaku secara eksplisit. Secara pragmatis, kalimat pasif dalam bahasa Indonesia juga digunakan ketika pelaku tidak diketahui, tidak relevan, atau ketika penulis ingin menonjolkan objek atau hasil tindakan daripada siapa yang melakukan. Ini mencerminkan orientasi komunal dalam budaya Indonesia."},
    {"q": "Apakah perbedaan makna antara 'bisa', 'dapat', 'mampu', dan 'sanggup' dalam bahasa Indonesia?", "a": "Makna kata ini berubah bergantung pada konteks dan nuansa yang ingin disampaikan. 'Bisa' adalah kata paling umum dan informal yang menunjukkan kemampuan atau kemungkinan. 'Dapat' lebih formal dan sering menunjukkan kemungkinan yang bergantung pada kondisi eksternal. 'Mampu' menekankan kapasitas atau kekuatan internal seseorang, sering dipakai dalam konteks finansial atau fisik. 'Sanggup' mengandung nuansa kesanggupan yang disertai kesediaan atau janji, sering digunakan dalam konteks komitmen. Untuk memilih kata yang tepat, kita perlu mempertimbangkan register, apakah kemampuan itu internal atau eksternal, dan apakah ada dimensi komitmen di dalamnya."},
    {"q": "Mengapa pengulangan kata (reduplikasi) dalam bahasa Indonesia dapat memiliki makna yang berbeda-beda?", "a": "Reduplikasi dalam bahasa Indonesia memiliki beberapa fungsi semantis yang berbeda bergantung pada jenis dan konteksnya. Reduplikasi penuh seperti 'buku-buku' menunjukkan jamak atau banyak. Reduplikasi berubah bunyi seperti 'bolak-balik' menunjukkan tindakan berulang atau variasi. Reduplikasi sebagian seperti 'berlari-lari' menunjukkan tindakan yang tidak intensif atau santai. Ada juga reduplikasi semu seperti 'kupu-kupu' yang bukan merupakan pengulangan bermakna, melainkan memang bentuk kata aslinya. Aturan ini ada karena bahasa Indonesia warisan dari rumpun Austronesia yang secara tipologis menggunakan reduplikasi sebagai alat gramatikal produktif untuk memodifikasi makna."},
    {"q": "Bagaimana membedakan penggunaan kata 'karena' dan 'sebab' dalam kalimat bahasa Indonesia?", "a": "Secara gramatikal, keduanya adalah konjungsi kausal yang menghubungkan sebab dan akibat, namun terdapat perbedaan posisi dan register. 'Karena' lebih fleksibel dalam posisi kalimat dan lebih umum digunakan dalam ragam formal maupun informal. 'Sebab' lebih formal dan sering diletakkan di awal kalimat tunggal sebagai konjungsi antar-kalimat. Dalam konstruksi tertentu, 'sebab' cenderung merujuk pada penyebab yang lebih mendasar, sedangkan 'karena' lebih netral. Misalnya 'Ia gagal karena tidak belajar' terasa lebih natural daripada 'Ia gagal sebab tidak belajar' dalam percakapan. Memahami register membantu kita memilih konjungsi yang tepat sesuai konteks."},
    {"q": "Apa itu 'interferensi bahasa' dan bagaimana dampaknya pada bahasa Indonesia?", "a": "Interferensi bahasa adalah pengaruh sistem bahasa pertama atau bahasa ibu terhadap bahasa kedua yang sedang dipelajari atau digunakan. Dalam konteks Indonesia, penutur dengan bahasa ibu Jawa atau Sunda sering membawa pola gramatikal daerahnya ke dalam bahasa Indonesia. Contohnya, penghilangan awalan 'me-' seperti 'aku nulis' alih-alih 'aku menulis' adalah interferensi dari bahasa Jawa yang tidak mengenal awalan tersebut. Dampaknya ada yang negatif berupa pelanggaran kaidah baku, namun ada pula yang memperkaya bahasa melalui kosakata serapan dari bahasa daerah. Fenomena ini menunjukkan bahwa bahasa adalah sistem hidup yang dinamis, bukan entitas statis."},
    {"q": "Mengapa ada kata-kata bahasa Indonesia yang ditulis sama tetapi memiliki makna berbeda, seperti kata 'apel'?", "a": "Fenomena ini disebut homografi atau polisemi, yaitu kata-kata yang ditulis sama tetapi memiliki makna berbeda. Kata 'apel' bisa berarti buah (serapan dari Belanda atau Inggris) atau upacara bendera (dari bahasa Belanda 'appel'). Dalam konteks tertulis, disambiguasi dilakukan melalui konteks kalimat. Makna kata ini berubah bergantung pada konteks: 'makan apel' merujuk pada buah, sementara 'apel pagi' merujuk pada kegiatan upacara. Fenomena serupa terjadi pada 'teras' (bagian depan rumah vs. inti kayu) dan 'bisa' (mampu vs. racun). Ini mencerminkan evolusi bahasa yang menyerap kata dari berbagai sumber dengan adaptasi yang tidak selalu mengubah bentuk tulisan."},
    {"q": "Apa yang dimaksud dengan 'pleonasme' dalam bahasa Indonesia dan mengapa dianggap sebagai kesalahan?", "a": "Pleonasme adalah penggunaan kata-kata berlebihan yang maknanya sudah terkandung dalam kata sebelumnya. Contohnya: 'naik ke atas' (naik sudah berarti ke arah atas), 'turun ke bawah', 'maju ke depan', 'mundur ke belakang', 'para hadirin sekalian' (para dan sekalian sama-sama menunjukkan jamak). Pleonasme dianggap tidak efisien karena melanggar prinsip kehematan bahasa: sebuah kalimat yang baik menggunakan kata sesedikit mungkin untuk menyampaikan makna selengkap mungkin. Aturan ini ada karena efisiensi komunikasi adalah salah satu tujuan utama bahasa. Namun dalam konteks puisi atau pidato, pengulangan semacam ini kadang disengaja untuk efek penekanan emosional."},
    {"q": "Bagaimana konteks sosial memengaruhi pilihan kata dalam bahasa Indonesia, misalnya antara 'tinggal', 'menetap', 'bermukim', dan 'menghuni'?", "a": "Untuk memilih kata yang tepat, kita perlu memahami register sosial dan konotasi masing-masing kata. 'Tinggal' adalah kata paling netral dan umum. 'Menetap' mengandung nuansa permanensi dan cocok dalam konteks administratif atau formal. 'Bermukim' lebih formal dan sering digunakan dalam konteks kependudukan atau jurnalistik. 'Menghuni' memiliki nuansa kuat tentang keberadaan fisik dalam ruang dan sering digunakan dalam konteks perumahan. Makna kata ini berubah bergantung pada konteks: 'Ia tinggal di gang sempit' terasa lebih natural daripada 'Ia bermukim di gang sempit'. Pemilihan kata mencerminkan tingkat formalitas dan tujuan komunikatif penulis."},
    {"q": "Mengapa penggunaan kata 'daripada' sering keliru dalam kalimat bahasa Indonesia?", "a": "Kesalahan ini terjadi karena 'daripada' sering digunakan sebagaimana kata 'dari' atau 'tentang'. 'Daripada' secara baku berfungsi sebagai konjungsi komparatif untuk membandingkan dua hal, misalnya 'lebih baik daripada'. Namun banyak penutur menggunakannya untuk menunjukkan asal, misalnya 'salah satu daripada mereka' yang seharusnya 'salah satu dari mereka'. Dalam bahasa Melayu klasik, 'daripada' memang bisa berarti 'dari', namun dalam bahasa Indonesia baku modern, fungsinya sudah dipersempit. Aturan ini ada untuk menghindari ambiguitas: jika 'daripada' dipakai dalam dua makna berbeda, pendengar atau pembaca bisa bingung apakah yang dimaksud adalah perbandingan atau asal-usul."},
    {"q": "Mengapa struktur kalimat bahasa Indonesia bisa fleksibel dalam hal urutan S-P-O, berbeda dengan banyak bahasa lain?", "a": "Aturan ini ada karena bahasa Indonesia termasuk bahasa bertipe topik-menonjol, bukan hanya subjek-menonjol. Ini berarti elemen yang ingin ditonjolkan bisa diletakkan di awal kalimat. Misalnya, 'Buku itu saya beli kemarin' dengan urutan O-S-P adalah kalimat yang gramatikal. Fleksibilitas ini dimungkinkan karena bahasa Indonesia menggunakan sistem afiksasi untuk menandai fungsi gramatikal: awalan 'me-' untuk aktif dan 'di-' untuk pasif, sehingga urutan kata bukan satu-satunya cara menentukan siapa pelaku dan siapa objek. Dalam bahasa Inggris sebaliknya, urutan kata adalah mekanisme utama penanda fungsi gramatikal, sehingga urutannya lebih kaku."},
    {"q": "Apa yang dimaksud dengan 'kata bermakna konotatif' dan bagaimana pengaruhnya dalam retorika?", "a": "Konotasi adalah makna tambahan atau asosiasi emosional yang melekat pada kata di luar makna denotatif atau harfiahnya. Misalnya, 'kurus' dan 'ramping' secara denotatif bermakna sama, namun 'kurus' berkonotasi negatif sementara 'ramping' berkonotasi positif. Dalam retorika dan persuasi, pemilihan kata berkonotasi positif atau negatif sangat berpengaruh terhadap respons audiens. Aturan ini ada karena bahasa bukan hanya alat transmisi informasi, melainkan juga alat pembentuk persepsi dan emosi. Untuk memilih kata yang tepat dalam retorika, kita perlu memahami nilai-nilai budaya dan persepsi sosial audiens, karena konotasi bersifat kultural dan bisa berbeda antar kelompok masyarakat."},
    {"q": "Mengapa kata majemuk dalam bahasa Indonesia kadang ditulis serangkai dan kadang terpisah?", "a": "Aturan penulisan kata majemuk diatur dalam Pedoman Umum Ejaan Bahasa Indonesia (PUEBI). Kata majemuk yang sudah dianggap satu kesatuan makna penuh atau mengalami leksikalisasi ditulis serangkai, misalnya 'kacamata', 'bumiputra', 'kosakata'. Sementara kata majemuk yang masih terasa sebagai gabungan dua kata terpisah ditulis terpisah, misalnya 'rumah sakit', 'meja makan', 'kambing hitam'. Kerumitan muncul karena batas antara keduanya tidak selalu jelas, dan proses leksikalisasi bersifat gradual. Aturan ini ada untuk membantu pembaca memproses makna: kata serangkai diproses sebagai satu unit leksikal, sedangkan kata terpisah masih diproses sebagai kombinasi dua konsep. Ketika ragu, KBBI adalah rujukan utama."},
    {"q": "Apa perbedaan antara 'prosa' dan 'puisi' secara linguistik, bukan sekadar secara bentuk?", "a": "Secara linguistik, perbedaan prosa dan puisi bukan hanya soal ada-tidaknya baris atau rima. Prosa mengutamakan fungsi referensial bahasa, yaitu menyampaikan informasi atau narasi dengan mengikuti sintaksis kalimat standar. Puisi mengutamakan fungsi ekspresif dan estetis bahasa: bunyi, irama, pilihan diksi, dan makna berlapis menjadi elemen utama. Dalam puisi, penyimpangan dari tata bahasa standar yang disebut licentia poetica bisa menjadi alat artistik. Makna kata dalam puisi berubah bergantung pada konteks karena puisi secara sengaja mengeksploitasi polisemi dan konotasi. Puisi memiliki aturan estetika tersendiri yang memperbolehkan dan bahkan mendorong penyimpangan gramatikal demi efek artistik."},
    {"q": "Mengapa penting memahami konteks dalam menafsirkan makna kata ambigu seperti 'bunga'?", "a": "Kata 'bunga' adalah contoh klasik polisemi dalam bahasa Indonesia. Secara denotatif, 'bunga' bisa berarti bagian reproduktif tumbuhan, hiasan, bunga bank sebagai imbal hasil, atau digunakan secara metaforis seperti 'bunga bangsa' yang bermakna tokoh terbaik bangsa. Makna kata ini berubah bergantung pada konteks: 'bunga tabungan' jelas merujuk pada dunia keuangan, bukan botani. Untuk menafsirkan kata yang tepat, kita perlu menerapkan prinsip relevansi konteks berdasarkan teori Grice: penutur yang kooperatif akan menggunakan kata sesuai konteks diskursus. Fenomena polisemi ini menunjukkan bahwa makna adalah hasil interaksi antara kata, konteks, dan pengetahuan bersama penutur dan pendengar, bukan properti kata itu sendiri."},
    {"q": "Bagaimana gaya bahasa antitesis bekerja dalam retorika berbahasa Indonesia?", "a": "Antitesis adalah gaya bahasa yang mempertentangkan dua gagasan berlawanan dalam satu kalimat atau paragraf untuk menciptakan efek penekanan dan keindahan. Contoh klasik dalam bahasa Indonesia: 'Bersatu kita teguh, bercerai kita runtuh.' Secara linguistik, antitesis mengeksploitasi hubungan antonimi atau lawan kata dan paralelisme sintaksis: dua klausa dengan struktur serupa namun makna berlawanan. Aturan ini ada dalam retorika karena kontras menciptakan efek memori yang kuat. Otak manusia lebih mudah mengingat informasi yang disajikan dalam pola kontras. Dalam pidato persuasif, antitesis efektif karena membantu audiens memahami konsekuensi dua pilihan secara langsung dan dramatik."},
    {"q": "Mengapa bahasa Indonesia memiliki banyak kata serapan dari bahasa Belanda, Arab, dan Inggris?", "a": "Kosakata serapan dalam bahasa Indonesia mencerminkan sejarah panjang kontak budaya. Kata dari bahasa Belanda masuk selama masa kolonial, meliputi bidang administrasi dan teknologi seperti 'polisi' dari politie dan 'kantor' dari kantoor. Kata Arab masuk melalui penyebaran Islam sejak abad ke-13, terutama dalam bidang agama dan nilai moral seperti 'masjid' dan 'musyawarah'. Kata Inggris semakin banyak masuk di era globalisasi untuk konsep modern dan teknologi seperti 'komputer' dan 'manajemen'. Fenomena ini terjadi karena bahasa selalu berkembang sesuai kebutuhan penuturnya; ketika ada konsep baru, bahasa meminjam kata dari sumber terdekat dan menyesuaikannya dengan sistem fonologi dan morfologi bahasa Indonesia."},
    {"q": "Apa yang dimaksud dengan 'kalimat efektif' dalam bahasa Indonesia dan apa ciri-cirinya?", "a": "Kalimat efektif adalah kalimat yang berhasil menyampaikan informasi secara tepat, jelas, dan hemat kepada pembaca atau pendengar tanpa menimbulkan kesalahpahaman. Aturan ini ada karena tujuan utama komunikasi adalah pertukaran makna yang akurat dan efisien. Ciri-ciri kalimat efektif meliputi: kesepadanan antara struktur bahasa dan jalan pikiran, ketepatan pilihan kata, kehematan dengan tidak menggunakan kata berlebihan, kelogisan dengan hubungan antarunsur yang masuk akal, dan kevariasian untuk menghindari monoton. Untuk memilih kata yang tepat dan membangun kalimat efektif, kita perlu mempertimbangkan siapa pembaca kita, apa tujuan komunikasi, dan register bahasa yang sesuai dengan konteksnya."},
    {"q": "Mengapa penulisan huruf kapital dalam bahasa Indonesia memiliki aturan yang spesifik?", "a": "Aturan penggunaan huruf kapital diatur dalam PUEBI dan memiliki fungsi semantis serta gramatikal yang penting. Huruf kapital digunakan untuk nama diri seperti nama orang, tempat, dan lembaga untuk membedakannya dari kata umum bermakna serupa. Misalnya, 'gula jawa' (makanan) berbeda dari 'orang Jawa' (nama suku). Aturan ini ada karena huruf kapital berfungsi sebagai penanda referensi khusus, membantu pembaca mengidentifikasi apakah yang dimaksud adalah entitas spesifik atau konsep umum. Kesalahan penggunaan kapital dapat mengubah makna: 'menteri pendidikan' (jabatan umum) berbeda dari 'Menteri Pendidikan' (orang yang memegang jabatan spesifik saat ini)."},
    {"q": "Bagaimana cara mengenali dan menghindari kalimat ambigu dalam bahasa Indonesia?", "a": "Kalimat ambigu adalah kalimat yang dapat ditafsirkan dengan dua atau lebih makna yang berbeda, menghambat komunikasi yang efektif. Ambigu leksikal terjadi karena kata bermakna ganda, misalnya 'Ia melihat orang tua itu' bisa berarti orang berusia tua atau orang tuanya sendiri. Ambigu struktural terjadi karena susunan kalimat yang tidak jelas, misalnya 'Ibu memukul anak dengan tongkat besi' yang tidak jelas apakah tongkat itu digunakan ibu atau dimiliki anak. Untuk memilih kata yang tepat dan menghindari ambiguitas, kita perlu menambah penanda gramatikal yang jelas, memilih kata yang lebih spesifik, atau merestrukturisasi kalimat. Makna kata ini berubah bergantung pada konteks, sehingga konteks yang jelas adalah kunci kalimat tidak ambigu."},
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
    
    # REPETISI: Ulangi data agar model lebih hafal
    print("\n🔁 Repeating data for better memorization...")
    formatted = repeat_data(formatted, min_samples=3000)
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
