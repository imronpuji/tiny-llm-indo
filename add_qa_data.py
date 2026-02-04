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
    
    # REPETISI: Ulangi data agar model lebih hafal
    print("\n🔁 Repeating data for better memorization...")
    formatted = repeat_data(formatted, min_samples=10000)
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
