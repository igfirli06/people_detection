tech stack: 
1. core logic and AI (obejct detection)
Python: Bahasa pemrograman utama.
Ultralytics YOLOv8: Framework AI yang digunakan untuk mendeteksi dan melacak (tracking) objek (orang) secara real-time. Di kode ini, kamu menggunakan yolov8n.pt (model nano yang cepat).
OpenCV (cv2): Digunakan untuk pemrosesan gambar, seperti membaca stream RTSP dari kamera, menggambar bounding box, menangani poligon zona, dan mengolah video (VideoWriter).
NumPy: Digunakan untuk operasi matriks dan perhitungan koordinat poligon zona.

2. backend & web server
Flask: Framework web mikro yang digunakan untuk membuat API, menyediakan endpoint stream video, dan merender antarmuka pengguna.
Threading: Proyek ini menggunakan threading secara intensif untuk memisahkan tugas:
Satu thread untuk mengambil frame dari kamera (capture thread).
Satu thread untuk menjalankan deteksi AI (detect thread).
Hal ini dilakukan agar aplikasi tetap responsif dan tidak lag.

3. Database & Data Management
MySQL (melalui modul database.py): Digunakan untuk menyimpan konfigurasi kamera, koordinat zona, dan log sesi (kapan orang masuk/keluar zona).
JSON: Digunakan untuk menyimpan dan memproses koordinat zona yang kompleks.
Pandas: Digunakan (dalam impor) yang biasanya ditujukan untuk manipulasi data atau pembuatan laporan statistik.

4. Communication & Protokol
RTSP (Real-Time Streaming Protocol): Protokol yang digunakan untuk mengambil source video dari CCTV/IP Camera.
Requests: Digunakan untuk mengirim notifikasi ke sistem eksternal (dalam kode kamu: http://p2.kti.co.id/checkNotif).
Multipart HTTP Stream: Teknik yang digunakan Flask untuk mengirimkan frame-by-frame gambar ke browser sehingga terlihat seperti video (mjpeg).

Arsitektur Aliran Data
Input: Kamera mengirim stream video via RTSP.
Processing: Python menggunakan YOLOv8 untuk mendeteksi tracking_id orang di dalam koordinat poligon yang ditentukan.
Storage: Setiap kejadian (orang masuk/zona kosong) dicatat ke MySQL.
Output: Dashboard Flask menampilkan video yang sudah diberi anotasi dan tabel event secara real-time.
