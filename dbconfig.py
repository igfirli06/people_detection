# dbconfig.py
import mysql.connector

# Konfigurasi koneksi database
db_config = {
    "host": "localhost",      # alamat server database
    "user": "root",           # username database
    "password": "",           # password database (kosong kalau default XAMPP)
    "database": "people_detection"     # nama database yang sudah kamu buat
}

# Fungsi untuk membuat koneksi
def get_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as err:
        print(f"[ERROR] Gagal konek ke database: {err}")
        return None
