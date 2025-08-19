import mysql.connector
from datetime import datetime
import pandas as pd
import json   # WAJIB buat load zona


def get_connection():
    """Buat koneksi ke database MySQL"""
    return mysql.connector.connect(
        host="localhost",
        user="root",          # sesuaikan
        password="",          # sesuaikan
        database="people_detection"
    )

def load_cameras_from_db():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM cctv")
    rows = cursor.fetchall()
    conn.close()

    cameras = []
    for row in rows:
        zone = []
        if row["zone"]:
            try:
                zone = json.loads(row["zone"])
            except Exception as e:
                print(f"[WARNING] Zone tidak valid untuk kamera {row['id']}: {row['zone']} -> {e}")
                zone = []
        cameras.append({
            "id": row["id"],
            "name": row["name"],
            "rtsp_url": row["rtsp_url"],
            "zone": zone
        })
    return cameras

def save_event(camera_id, description):
    """Simpan event orang keluar ke tabel person_exit"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        now = datetime.now()
        cursor.execute("""
            INSERT INTO person_exit (camera_id, exit_time, description)
            VALUES (%s, %s, %s)
        """, (camera_id, now, description))
        conn.commit()
    finally:
        cursor.close()
        conn.close()


def get_recent_events(limit=15):
    """Ambil event terakhir"""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT pe.*, c.name AS name
            FROM person_exit pe
            JOIN cctv c ON pe.camera_id = c.id
            ORDER BY pe.exit_time DESC
            LIMIT %s
        """, (limit,))
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()


def export_events_to_excel(file_path):
    """Export semua event ke file Excel"""
    conn = get_connection()
    query = """
        SELECT pe.id, c.name AS name, pe.exit_time, pe.description
        FROM person_exit pe
        JOIN cctv c ON pe.camera_id = c.id
        ORDER BY pe.exit_time DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df.to_excel(file_path, index=False)
