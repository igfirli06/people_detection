import mysql.connector
from datetime import datetime
import pandas as pd
import json
from typing import List, Dict, Any

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="people_detection"
    )

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cctv (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(80) NOT NULL,
                rtsp_url VARCHAR(255) NOT NULL UNIQUE,
                is_active BOOLEAN DEFAULT TRUE,
                min_session_duration INT DEFAULT 10,
                record_schedule_enabled BOOLEAN DEFAULT FALSE,
                record_start_time TIME,
                record_end_time TIME
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS zones (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(80) NOT NULL,
                camera_id INT,
                coordinates VARCHAR(500) NOT NULL,
                max_people INT DEFAULT 4,
                FOREIGN KEY (camera_id) REFERENCES cctv(id) ON DELETE CASCADE
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS person_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                camera_id INT,
                zone_id INT,
                tracking_id INT,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                duration INT,
                FOREIGN KEY (camera_id) REFERENCES cctv(id) ON DELETE CASCADE,
                FOREIGN KEY (zone_id) REFERENCES zones(id) ON DELETE SET NULL
            );
        """)
        conn.commit()
        print("Tables checked/created successfully.")
    except mysql.connector.Error as err:
        print(f"Failed creating tables: {err}")
    finally:
        cursor.close()
        conn.close()

def save_empty_zone_session(camera_id, zone_id, start_time, end_time):
    """
    Menyimpan sesi zona kosong ke dalam database.
    
    Args:
        camera_id (int): ID kamera terkait.
        zone_id (int): ID zona yang kosong.
        start_time (datetime): Waktu mulai zona kosong.
        end_time (datetime): Waktu berakhir zona kosong.
    """
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Hitung durasi dalam detik
        duration_seconds = (end_time - start_time).total_seconds()
        
        sql = """
            INSERT INTO empty_zone_sessions 
            (camera_id, zone_id, start_time, end_time, duration) 
            VALUES (%s, %s, %s, %s, %s)
        """
        cur.execute(sql, (camera_id, zone_id, start_time, end_time, duration_seconds))
        conn.commit()
        print("Sesi zona kosong berhasil disimpan.")
        
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def get_empty_zone_sessions(limit=50):
    """
    Mengambil data sesi zona kosong dari database.
    
    Args:
        limit (int): Jumlah data sesi yang akan diambil.
        
    Returns:
        list: Daftar kamus yang berisi data sesi zona kosong.
    """
    conn = None
    cur = None
    sessions = []
    try:
        conn = get_connection()
        cur = conn.cursor(dictionary=True) # Mengembalikan hasil sebagai dictionary
        
        sql = """
            SELECT
                ez.id, ez.camera_id, c.name AS camera_name,
                ez.zone_id, z.zone_name AS zone_name,
                ez.start_time, ez.end_time, ez.duration
            FROM empty_zone_sessions ez
            JOIN cctv c ON ez.camera_id = c.id
            LEFT JOIN zones z ON ez.zone_id = z.id
            ORDER BY ez.start_time DESC
            LIMIT %s
        """
        cur.execute(sql, (limit,))
        sessions = cur.fetchall()
        
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            
    return sessions

def load_cameras_from_db():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT id, name, rtsp_url, is_active, min_session_duration FROM cctv")
        cams = cur.fetchall()
        return cams
    finally:
        cur.close()
        conn.close()

def load_zones_from_db() -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT id, name, camera_id, coordinates, max_people FROM zones")
        zones = cur.fetchall()
        for zone in zones:
            zone['coordinates'] = json.loads(zone['coordinates'])
        return zones
    finally:
        cur.close()
        conn.close()

def load_zones_for_camera(camera_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT id, name, camera_id, coordinates, max_people FROM zones WHERE camera_id = %s", (camera_id,))
        zones = cur.fetchall()
        for zone in zones:
            zone['coordinates'] = json.loads(zone['coordinates'])
        return zones
    finally:
        cur.close()
        conn.close()

def add_camera(name: str, ip_address: str):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO cctv (name, rtsp_url) VALUES (%s, %s)",
            (name, ip_address)
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def add_zone(camera_id: int, name: str, coordinates: str, max_people: int):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO zones (camera_id, name, coordinates, max_people) VALUES (%s, %s, %s, %s)",
            (camera_id, name, coordinates, max_people)
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()
        
def update_camera_duration(camera_id: int, duration: int):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE cctv SET min_session_duration = %s WHERE id = %s",
            (duration, camera_id)
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def delete_camera_by_id(camera_id: int):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM cctv WHERE id = %s", (camera_id,))
        conn.commit()
    finally:
        cur.close()
        conn.close()

def delete_zone_by_id(zone_id: int):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM zones WHERE id = %s", (zone_id,))
        conn.commit()
    finally:
        cur.close()
        conn.close()

def save_person_session_start(camera_id: int, zone_id: int, tracking_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO person_sessions (camera_id, zone_id, tracking_id, start_time) VALUES (%s, %s, %s, NOW())",
            (camera_id, zone_id, tracking_id)
        )
        conn.commit()
        return cursor.lastrowid
    except Exception as e:
        print(f"Failed to save session start: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

def update_person_session_end(session_id: int, duration: int):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        now = datetime.now()
        cursor.execute("""
            UPDATE person_sessions
            SET end_time = %s, duration = %s
            WHERE id = %s
        """, (now, duration, session_id))
        conn.commit()
    except Exception as e:
        print(f"Failed to update session end: {e}")
    finally:
        cursor.close()
        conn.close()

def delete_person_session_by_id(session_id: int):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM person_sessions WHERE id = %s AND end_time IS NULL",
            (session_id,)
        )
        conn.commit()
    except Exception as e:
        print(f"Failed to delete short session: {e}")
    finally:
        cur.close()
        conn.close()

def get_person_sessions(limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
        SELECT
            ps.id,
            ps.camera_id,
            ps.tracking_id,
            ps.start_time,
            ps.end_time,
            ps.duration,
            c.name AS camera_name,
            z.name AS zone_name
        FROM person_sessions ps
        JOIN cctv c ON ps.camera_id = c.id
        LEFT JOIN zones z ON ps.zone_id = z.id
        ORDER BY ps.start_time DESC
        LIMIT %s
        """
        cursor.execute(query, (limit,))
        return cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

def export_sessions_to_excel(file_path):
    conn = get_connection()
    query = """
        SELECT c.name AS `Kamera`, z.name AS `Zona`, ps.start_time AS `Waktu Masuk`, ps.end_time AS `Waktu Keluar`, ps.duration AS `Durasi (detik)`
        FROM person_sessions ps
        JOIN cctv c ON ps.camera_id = c.id
        LEFT JOIN zones z ON ps.zone_id = z.id
        WHERE ps.end_time IS NOT NULL
        ORDER BY ps.end_time DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df.to_excel(file_path, index=False)
