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

# ===================== CREATE TABLES =====================
def create_tables():
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cctv (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(80) NOT NULL,
                rtsp_url VARCHAR(255) NOT NULL UNIQUE,
                is_active BOOLEAN DEFAULT TRUE,
                min_session_duration INT DEFAULT 10,
                record_schedule_enabled BOOLEAN DEFAULT FALSE,
                record_start_time TIME,
                record_end_time TIME
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS zones (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(80) NOT NULL,
                camera_id INT,
                coordinates TEXT NOT NULL,
                max_people INT DEFAULT 4,
                FOREIGN KEY (camera_id) REFERENCES cctv(id) ON DELETE CASCADE
            )
        """)
        cur.execute("""
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
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS empty_zone_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                camera_id INT NOT NULL,
                zone_id INT,
                start_time DATETIME NOT NULL,
                end_time DATETIME NOT NULL,
                duration INT,
                FOREIGN KEY (camera_id) REFERENCES cctv(id) ON DELETE CASCADE,
                FOREIGN KEY (zone_id) REFERENCES zones(id) ON DELETE SET NULL
            )
        """)
        conn.commit()
        print("Tables checked/created successfully.")
    except mysql.connector.Error as err:
        print(f"Failed creating tables: {err}")
    finally:
        cur.close()
        conn.close()

# ===================== CAMERA & ZONES =====================
def load_cameras_from_db():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("SELECT id, name, rtsp_url, is_active, min_session_duration FROM cctv")
        return cur.fetchall()
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
            try:
                zone['coordinates'] = json.loads(zone['coordinates'])
            except Exception:
                zone['coordinates'] = []
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
            try:
                zone['coordinates'] = json.loads(zone['coordinates'])
            except Exception:
                zone['coordinates'] = []
        return zones
    finally:
        cur.close()
        conn.close()

def add_camera(name: str, ip_address: str):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO cctv (name, rtsp_url) VALUES (%s, %s)", (name, ip_address))
        conn.commit()
    finally:
        cur.close()
        conn.close()

def add_zone(camera_id: int, name: str, coordinates: str, max_people: int):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO zones (camera_id, name, coordinates, max_people) VALUES (%s, %s, %s, %s)",
            (camera_id, name, coordinates, max_people)
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

def update_camera_duration(camera_id: int, duration: int):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE cctv SET min_session_duration = %s WHERE id = %s", (duration, camera_id))
        conn.commit()
    finally:
        cur.close()
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

# ===================== PERSON SESSIONS =====================
def save_person_session_start(camera_id: int, zone_id: int, tracking_id: int):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO person_sessions (camera_id, zone_id, tracking_id, start_time) 
            VALUES (%s, %s, %s, NOW())
        """, (camera_id, zone_id, tracking_id))
        conn.commit()
        return cur.lastrowid
    except Exception as e:
        print(f"Failed to save session start: {e}")
        return None
    finally:
        cur.close()
        conn.close()

def update_person_session_end(session_id: int, duration: int):
    conn = get_connection()
    cur = conn.cursor()
    try:
        now = datetime.now()
        cur.execute("""
            UPDATE person_sessions
            SET end_time = %s, duration = %s
            WHERE id = %s
        """, (now, duration, session_id))
        conn.commit()
    except Exception as e:
        print(f"Failed to update session end: {e}")
    finally:
        cur.close()
        conn.close()

def delete_person_session_by_id(session_id: int):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM person_sessions WHERE id = %s AND end_time IS NULL", (session_id,))
        conn.commit()
    except Exception as e:
        print(f"Failed to delete short session: {e}")
    finally:
        cur.close()
        conn.close()

def get_person_sessions(limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT ps.id, ps.camera_id, ps.tracking_id, ps.start_time, ps.end_time, ps.duration,
                   c.name AS camera_name, z.name AS zone_name
            FROM person_sessions ps
            JOIN cctv c ON ps.camera_id = c.id
            LEFT JOIN zones z ON ps.zone_id = z.id
            ORDER BY ps.start_time DESC
            LIMIT %s
        """, (limit,))
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()

# ===================== EMPTY ZONE SESSIONS =====================
def save_empty_zone_session(camera_id, zone_id, start_time, end_time):
    conn = get_connection()
    cur = conn.cursor()
    try:
        duration_seconds = int((end_time - start_time).total_seconds())
        cur.execute("""
            INSERT INTO empty_zone_sessions (camera_id, zone_id, start_time, end_time, duration)
            VALUES (%s, %s, %s, %s, %s)
        """, (camera_id, zone_id, start_time, end_time, duration_seconds))
        conn.commit()
    except Exception as e:
        print(f"Error saving empty zone session: {e}")
        if conn: conn.rollback()
    finally:
        cur.close()
        conn.close()

def get_empty_zone_sessions(limit=50):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT ez.id, ez.camera_id, c.name AS camera_name, ez.zone_id, z.name AS zone_name,
                   ez.start_time, ez.end_time, ez.duration
            FROM empty_zone_sessions ez
            JOIN cctv c ON ez.camera_id = c.id
            LEFT JOIN zones z ON ez.zone_id = z.id
            ORDER BY ez.start_time DESC
            LIMIT %s
        """, (limit,))
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()

# ===================== UTILITIES =====================
def get_min_duration_for_camera(camera_id: int) -> int:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT min_session_duration FROM cctv WHERE id = %s", (camera_id,))
        result = cur.fetchone()
        return result[0] if result else 0
    finally:
        cur.close()
        conn.close()

# =====================================================
# EKSPOR EXCEL
# =====================================================
def export_person_sessions_to_excel(file_path: str):
    conn = get_connection()
    query = """
        SELECT c.name AS `Nama Kamera`,
               z.name AS `Nama Zona`,
               ps.tracking_id AS `Tracking ID`,
               ps.start_time AS `Waktu Masuk`,
               ps.end_time AS `Waktu Keluar`,
               ps.duration AS `Durasi (detik)`
        FROM person_sessions ps
        JOIN cctv c ON ps.camera_id = c.id
        LEFT JOIN zones z ON ps.zone_id = z.id
        ORDER BY ps.start_time DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df.to_excel(file_path, index=False)

def export_sessions_to_excel(file_path):
    conn = get_connection()
    query = """
        SELECT ps.id, c.name AS name, ps.start_time, ps.end_time, ps.duration
        FROM person_sessions ps
        JOIN cctv c ON ps.camera_id = c.id
        WHERE ps.end_time IS NOT NULL
        ORDER BY ps.end_time DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df['duration_formatted'] = df['duration'].apply(
        lambda x: f"{int(x // 3600):02d}:{int((x % 3600) // 60):02d}:{int(x % 60):02d}"
    )
    df.to_excel(file_path, index=False)