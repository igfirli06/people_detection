from flask import app
import mysql.connector
from datetime import datetime
import pandas as pd
import json
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

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
                   c.name AS camera_name, z.zone_name AS zone_name
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
        print(f"[DB] Gagal simpan empty_zone_sessions: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

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

def get_empty_zone_sessions(limit=1000):
    """Mengambil sesi zona tidak aktif dari database - VERSION FIXED"""
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        # Cek struktur tabel zones untuk menentukan nama kolom
        cur.execute("SHOW COLUMNS FROM zones LIKE 'name'")
        has_name_column = cur.fetchone()
        
        cur.execute("SHOW COLUMNS FROM zones LIKE 'zone_name'") 
        has_zone_name_column = cur.fetchone()
        
        # Tentukan nama kolom zone berdasarkan struktur sebenarnya
        if has_name_column:
            zone_name_field = "z.name"
        elif has_zone_name_column:
            zone_name_field = "z.zone_name"
        else:
            zone_name_field = "NULL"  # Fallback
            
        print(f"DEBUG: Using zone name field: {zone_name_field}")
        
        query = f"""
            SELECT 
                ez.id,
                ez.camera_id,
                c.name AS camera_name,
                ez.zone_id,
                {zone_name_field} AS zone_name,
                ez.start_time,
                ez.end_time,
                ez.duration
            FROM empty_zone_sessions ez
            JOIN cctv c ON ez.camera_id = c.id
            LEFT JOIN zones z ON ez.zone_id = z.id
            WHERE ez.duration > 0 OR ez.end_time IS NOT NULL
            ORDER BY ez.start_time DESC
            LIMIT %s
        """
        
        print(f"DEBUG: Executing query: {query}")
        cur.execute(query, (limit,))
        sessions = cur.fetchall()
        
        print(f"=== DEBUG EMPTY ZONE SESSIONS ===")
        print(f"Query successful. Found {len(sessions)} records")
        
        if sessions:
            for i, session in enumerate(sessions):
                print(f"Record {i+1}: ID={session['id']}, Camera={session.get('camera_name', 'N/A')}, "
                      f"Zone={session.get('zone_name', 'N/A')}, Duration={session.get('duration', 0)}s")
        else:
            print("No records found in empty_zone_sessions table")
            # Cek apakah tabel kosong
            cur.execute("SELECT COUNT(*) as total FROM empty_zone_sessions")
            total_count = cur.fetchone()['total']
            print(f"Total records in empty_zone_sessions table: {total_count}")
            
        print("=================================")
        
        return sessions
        
    except Exception as e:
        print(f"ERROR in get_empty_zone_sessions: {e}")
        return []
    finally:
        cur.close()
        conn.close()

def generate_real_empty_zone_data():
    """Generate data real untuk empty_zone_sessions berdasarkan data yang ada - FIXED"""
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    
    try:
        # Query yang lebih sederhana dan robust
        query = """
            SELECT 
                camera_id,
                zone_id,
                MAX(end_time) as last_end_time
            FROM person_sessions 
            WHERE end_time IS NOT NULL
            GROUP BY camera_id, zone_id
            HAVING last_end_time < NOW() - INTERVAL 1 MINUTE
            LIMIT 5
        """
        
        cur.execute(query)
        recent_sessions = cur.fetchall()
        
        inserted_count = 0
        for session in recent_sessions:
            if session['last_end_time']:
                # Buat empty session dari last_end_time sampai sekarang
                start_time = session['last_end_time']
                end_time = datetime.now()
                duration = int((end_time - start_time).total_seconds())
                
                if duration > 60:  # Minimal 1 menit
                    cur.execute(
                        "INSERT INTO empty_zone_sessions (camera_id, zone_id, start_time, end_time, duration) VALUES (%s, %s, %s, %s, %s)",
                        (session['camera_id'], session['zone_id'], start_time, end_time, duration)
                    )
                    inserted_count += 1
                    print(f"Generated empty session: Camera {session['camera_id']}, Zone {session['zone_id']}, Duration {duration}s")
        
        conn.commit()
        print(f"Generated {inserted_count} real empty zone sessions")
        return inserted_count
        
    except Exception as e:
        conn.rollback()
        print(f"Error generating empty zone data: {e}")
        return 0
    finally:
        cur.close()
        conn.close()

def get_person_sessions(limit: int = 1000) -> List[Dict[str, Any]]:
    """Mengambil sesi zona aktif dari database - FIXED untuk handle sesi aktif"""
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        # Cek struktur tabel zones untuk menentukan nama kolom
        cur.execute("SHOW COLUMNS FROM zones LIKE 'name'")
        has_name_column = cur.fetchone()
        
        cur.execute("SHOW COLUMNS FROM zones LIKE 'zone_name'") 
        has_zone_name_column = cur.fetchone()
        
        # Tentukan nama kolom zone berdasarkan struktur sebenarnya
        if has_name_column:
            zone_name_field = "z.name"
        elif has_zone_name_column:
            zone_name_field = "z.zone_name"
        else:
            zone_name_field = "NULL"
            
        query = f"""
            SELECT 
                ps.id,
                ps.camera_id,
                c.name AS camera_name,
                ps.zone_id,
                {zone_name_field} AS zone_name,
                ps.start_time,
                ps.end_time,
                ps.duration,
                CASE 
                    WHEN ps.end_time IS NULL THEN 'Sesi Aktif'
                    ELSE 'Selesai'
                END AS status
            FROM person_sessions ps
            JOIN cctv c ON ps.camera_id = c.id
            LEFT JOIN zones z ON ps.zone_id = z.id
            WHERE ps.duration > 0 OR ps.end_time IS NULL
            ORDER BY ps.start_time DESC
            LIMIT %s
        """
        
        cur.execute(query, (limit,))
        sessions = cur.fetchall()
        
        print(f"=== DEBUG PERSON SESSIONS ===")
        print(f"Query successful. Found {len(sessions)} records")
        
        # Hitung durasi untuk sesi yang masih aktif
        now = datetime.now()
        for session in sessions:
            if session['end_time'] is None and session['duration'] is None:
                # Hitung durasi untuk sesi aktif
                start_time = session['start_time']
                if isinstance(start_time, str):
                    start_time = datetime.fromisoformat(str(start_time))
                duration_seconds = int((now - start_time).total_seconds())
                session['duration'] = duration_seconds
                session['end_time'] = None  # Tetap None untuk menandakan sesi aktif
            elif session['end_time'] is None and session['duration'] is not None:
                # Durasi sudah dihitung, tetap gunakan
                pass
                
        if sessions:
            for i, session in enumerate(sessions):
                print(f"Record {i+1}: ID={session['id']}, Status={session['status']}, "
                      f"Camera={session.get('camera_name', 'N/A')}, Duration={session.get('duration', 0)}s")
        print("=================================")
        
        return sessions
    except Exception as e:
        print(f"ERROR getting person sessions: {e}")
        return []
    finally:
        cur.close()
        conn.close()