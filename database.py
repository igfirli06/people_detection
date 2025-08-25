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

def save_person_session_start(camera_id, tracking_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        now = datetime.now()
        cursor.execute("""
            INSERT INTO person_sessions (camera_id, tracking_id, start_time)
            VALUES (%s, %s, %s)
        """, (camera_id, tracking_id, now))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def update_person_session_end(camera_id, tracking_id, duration):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        now = datetime.now()
        cursor.execute("""
            UPDATE person_sessions
            SET end_time = %s, duration = %s
            WHERE camera_id = %s AND tracking_id = %s AND end_time IS NULL
        """, (now, duration, camera_id, tracking_id))
        conn.commit()
    finally:
        cursor.close()
        conn.close()

# Gabungkan kedua fungsi get_person_sessions menjadi satu fungsi yang lebih baik
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
            c.name AS camera_name
        FROM person_sessions ps
        JOIN cctv c ON ps.camera_id = c.id
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
        SELECT ps.id, c.name AS name, ps.start_time, ps.end_time, ps.duration
        FROM person_sessions ps
        JOIN cctv c ON ps.camera_id = c.id
        WHERE ps.end_time IS NOT NULL
        ORDER BY ps.end_time DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df['duration_formatted'] = df['duration'].apply(
        lambda x: f"{x // 3600:02d}:{(x % 3600) // 60:02d}:{x % 60:02d}"
    )
    df.to_excel(file_path, index=False)
    
def delete_person_session_by_id(camera_id, tracking_id):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM person_sessions WHERE camera_id = %s AND tracking_id = %s AND end_time IS NULL",
            (camera_id, tracking_id)
        )
        conn.commit()
    except Exception as e:
        print(f"Failed to delete short session: {e}")
    finally:
        cur.close()
        conn.close()