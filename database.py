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
                if isinstance(row["zone"], str):
                    zone = json.loads(row["zone"])
                else:
                    zone = row["zone"]
            except (json.JSONDecodeError, TypeError) as e:
                print(f"[WARNING] Zone tidak valid untuk kamera {row['id']}: {row['zone']} -> {e}")
                zone = []
        cameras.append({
            "id": row["id"],
            "name": row["name"],
            "rtsp_url": row["rtsp_url"],
            "zone": zone,
            "min_session_duration": row.get("min_session_duration", 10),
            "is_active": row.get("is_active", 1)
        })
    return cameras

def save_person_session_start(camera_id, tracking_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO person_sessions (camera_id, tracking_id, start_time) VALUES (%s, %s, NOW())",
            (camera_id, tracking_id)
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
        lambda x: f"{int(x // 3600):02d}:{int((x % 3600) // 60):02d}:{int(x % 60):02d}"
    )
    df.to_excel(file_path, index=False)