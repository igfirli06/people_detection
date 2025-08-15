import mysql.connector
from datetime import datetime
import pandas as pd

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="people_detection"
    )

def save_event(camera_id, description):
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
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT pe.*, c.name AS camera_name
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
    conn = get_connection()
    query = """
        SELECT pe.id, c.name AS camera_name, pe.exit_time, pe.description
        FROM person_exit pe
        JOIN cctv c ON pe.camera_id = c.id
        ORDER BY pe.exit_time DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df.to_excel(file_path, index=False)

def load_cameras_from_db():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM cctv")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows
