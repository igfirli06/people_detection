import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  
        database="people_detection"
    )

def get_cameras():
    conn = get_connection()
    cur = conn.cursor(dictionary=True)  
    try:
        cur.execute("SELECT * FROM cctv")
        rows = cur.fetchall()
        return rows
    finally:
        cur.close()
        conn.close()


def load_cameras_from_db():
    return get_cameras()

def add_camera(name, rtsp_url):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO cctv (name, rtsp_url, is_active) VALUES (%s, %s, TRUE)", (name, rtsp_url))
        conn.commit()
    finally:
        cur.close()
        conn.close()
