
import os
import time
import threading
from typing import Tuple, List, Dict, Any
import numpy as np
from datetime import datetime, timedelta
import cv2
import json
import ast
import collections
import requests

from flask import (
    Flask,
    Response,
    render_template,
    redirect,
    url_for,
    jsonify,
    request,
    flash,
    send_file,
)
from ultralytics import YOLO
from database import (
    load_cameras_from_db,
    get_connection,
    update_person_session_end,
    export_person_sessions_to_excel,
    export_empty_zone_sessions_to_excel,
)


MODEL_PATH = os.environ.get("YOLO_MODEL", "yolov8n.pt")
RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)
DETECTION_SIZE = (960, 540)
TARGET_RECORD_FPS = float(os.environ.get("TARGET_RECORD_FPS", "25.0"))
SLOW_FACTOR = float(os.environ.get("SLOW_FACTOR", "4.00"))
RTSP_OPEN_RETRY_SECONDS = float(os.environ.get("RTSP_RETRY", "2.0"))
EXPORTS_DIR = os.environ.get("EXPORTS_DIR", "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)
MIN_SESSION_DURATION = 10
DEBUG_DRAW_ALL = True
pending_notifications = collections.deque(maxlen=20) 
last_state = None          # "aktif" atau "tidak_aktif"
start_time = None
zona_aktif = timedelta(0)
zona_tidak_aktif = timedelta(0)
pending_end_sessions = {}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "supersecret")

capture_threads: Dict[int, threading.Thread] = {}
detect_threads: Dict[int, threading.Thread] = {}
stop_flags: Dict[int, bool] = {}
thread_locks: Dict[int, threading.Lock] = {}
latest_frame: Dict[int, np.ndarray] = {}
annotated_frame: Dict[int, np.ndarray] = {}
people_count: Dict[int, int] = {}
frames_lock = threading.Lock()
recording_locks: Dict[int, threading.Lock] = {}
writers: Dict[int, cv2.VideoWriter] = {}
writer_info: Dict[int, Dict[str, Any]] = {}
recording_status: Dict[int, bool] = {}
last_write_time: Dict[int, float] = {}
CAM_ZONES: Dict[int, List[Dict[str, Any]]] = {}
active_sessions: Dict[int, Dict[int, Dict[str, Any]]] = {}
MIN_SESSION_DURATIONS: Dict[int, int] = {}

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    app.logger.error(f"Gagal load YOLO model di {MODEL_PATH}: {e}")
    model = None

def load_all_zones_from_db() -> Dict[int, List[Dict[str, Any]]]:
    """Memuat semua zona dari database dan mengelompokkannya berdasarkan camera_id."""
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute("""
            SELECT id, camera_id, zone_name AS name, coordinates, 
                   max_people, max_inactive_duration
            FROM zones
            ORDER BY name ASC
        """)
        zones_data = cur.fetchall()

        all_zones: Dict[int, List[Dict[str, Any]]] = {}

        for zone in zones_data:
            cam_id = zone['camera_id']
            if cam_id not in all_zones:
                all_zones[cam_id] = []

            try:
                coords = json.loads(zone['coordinates'])
                all_zones[cam_id].append({
                    'id': zone['id'],
                    'name': zone['name'],
                    'points': coords,
                    'max_people': zone['max_people'],
                    'max_inactive_duration': zone['max_inactive_duration'],
                    'notified_full': False,
                    'last_count': 0
                })
            except (json.JSONDecodeError, TypeError):
                app.logger.error(
                    f"Gagal parse koordinat untuk zona {zone['id']} di kamera {cam_id}"
                )

        return all_zones

    finally:
        cur.close()
        conn.close()


def update_zone_status(people_count: int):
    global last_state, start_time, zona_aktif, zona_tidak_aktif

    now = datetime.now()

    if people_count > 0:  # ada orang
        if last_state != "aktif":
            # reset timer saat status berubah
            start_time = now
            zona_aktif = timedelta(0)
            last_state = "aktif"
        else:
            zona_aktif = now - start_time
        zona_tidak_aktif = timedelta(0)  # pastikan reset

    else:  # tidak ada orang
        if last_state != "tidak_aktif":
            start_time = now
            zona_tidak_aktif = timedelta(0)
            last_state = "tidak_aktif"
        else:
            zona_tidak_aktif = now - start_time
        zona_aktif = timedelta(0)

    # mulai dari 1 detik kalau sudah ada hitungan
    aktif_seconds = int(zona_aktif.total_seconds())
    tidak_aktif_seconds = int(zona_tidak_aktif.total_seconds())

    if aktif_seconds > 0:
        aktif_seconds = max(1, aktif_seconds)
    if tidak_aktif_seconds > 0:
        tidak_aktif_seconds = max(1, tidak_aktif_seconds)

    return str(timedelta(seconds=aktif_seconds)), str(timedelta(seconds=tidak_aktif_seconds))

def send_notification_get(cam_id: int, message: str):
    url_notif = "http://p2.kti.co.id/magang/checkNotif"
    payload = {
        "camera_id": cam_id,
        "message": message,
    }
    
    try:
        response = requests.get(url_notif, params=payload)
        response.raise_for_status() 
        print(f"Notifikasi berhasil dikirim: {response.text}")
        
    except requests.exceptions.RequestException as e:
        print(f"Gagal mengirim notifikasi: {e}")

def save_person_session_start_with_zone(camera_id: int, tracking_id: int, zone_id: int, start_time: datetime) -> int | None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO person_sessions (camera_id, zone_id, tracking_id, start_time) VALUES (%s, %s, %s, %s)",
            (camera_id, zone_id, tracking_id, start_time)
        )
        conn.commit()
        last_id = cur.lastrowid
        return last_id
    except Exception as e:
        conn.rollback()
        app.logger.error(f"Gagal menyimpan sesi awal ke DB: {e}")
        return None
    finally:
        cur.close()
        conn.close()

def get_person_sessions_with_zones(limit: int = 50) -> List[Dict[str, Any]]:
    """Mengambil data sesi orang dengan nama kamera dan nama zona."""
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT
                ps.id, ps.camera_id, c.name AS camera_name,
                ps.zone_id, z.zone_name AS zone_name,
                ps.start_time, ps.end_time, ps.duration
            FROM person_sessions ps
            JOIN cctv c ON ps.camera_id = c.id
            LEFT JOIN zones z ON ps.zone_id = z.id
            ORDER BY ps.start_time DESC LIMIT %s
        """
        cur.execute(query, (limit,))
        sessions = cur.fetchall()
        return sessions
    except Exception as e:
        app.logger.error(f"Gagal mengambil sesi orang dari DB: {e}")
        return []
    finally:
        cur.close()
        conn.close()

def is_inside_zone(bbox, zone_points, min_overlap_ratio=0.2):
    bx1, by1, bx2, by2 = bbox
    bbox_area = (bx2 - bx1) * (by2 - by1)
    if bbox_area <= 0: return False
    if bx1 > DETECTION_SIZE[0] or by1 > DETECTION_SIZE[1] or bx2 < 0 or by2 < 0: return False
    pts = np.array(zone_points, np.intp)
    poly_mask = np.zeros(DETECTION_SIZE[::-1], dtype=np.uint8)
    cv2.fillPoly(poly_mask, [pts], 255)
    bbox_mask = np.zeros_like(poly_mask)
    x_min, y_min = max(0, int(bx1)), max(0, int(by1))
    x_max, y_max = min(DETECTION_SIZE[0], int(bx2)), min(DETECTION_SIZE[1], int(by2))
    cv2.rectangle(bbox_mask, (x_min, y_min), (x_max, y_max), 255, -1)
    intersection_mask = cv2.bitwise_and(poly_mask, bbox_mask)
    intersection_area = np.sum(intersection_mask > 0)
    overlap_ratio = intersection_area / max(1, bbox_area)
    return overlap_ratio >= min_overlap_ratio

def ensure_schema() -> None:
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # === Buat tabel cctv ===
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cctv (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                rtsp_url TEXT,
                zone TEXT NULL,
                is_active TINYINT(1) DEFAULT 1
            )
        """)

        # === Buat tabel person_sessions ===
        cur.execute("""
            CREATE TABLE IF NOT EXISTS person_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                camera_id INT NOT NULL,
                tracking_id INT NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME NULL,
                duration INT NULL,
                FOREIGN KEY (camera_id) REFERENCES cctv(id) ON DELETE CASCADE
            )
        """)

        # === Buat tabel people_detection ===
        cur.execute("""
            CREATE TABLE IF NOT EXISTS people_detection (
                id INT AUTO_INCREMENT PRIMARY KEY,
                camera_id INT NOT NULL,
                count INT NOT NULL,
                timestamp DATETIME NOT NULL,
                INDEX (camera_id),
                FOREIGN KEY (camera_id) REFERENCES cctv(id) ON DELETE CASCADE
            )
        """)

        # === Tambahan kolom opsional di cctv ===
        cur.execute("SHOW COLUMNS FROM cctv LIKE 'record_schedule_enabled'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE cctv ADD COLUMN record_schedule_enabled TINYINT(1) DEFAULT 0")

        cur.execute("SHOW COLUMNS FROM cctv LIKE 'record_start_time'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE cctv ADD COLUMN record_start_time TIME NULL")

        cur.execute("SHOW COLUMNS FROM cctv LIKE 'record_end_time'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE cctv ADD COLUMN record_end_time TIME NULL")

        cur.execute("SHOW COLUMNS FROM cctv LIKE 'min_session_duration'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE cctv ADD COLUMN min_session_duration INT DEFAULT 10")

        # === Buat tabel zones ===
        cur.execute("""
            CREATE TABLE IF NOT EXISTS zones (
                id INT AUTO_INCREMENT PRIMARY KEY,
                camera_id INT NOT NULL,
                zone_name VARCHAR(255) NOT NULL,
                coordinates TEXT NOT NULL,
                FOREIGN KEY (camera_id) REFERENCES cctv(id) ON DELETE CASCADE
            )
        """)

        cur.execute("SHOW COLUMNS FROM person_sessions LIKE 'zone_id'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE person_sessions ADD COLUMN zone_id INT NULL AFTER camera_id")

        cur.execute("SHOW COLUMNS FROM zones LIKE 'max_people'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE zones ADD COLUMN max_people INT DEFAULT 0")

        cur.execute("SHOW COLUMNS FROM zones LIKE 'max_inactive_duration'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE zones ADD COLUMN max_inactive_duration INT DEFAULT 0")

        # === Buat tabel empty_zone_sessions untuk zona tidak aktif ===
        cur.execute("""
            CREATE TABLE IF NOT EXISTS empty_zone_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                camera_id INT NOT NULL,
                zone_id INT NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME NULL,
                duration INT NULL,
                inactive_threshold INT DEFAULT 0,
                FOREIGN KEY (camera_id) REFERENCES cctv(id) ON DELETE CASCADE,
                FOREIGN KEY (zone_id) REFERENCES zones(id) ON DELETE CASCADE
            )
        """)

        conn.commit()
        app.logger.info("Skema database berhasil diverifikasi dan diperbarui.")

    except Exception as e:
        app.logger.error(f"ensure_schema() GAGAL: {e}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def init_camera_data(cams: List[Dict[str, Any]]) -> None:
    all_camera_zones = load_all_zones_from_db()
    CAM_ZONES.clear()
    MIN_SESSION_DURATIONS.clear()
    for cam in cams:
        cid = cam["id"]
        latest_frame.setdefault(cid, None)
        annotated_frame.setdefault(cid, None)
        people_count.setdefault(cid, 0)
        writer_info.setdefault(cid, {"fps": TARGET_RECORD_FPS, "size": None, "filename": None})
        recording_status.setdefault(cid, False)
        stop_flags.setdefault(cid, False)
        recording_locks.setdefault(cid, threading.Lock())
        thread_locks.setdefault(cid, threading.Lock())
        last_write_time.setdefault(cid, 0.0)
        MIN_SESSION_DURATIONS[cid] = cam.get("min_session_duration", 10)

        CAM_ZONES[cid] = all_camera_zones.get(cid, [])
        if not CAM_ZONES[cid]:
            zone_points = []
            z = cam.get("zone")
            if isinstance(z, str) and z.strip():
                try: zone_points = ast.literal_eval(z)
                except (ValueError, SyntaxError): zone_points = []
            if isinstance(zone_points, list) and len(zone_points) >= 3:
                app.logger.warning(f"Kamera {cid} menggunakan zona lama. Harap migrasi ke sistem multi-zona.")
                CAM_ZONES[cid] = [{"id": 0, "name": "Zona Default", "points": zone_points, "max_people": 0, "notified_full": False}]

def close_writer(cam_id: int):
    with recording_locks.get(cam_id, threading.Lock()):
        w = writers.get(cam_id)
        if w is not None:
            try: w.release()
            except Exception as e: app.logger.error(f"Gagal menutup writer untuk kamera {cam_id}: {e}")
        app.logger.info(f"Rekaman dihentikan kamera {cam_id}")
        writers[cam_id] = None
        info = writer_info.get(cam_id) or {}
        if "filename" in info:
            info["filename"] = None
            writer_info[cam_id] = info
        last_write_time[cam_id] = 0.0

def capture_thread_fn(cam_id: int, rtsp_url: str):
    cap = None
    backoff = RTSP_OPEN_RETRY_SECONDS
    app.logger.info(f"Starting capture thread for camera {cam_id}")
    while not stop_flags.get(cam_id, False):
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if not cap.isOpened(): cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                app.logger.warning(f"Camera {cam_id} cannot open, retry in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 30.0)
                continue
            try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception: pass
            backoff = RTSP_OPEN_RETRY_SECONDS
            app.logger.info(f"Camera {cam_id} opened")
        grabbed, frame = cap.read()
        if not grabbed or frame is None:
            app.logger.warning(f"Frame grab failed for camera {cam_id}. Reconnecting...")
            if cap: cap.release()
            cap = None
            time.sleep(backoff)
            continue
        with frames_lock:
            latest_frame[cam_id] = frame
            if writer_info.get(cam_id, {}).get("size") is None:
                writer_info[cam_id]["size"] = (frame.shape[1], frame.shape[0])
        time.sleep(0.001)
    if cap is not None:
        try: cap.release()
        except Exception: pass
    app.logger.info(f"Capture thread stopped for camera {cam_id}")

def format_duration(td):
    if not td:
        return "-"
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    else:
        return f"{minutes:02}:{seconds:02}"

def process_detection(frame, cam_id: int):
    global active_sessions
    try:
        if frame is None or model is None:
            return frame, 0, "Unknown"

        resize_for_det = DETECTION_SIZE
        small = cv2.resize(frame, resize_for_det)
        results = model.track(
            small, persist=True, classes=[0],
            verbose=False, conf=0.25, iou=0.4, max_det=300
        )
        annotated = frame.copy()
        
        min_duration = MIN_SESSION_DURATIONS.get(cam_id, 10)
        zones_for_cam = CAM_ZONES.get(cam_id, [])
        fw, fh = frame.shape[1], frame.shape[0]
        rw, rh = resize_for_det
        scale_x, scale_y = rw / fw, rh / fh
        
        for zone in zones_for_cam:
            original_points_np = np.array(zone['points'], np.intp)
            cv2.polylines(annotated, [original_points_np], isClosed=True, color=(255, 255, 0), thickness=2)
            label_pos = (original_points_np[0][0], original_points_np[0][1] - 10)
            cv2.putText(annotated, zone['name'], label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if cam_id not in active_sessions:
            active_sessions[cam_id] = {}

        current_person_locations = {}
        if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
            for r in results:
                if not hasattr(r, "boxes") or not hasattr(r.boxes, 'id') or r.boxes is None:
                    continue
                for box in r.boxes:
                    tracking_id = int(box.id[0]) if box.id is not None else -1
                    if tracking_id == -1:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    for zone in zones_for_cam:
                        scaled_zone_points = [[int(x * scale_x), int(y * scale_y)] for x, y in zone['points']]
                        if is_inside_zone((x1, y1, x2, y2), scaled_zone_points):
                            current_person_locations[tracking_id] = {
                                'zone_id': zone['id'],
                                'zone_name': zone['name'],
                                'max_people': zone.get('max_people', 0)
                            }
                            break

        active_cam_sessions = active_sessions[cam_id]
        current_tracking_ids = set(current_person_locations.keys())
        previous_tracking_ids = set(active_cam_sessions.keys())

        zone_status = "Zona Tidak Aktif"
        if len(current_tracking_ids) > len(previous_tracking_ids):  
            zone_status = "Zona Aktif"
            app.logger.info(f"[CAM {cam_id}] Zona Aktif (orang bertambah)")
        elif len(current_tracking_ids) < len(previous_tracking_ids):  
            zone_status = "Zona Tidak Aktif"
            app.logger.info(f"[CAM {cam_id}] Zona Tidak Aktif (orang berkurang)")

        for tid in (current_tracking_ids - previous_tracking_ids):
            zone_info = current_person_locations[tid]
            active_cam_sessions[tid] = {
                'start_time': datetime.now(),
                'zone_id': zone_info['zone_id'],
                'zone_name': zone_info['zone_name'],
                'max_people': zone_info.get('max_people', 0)
            }
            app.logger.info(f"Orang masuk zona: cam={cam_id}, tid={tid}, zona={zone_info['zone_name']}")

        for tid in (previous_tracking_ids - current_tracking_ids):
            session_data = active_cam_sessions.pop(tid, None)
            if session_data:
                duration = (datetime.now() - session_data['start_time']).total_seconds()
                end_time = datetime.now()
                session_db_id = save_person_session_start_with_zone(
                    cam_id, tid, session_data['zone_id'], session_data['start_time']
                )
                if session_db_id:
                    update_person_session_end(session_db_id, int(duration))
                app.logger.info(
                    f"Sesi selesai (keluar): cam={cam_id}, tid={tid}, zona={session_data['zone_name']}, durasi={duration:.2f}s"
                )

        # Simpan sesi kosong jika sekarang tidak ada orang sama sekali
        if len(current_tracking_ids) == 0 and len(previous_tracking_ids) > 0:
            for zone in zones_for_cam:
                zone_id = zone['id']
                save_empty_zone_session(cam_id, zone_id)
                app.logger.info(f"[CAM {cam_id}] Zona {zone['name']} kosong → simpan sesi kosong ke DB")

        # Gambar bounding box orang
        total_people_in_zones = len(current_person_locations)
        if results and hasattr(results[0], "boxes"):
            for r in results:
                if not hasattr(r, "boxes") or not hasattr(r.boxes, 'id') or r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    tracking_id = int(box.id[0]) if box.id is not None else -1
                    ox1, oy1, ox2, oy2 = int(x1 * fw/rw), int(y1 * fh/rh), int(x2 * fw/rw), int(y2 * fh/rh)
                    location_info = current_person_locations.get(tracking_id)
                    if location_info:
                        color, text = (0, 255, 0), f"ID:{tracking_id} @ {location_info['zone_name']}"
                    else: 
                        color, text = (0, 0, 255), f"ID:{tracking_id}"
                    cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), color, 2)
                    cv2.putText(annotated, text, (ox1, oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Tampilkan status Zona
        display_text = f"{zone_status} | Orang: {total_people_in_zones}"
        cv2.putText(annotated, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        return annotated, total_people_in_zones, zone_status

    except Exception as e:
        app.logger.error(f"Error di process_detection camera {cam_id}: {e}", exc_info=True)
        return frame, 0, "Error"


def init_writer_if_needed(cam_id: int, frame, fps: float):
    with recording_locks.get(cam_id, threading.Lock()):
        if writers.get(cam_id) is None:
            w, h = frame.shape[1], frame.shape[0]
            filename = os.path.join(RECORDINGS_DIR, f"cam_{cam_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
            if not writer.isOpened():
                app.logger.error(f"Gagal membuat VideoWriter untuk {filename}")
                return
            writers[cam_id] = writer
            writer_info[cam_id]["filename"] = filename
            app.logger.info(f"Rekaman dimulai: {filename}")

def detect_and_record_thread_fn(cam_id: int):
    app.logger.info(f"Starting detection thread for camera {cam_id}")
    while not stop_flags.get(cam_id, False):
        frame = None
        with frames_lock: frame = latest_frame.get(cam_id)
        if frame is None:
            time.sleep(0.05)
            continue
        annotated, current_count, status = process_detection(frame, cam_id)
        with frames_lock:
            annotated_frame[cam_id] = annotated
            people_count[cam_id] = current_count
        if recording_status.get(cam_id, False):
            fps = max(6.0, float(writer_info.get(cam_id, {}).get("fps", TARGET_RECORD_FPS / SLOW_FACTOR)))
            interval = 1.0 / fps
            now = time.time()
            with recording_locks.setdefault(cam_id, threading.Lock()):
                init_writer_if_needed(cam_id, annotated, fps=fps)
                w = writers.get(cam_id)
                if w is not None and (now - last_write_time.get(cam_id, 0.0) >= interval / SLOW_FACTOR):
                    try:
                        w.write(annotated)
                        last_write_time[cam_id] = now
                    except Exception as e:
                        app.logger.error(f"Gagal menulis frame rekaman cam {cam_id}: {e}")
                        close_writer(cam_id)
        time.sleep(0.03)

def generate_stream(cam_id: int):
    while not stop_flags.get(cam_id, False):
        frame = annotated_frame.get(cam_id)
        if frame is None:
            time.sleep(0.1)
            continue
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret: continue
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    app.logger.info(f"Streaming for camera {cam_id} stopped.")

def start_camera_threads():
    cams = load_cameras_from_db()
    active_cams = [c for c in cams if c.get("is_active", True)]
    init_camera_data(active_cams)
    for cam in active_cams:
        cid = cam["id"]
        rtsp = cam.get("rtsp_url") or ""
        stop_flags[cid] = False
        if cid not in capture_threads or not capture_threads[cid].is_alive():
            tcap = threading.Thread(target=capture_thread_fn, args=(cid, rtsp), daemon=True)
            capture_threads[cid] = tcap
            tcap.start()
        if cid not in detect_threads or not detect_threads[cid].is_alive():
            tdet = threading.Thread(target=detect_and_record_thread_fn, args=(cid,), daemon=True)
            detect_threads[cid] = tdet
            tdet.start()
    all_cam_ids = {cam["id"] for cam in cams}
    for cid in list(capture_threads.keys()):
        if cid not in all_cam_ids:
            stop_flags[cid] = True
            time.sleep(1)
            if cid in capture_threads: del capture_threads[cid]
            if cid in detect_threads: del detect_threads[cid]

def save_empty_zone_session(cam_id, zone_id):
    conn = get_connection()
    cur = conn.cursor()
    try:
        now = datetime.now()
        cur.execute("""
            INSERT INTO person_sessions (camera_id, zone_id, start_time, end_time, duration, people_count)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (cam_id, zone_id, now, now, 0, 0))
        conn.commit()
    finally:
        cur.close()
        conn.close()

@app.route("/")
def index():
    cams = load_cameras_from_db()
    start_camera_threads()
    return render_template("index.html", cameras=cams, recording_status=recording_status, people_count=people_count)

@app.route("/video_feed/<int:camera_id>")
def video_feed(camera_id: int):
    return Response(generate_stream(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/events_json")
def events_json():
    try:
        # Ambil riwayat sesi dari DB
        events_db = get_person_sessions_with_zones(limit=50)
        out = []

        # Ambil data kamera
        all_cams = {c['id']: c for c in load_cameras_from_db()}

        # Tambahkan jumlah orang ke sesi aktif
        for cam_id, sessions in active_sessions.items():
            current_count = people_count.get(cam_id, 0)
            for tid, session_data in sessions.items():
                session_data['people_count'] = current_count

        # Dapatkan semua zona yang sedang aktif
        active_zones_current = set()
        for cam_id, sessions in active_sessions.items():
            for tid, s in sessions.items():
                zid = s.get('zone_id')
                if zid is not None:
                    active_zones_current.add((cam_id, zid))

        # Ambil sesi terakhir per zona dari DB
        conn = get_connection()
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute("""
                SELECT ps.camera_id, ps.zone_id, MAX(ps.end_time) AS last_end
                FROM person_sessions ps
                GROUP BY ps.camera_id, ps.zone_id
            """)
            last_sessions = cur.fetchall()
        finally:
            cur.close()
            conn.close()

        # Buat peta last_end untuk menghitung durasi tidak aktif
        last_end_map = {
            (ls["camera_id"], ls["zone_id"]): ls["last_end"]
            for ls in (last_sessions or [])
            if ls.get("last_end") is not None
        }

        now = datetime.now()

        # Tambahkan sesi aktif ke daftar output
        for cam_id, sessions in active_sessions.items():
            camera_name = all_cams.get(cam_id, {}).get('name', f"Camera {cam_id}")
            for tid, session_data in sessions.items():
                start_time = session_data.get('start_time')
                if not start_time:
                    continue
                sec = max(1, int((now - start_time).total_seconds()))
                duration_str = format_duration(timedelta(seconds=sec))
                out.append({
                    "id": f"active_{cam_id}_{tid}",
                    "camera_id": cam_id,
                    "camera_name": camera_name,
                    "zone_name": session_data.get('zone_name', 'N/A'),
                    "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": "Sesi Aktif",
                    "duration": duration_str,
                    "inactive_duration": "-",
                    "people_count": session_data.get('people_count', 0),
                    "notification": session_data.get('notification', "")
                })

        # Tambahkan notifikasi tertunda ke daftar output
        while pending_notifications:
            message = pending_notifications.popleft()
            out.append({
                "id": f"pending_{datetime.now().timestamp()}",
                "camera_id": 0,
                "camera_name": "N/A",
                "zone_name": "N/A",
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": "N/A",
                "duration": "N/A",
                "inactive_duration": "N/A",
                "people_count": 0,
                "notification": message
            })

        # Tambahkan sesi yang sudah selesai ke daftar output
        for e in events_db:
            cam_id = e.get("camera_id")
            zone_id = e.get("zone_id")
            camera_name = all_cams.get(cam_id, {}).get('name', 'N/A')

            # Durasi sesi
            if e.get("duration") is not None:
                duration_str = format_duration(timedelta(seconds=int(e["duration"])))
            else:
                duration_str = "N/A"

            # Hitung durasi tidak aktif (jika zona sedang kosong)
            inactive_duration = "-"
            e_end = e.get("end_time")
            if e_end:
                last_end = last_end_map.get((cam_id, zone_id))
                if last_end and last_end == e_end:
                    if (cam_id, zone_id) not in active_zones_current:
                        diff_seconds = max(1, int((now - e_end).total_seconds()))
                        inactive_duration = format_duration(timedelta(seconds=diff_seconds))
                    else:
                        inactive_duration = "-"

            out.append({
                "id": e.get("id"),
                "camera_id": cam_id,
                "camera_name": camera_name,
                "zone_name": e.get("zone_name", "Tanpa Zona") if e.get("zone_id") else "Tanpa Zona",
                "start_time": e.get("start_time").strftime("%Y-%m-%d %H:%M:%S") if e.get("start_time") else "N/A",
                "end_time": e_end.strftime("%Y-%m-%d %H:%M:%S") if e_end else "N/A",
                "duration": duration_str,
                "inactive_duration": inactive_duration,
                "people_count": e.get("people_count", 0),
                "notification": ""
            })

        # Urutkan terbaru dulu berdasarkan start_time
        out.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        return jsonify(out)

    except Exception as ex:
        app.logger.error(f"Gagal ambil events_json: {ex}", exc_info=True)
        return jsonify([]), 500


@app.route("/add_camera", methods=["GET", "POST"])
def add_camera():
    if request.method == "POST":
        name = request.form.get("name")
        rtsp_url = request.form.get("rtsp_url")
        if not name or not rtsp_url:
            flash("Nama kamera dan URL RTSP wajib diisi.", "error")
            return redirect(url_for("add_camera"))
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO cctv (name, rtsp_url, is_active) VALUES (%s, %s, TRUE)",(name, rtsp_url))
            conn.commit()
            flash("Kamera berhasil ditambahkan!", "success")
        except Exception as e:
            conn.rollback()
            app.logger.error(f"Error adding camera: {e}")
            flash(f"Error menambah kamera: {e}", "error")
        finally:
            cursor.close()
            conn.close()
        start_camera_threads()
        return redirect(url_for("index"))
    return render_template("add_camera.html")

@app.route("/edit-zone/<int:camera_id>", methods=["GET", "POST"])
def edit_zone(camera_id):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT * FROM person_sessions WHERE id = %s", (camera_id,))
    camera = cur.fetchone()

    cur.execute("SELECT * FROM zones WHERE camera_id = %s", (camera_id,))
    zones = cur.fetchall()

    cur.execute("SELECT inactive_threshold FROM empty_zone_sessions WHERE camera_id = %s LIMIT 1", (camera_id,))
    empty_session = cur.fetchone()

    cur.close()
    conn.close()

    return render_template(
        "edit_zone.html",
        camera=camera,
        zones=zones,
        empty_session=empty_session,   # <-- kirim ke template
    )


@app.route("/delete_zone_new/<int:zone_id>", methods=["POST"])
def delete_zone_new(zone_id):
    conn = get_connection()
    cursor = conn.cursor()
    camera_id = None
    try:
        cursor.execute("SELECT camera_id FROM zones WHERE id = %s", (zone_id,))
        result = cursor.fetchone()
        if result: camera_id = result[0]
        cursor.execute("DELETE FROM zones WHERE id = %s", (zone_id,))
        conn.commit()
        flash("Zona berhasil dihapus.", "success")
        threading.Thread(target=start_camera_threads, daemon=True).start()
    except Exception as e:
        conn.rollback()
        app.logger.error(f"Gagal menghapus zona {zone_id}: {e}")
        flash(f"Gagal menghapus zona: {e}", "error")
    finally:
        cursor.close()
        conn.close()
    return redirect(url_for("edit_zone", camera_id=camera_id)) if camera_id else redirect(url_for("index"))

@app.route("/person_count/<int:camera_id>")
def person_count_route(camera_id: int):
    return jsonify({"count": int(people_count.get(camera_id, 0))})

@app.route("/download_events")
def download_events():
    try:
        export_path = os.path.join(EXPORTS_DIR, "sessions.xlsx")
        
        # Ekspor ke 2 sheet dalam 1 file
        import pandas as pd
        from database import get_person_sessions, get_empty_zone_sessions

        # Ambil data dari DB
        person_sessions = get_person_sessions(limit=1000)
        empty_zone_sessions = get_empty_zone_sessions(limit=1000)

        # Buat DataFrame
        df_person = pd.DataFrame(person_sessions)
        df_empty = pd.DataFrame(empty_zone_sessions)

        # Simpan ke Excel (2 sheet)
        with pd.ExcelWriter(export_path) as writer:
            df_person.to_excel(writer, sheet_name="Person Sessions", index=False)
            df_empty.to_excel(writer, sheet_name="Empty Zone Sessions", index=False)

        return send_file(export_path, as_attachment=True)
    except Exception as e:
        app.logger.error(f"Gagal download sessions: {e}")
        return "Gagal download sessions", 500


@app.route("/toggle_record/<int:camera_id>", methods=["POST"])
def toggle_record(camera_id: int):
    new_status = not recording_status.get(camera_id, False)
    recording_status[camera_id] = new_status
    app.logger.info(f"Toggle record cam {camera_id} -> {new_status}")
    if not new_status:
        close_writer(camera_id)
    return redirect(url_for("index"))

@app.route("/deactivate/<int:camera_id>", methods=["POST"])
def deactivate_route(camera_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE cctv SET is_active = FALSE WHERE id = %s", (camera_id,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()
    stop_flags[camera_id] = True
    return redirect(url_for("index"))

@app.route("/activate/<int:camera_id>", methods=["POST"])
def activate_route(camera_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE cctv SET is_active = TRUE WHERE id = %s", (camera_id,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()
    start_camera_threads()
    return redirect(url_for("index"))

@app.route("/delete_camera/<int:camera_id>", methods=["POST"])
def delete_camera(camera_id: int):
    stop_flags[camera_id] = True
    time.sleep(1)

    conn = get_connection()
    cursor = conn.cursor(dictionary=True) # Tambahkan dictionary=True agar hasilnya dalam bentuk dict
    try:
        # Periksa apakah kamera ada sebelum mencoba menghapusnya
        cursor.execute("SELECT id FROM cctv WHERE id = %s", (camera_id,))
        result = cursor.fetchone()

        if result:  # Periksa apakah result tidak None
            # Jika kamera ditemukan, lanjutkan proses penghapusan
            cursor.execute("DELETE FROM cctv WHERE id = %s", (camera_id,))
            conn.commit()
            flash(f"Kamera {camera_id} berhasil dihapus.", "success")
        else:
            # Jika kamera tidak ditemukan, berikan pesan kesalahan
            flash(f"Kamera {camera_id} tidak ditemukan.", "error")

    except Exception as e:
        app.logger.error(f"Gagal menghapus kamera {camera_id}: {e}")
        conn.rollback()
        flash(f"Gagal menghapus kamera: {e}", "error")
    finally:
        cursor.close()
        conn.close()

    return redirect(url_for("index"))

@app.route('/save_camera_settings/<int:camera_id>', methods=['POST'])
def save_camera_settings(camera_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        zone_name = request.form.get("zone_name")
        zone_coordinates_str = request.form.get("zone_coordinates")
        zone_max_people = request.form.get("zone_max_people")
        if zone_name and zone_coordinates_str and zone_coordinates_str != '[]':
            try:
                points = json.loads(zone_coordinates_str)
                if len(points) >= 3:
                    max_people_value = int(zone_max_people) if zone_max_people and zone_max_people.isdigit() else 0
                    cursor.execute(
                        "INSERT INTO zones (camera_id, zone_name, coordinates, max_people) VALUES (%s, %s, %s, %s)",
                        (camera_id, zone_name, json.dumps(points), max_people_value)
                    )
                    conn.commit()
                    flash(f"Zona '{zone_name}' berhasil ditambahkan.", "success")
                else:
                    flash("Zona baru harus memiliki minimal 3 titik.", "error")
            except (json.JSONDecodeError, ValueError):
                flash("Format koordinat zona atau jumlah orang tidak valid.", "error")
            except Exception as e:
                app.logger.error(f"Gagal menambahkan zona: {e}")
                flash(f"Terjadi error saat menambahkan zona: {e}", "error")
        
        try:
            hours = int(request.form.get('min_duration_hours', '0') or '0')
            minutes = int(request.form.get('min_duration_minutes', '0') or '0')
            seconds = int(request.form.get('min_duration_seconds', '0') or '0')
            total_duration = (hours * 3600) + (minutes * 60) + seconds

            cursor.execute("UPDATE cctv SET min_session_duration = %s WHERE id = %s", (total_duration, camera_id))
            conn.commit()
            flash('Durasi sesi minimal berhasil diperbarui.', 'success')
        except (ValueError, TypeError):
            flash('Gagal memperbarui durasi sesi. Pastikan input valid.', 'danger')
        
        threading.Thread(target=start_camera_threads, daemon=True).start()

    except Exception as e:
        if conn and conn.is_connected():
            conn.rollback()
        app.logger.error(f"Kesalahan umum saat menyimpan pengaturan: {e}")
        flash(f"Terjadi kesalahan saat menyimpan pengaturan: {e}", "danger")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

    return redirect(url_for('edit_zone', camera_id=camera_id))

if __name__ == "__main__":
    ensure_schema()
    start_camera_threads()
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
