import os
import time
import threading
from typing import Tuple, List, Dict, Any
import numpy as np
from datetime import datetime
import cv2
import json
import ast

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
from database import load_cameras_from_db, get_connection

MODEL_PATH = os.environ.get("YOLO_MODEL", "yolov8n.pt")
RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)
DETECTION_SIZE = (960, 540)
TARGET_RECORD_FPS = float(os.environ.get("TARGET_RECORD_FPS", "25.0"))
SLOW_FACTOR = float(os.environ.get("SLOW_FACTOR", "1.25"))
RTSP_OPEN_RETRY_SECONDS = float(os.environ.get("RTSP_RETRY", "2.0"))
EXPORTS_DIR = os.environ.get("EXPORTS_DIR", "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)

DEBUG_DRAW_ALL = True

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
last_people_count: Dict[int, int] = {}
last_write_time: Dict[int, float] = {}
CAM_ZONES: Dict[int, List[Tuple[int, int]]] = {}

# ZONES_FALLBACK dihapus
# ZONES_FALLBACK = {
#     1: [[719, 40], [1339, 49], [1331, 629], [639, 653]]
# }

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    app.logger.error(f"Gagal load YOLO model di {MODEL_PATH}: {e}")
    model = None

def is_inside_zone(bbox, zone_points, min_overlap_ratio=0.2): 
    bx1, by1, bx2, by2 = bbox
    bbox_area = (bx2 - bx1) * (by2 - by1)
    if bbox_area <= 0:
        return False
    
    # Check if bbox coordinates are valid for image dimensions
    if bx1 > DETECTION_SIZE[0] or by1 > DETECTION_SIZE[1] or bx2 < 0 or by2 < 0:
        return False
    
    pts = np.array(zone_points, np.int32)
    poly_mask = np.zeros(DETECTION_SIZE[::-1], dtype=np.uint8)
    cv2.fillPoly(poly_mask, [pts], 255)
    
    bbox_mask = np.zeros_like(poly_mask)
    # Ensure bbox coordinates are within image bounds before creating mask
    x_min = max(0, int(bx1))
    y_min = max(0, int(by1))
    x_max = min(DETECTION_SIZE[0], int(bx2))
    y_max = min(DETECTION_SIZE[1], int(by2))
    
    cv2.rectangle(bbox_mask, (x_min, y_min), (x_max, y_max), 255, -1)
    
    intersection_mask = cv2.bitwise_and(poly_mask, bbox_mask)
    intersection_area = np.sum(intersection_mask > 0)
    
    # Use max(1, bbox_area) to prevent division by zero
    overlap_ratio = intersection_area / max(1, bbox_area)
    return overlap_ratio >= min_overlap_ratio

def ensure_schema() -> None:
    """Pastikan semua tabel inti tersedia."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cctv (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                rtsp_url TEXT,
                zone TEXT NULL,
                is_active TINYINT(1) DEFAULT 1
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS person_exit (
                id INT AUTO_INCREMENT PRIMARY KEY,
                camera_id INT NOT NULL,
                exit_time DATETIME NOT NULL,
                description VARCHAR(255),
                INDEX (camera_id),
                FOREIGN KEY (camera_id) REFERENCES cctv(id) ON DELETE CASCADE
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS people_detection (
                id INT AUTO_INCREMENT PRIMARY KEY,
                camera_id INT NOT NULL,
                count INT NOT NULL,
                timestamp DATETIME NOT NULL,
                INDEX (camera_id),
                FOREIGN KEY (camera_id) REFERENCES cctv(id) ON DELETE CASCADE
            )
            """
        )
        conn.commit()
    except Exception as e:
        app.logger.warning(f"ensure_schema() warning: {e}")
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass

def save_event(cam_id: int, description: str):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO person_exit (camera_id, exit_time, description) VALUES (%s, %s, %s)",
            (cam_id, datetime.now(), description)
        )
        conn.commit()
    except Exception as e:
        app.logger.error(f"Gagal simpan event ke DB: {e}")
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass

def save_people_count(cam_id: int, count: int):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO people_detection (camera_id, count, timestamp) VALUES (%s, %s, %s)",
            (cam_id, int(count), datetime.now()),
        )
        conn.commit()
    except Exception as e:
        app.logger.error(f"Gagal simpan people_detection: {e}")
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass

def get_recent_events(limit: int = 15) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        conn = get_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            """
            SELECT pe.id, pe.camera_id, pe.exit_time, pe.description,
                   COALESCE(c.name, CONCAT('Camera ', pe.camera_id)) AS camera_name
            FROM person_exit pe
            LEFT JOIN cctv c ON c.id = pe.camera_id
            ORDER BY pe.exit_time DESC
            LIMIT %s
            """,
            (int(limit),),
        )
        rows = cur.fetchall() or []
    except Exception as e:
        app.logger.error(f"Gagal get_recent_events: {e}")
        rows = []
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass
    return rows

def export_events_to_excel(filepath: str) -> str:
    try:
        import pandas as pd
    except ImportError:
        app.logger.error("Pandas tidak tersedia. Silakan install dengan 'pip install pandas openpyxl'.")
        raise
    
    data = get_recent_events(limit=10000)
    for d in data:
        ts = d.get("exit_time")
        if hasattr(ts, "strftime"):
            d["exit_time"] = ts.strftime("%Y-%m-%d %H:%M:%S")
    
    df = pd.DataFrame(data, columns=["id", "camera_id", "camera_name", "exit_time", "description"])
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        df.to_excel(filepath, index=False)
        return filepath
    except Exception as e:
        app.logger.error(f"Gagal menulis Excel: {e}")
        raise

def init_camera_data(cams: List[Dict[str, Any]]) -> None:
    CAM_ZONES.clear()
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
        last_people_count.setdefault(cid, 0)
        last_write_time.setdefault(cid, 0.0)


        zone_points = []
        z = cam.get("zone")
        
        if isinstance(z, str) and z.strip():  
            try:
                zone_points = json.loads(z.strip())
            except (json.JSONDecodeError, TypeError) as e:
                app.logger.warning(f"Zone tidak valid untuk kamera {cid}: '{z}' -> {e}")
                zone_points = []
        elif isinstance(z, list) and z:
             zone_points = z

        if not zone_points or len(zone_points) < 3:
            app.logger.warning(f"Tidak ada zona valid di DB untuk kamera {cid}. Deteksi akan dihitung di seluruh frame.")
            CAM_ZONES[cid] = []
            continue

        parsed_points = []
        try:
            for p in zone_points:
                if isinstance(p, (list, tuple)) and len(p) == 2:
                    parsed_points.append([int(p[0]), int(p[1])])
                else:
                    raise ValueError("Format titik zona tidak sesuai.")
        except (ValueError, IndexError) as e:
            app.logger.error(f"Gagal memproses titik zona untuk kamera {cid}: {e}")
            parsed_points = []
        
        CAM_ZONES[cid] = parsed_points

def init_writer_if_needed(cam_id: int, frame, fps=TARGET_RECORD_FPS):
    """Inisialisasi writer jika belum ada dan status rekam ON."""
    if writers.get(cam_id) is None and frame is not None:
        size = (frame.shape[1], frame.shape[0])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        filename = os.path.join(
            RECORDINGS_DIR, f"cam{cam_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
        vw = cv2.VideoWriter(filename, fourcc, fps, size)
        if not vw or not vw.isOpened():
            app.logger.error(f"VideoWriter gagal dibuka untuk kamera {cam_id}")
            writers[cam_id] = None
        else:
            writers[cam_id] = vw
            writer_info[cam_id] = {"filename": filename, "fps": fps, "size": size}
            app.logger.info(
                f"Rekaman mulai kamera {cam_id} ke {filename} (fps={fps:.2f}, size={size})"
            )
            
def close_writer(cam_id: int):
    w = writers.get(cam_id)
    if w is not None:
        try:
            w.release()
        except Exception:
            pass
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
    try:
        while not stop_flags.get(cam_id, False):
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    app.logger.warning(
                        f"Camera {cam_id} cannot open, retry in {backoff}s..."
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 1.5, 30.0)
                    continue
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                backoff = RTSP_OPEN_RETRY_SECONDS
                app.logger.info(f"Camera {cam_id} opened")

            grabbed, frame = cap.read()
            if not grabbed or frame is None:
                app.logger.warning(f"Frame grab failed for camera {cam_id}. Reconnecting...")
                if cap:
                    cap.release()
                cap = None
                time.sleep(backoff)
                continue
            with frames_lock:
                latest_frame[cam_id] = frame
                if writer_info.get(cam_id, {}).get("size") is None:
                    writer_info[cam_id]["size"] = (frame.shape[1], frame.shape[0])
            time.sleep(0.001)
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        app.logger.info(f"Capture thread stopped for camera {cam_id}")

def process_detection(frame, cam_id: int):
    try:
        if frame is None or model is None:
            return frame, 0, "Unknown"
        resize_for_det = DETECTION_SIZE
        small = cv2.resize(frame, resize_for_det)
        
        results = model.predict(
            small,
            classes=[0],
            verbose=False,
            conf=0.25,
            iou=0.4,
            max_det=300
        )
        
        annotated = frame.copy()
        count = 0
        status_text = "Kosong"
        zone_points = CAM_ZONES.get(cam_id) or []
        scaled_zone_points = []
        if zone_points and len(zone_points) >= 3:
            fw, fh = frame.shape[1], frame.shape[0]
            rw, rh = resize_for_det
            scale_x = rw / fw
            scale_y = rh / fh
            scaled_zone_points = [[int(x * scale_x), int(y * scale_y)] for x, y in zone_points]
            zone_np = np.array(zone_points, np.int32)
            cv2.polylines(annotated, [zone_np], isClosed=True, color=(255, 255, 0), thickness=2)
        
        # Check if there are any people detected
        if any(len(r.boxes) > 0 for r in results):
            status_text = "Orang terdeteksi"

        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                is_inside = False
                if scaled_zone_points:
                    is_inside = is_inside_zone((x1, y1, x2, y2), scaled_zone_points, min_overlap_ratio=0.2)
                
                # Scale the bounding box back to the original frame size
                ox1 = int(x1 * (frame.shape[1] / resize_for_det[0]))
                oy1 = int(y1 * (frame.shape[0] / resize_for_det[1]))
                ox2 = int(x2 * (frame.shape[1] / resize_for_det[0]))
                oy2 = int(y2 * (frame.shape[0] / resize_for_det[1]))

                if is_inside:
                    count += 1
                    status_text = "Orang ada di tempat"
                    color = (0, 255, 0)
                    text = "IN"
                else:
                    color = (0, 0, 255)
                    text = "OUT"

                cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), color, 2)
                cv2.putText(annotated, f"{text} ({conf:.2f})", (ox1, oy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        display_text = f"Status: {status_text}, Count: {count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3.0
        thickness = 5
        margin = 20
        (text_width, text_height), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)
        x = frame.shape[1] - text_width - margin
        y = margin + text_height

        cv2.putText(
            annotated, display_text, (x, y),
            font, font_scale, (255, 0, 0), thickness
        )
        return annotated, count, status_text
    except Exception as e:
        app.logger.error(f"Error process_detection camera {cam_id}: {str(e)}")
        return frame, 0, "Error"

def detect_and_record_thread_fn(cam_id: int):
    app.logger.info(f"Starting detection thread for camera {cam_id}")
    while not stop_flags.get(cam_id, False):
        frame = None
        with frames_lock:
            frame = latest_frame.get(cam_id)
        if frame is None:
            time.sleep(0.05)
            continue
        annotated, current_count, status = process_detection(frame, cam_id)
        with frames_lock:
            annotated_frame[cam_id] = annotated
            people_count[cam_id] = current_count
        if last_people_count.get(cam_id, 0) > 0 and current_count == 0:
            save_event(cam_id, "Orang meninggalkan ruangan")
        elif last_people_count.get(cam_id, 0) == 0 and current_count > 0:
            save_event(cam_id, "Orang ada di tempat")
        last_people_count[cam_id] = current_count
        if recording_status.get(cam_id, False):
            fps = max(1.0, float(writer_info.get(cam_id, {}).get("fps", TARGET_RECORD_FPS)))
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
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
    app.logger.info(f"Streaming for camera {cam_id} stopped.")

def start_camera_threads():
    cams = load_cameras_from_db()
    active_cams = [c for c in cams if c.get("is_active", True)]
    init_camera_data(active_cams)

    for cam in active_cams:
        cid = cam["id"]
        rtsp = cam.get("rtsp_url") or cam.get("url") or cam.get("rtsp") or ""
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
            
# -------------------- Routes --------------------
@app.route("/exit_data")
def exit_data():
    try:
        conn = get_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM person_exit ORDER BY exit_time DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return jsonify(rows)
    except Exception as e:
        app.logger.error(f"/exit_data error: {e}")
        return jsonify([]), 500

@app.route("/")
def index():
    cams = load_cameras_from_db()
    start_camera_threads()
    try:
        events = get_recent_events(limit=15)
    except Exception as e:
        app.logger.error(f"Gagal ambil recent events: {e}")
        events = []
    return render_template(
        "index.html",
        cameras=cams,
        recording_status=recording_status,
        people_count=people_count,
        recent_events=events,
    )

@app.route("/video_feed/<int:camera_id>")
def video_feed(camera_id: int):
    return Response(
        generate_stream(camera_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route("/person_count/<int:camera_id>")
def person_count_route(camera_id: int):
    return jsonify({"count": int(people_count.get(camera_id, 0))})

@app.route("/download_events")
def download_events():
    try:
        export_path = os.path.join(EXPORTS_DIR, "events.xlsx")
        export_events_to_excel(export_path)
        return send_file(export_path, as_attachment=True)
    except Exception as e:
        app.logger.error(f"Gagal download events: {e}")
        return "Gagal download events", 500

@app.route("/toggle_record/<int:camera_id>", methods=["POST"])
def toggle_record(camera_id: int):
    recording_locks.setdefault(camera_id, threading.Lock())
    new_status = not recording_status.get(camera_id, False)
    recording_status[camera_id] = new_status
    app.logger.info(f"Toggle record cam {camera_id} -> {new_status}")

    if not new_status:
        with recording_locks[camera_id]:
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
    recording_status[camera_id] = False
    with recording_locks.setdefault(camera_id, threading.Lock()):
        close_writer(camera_id)
    time.sleep(1)
    if camera_id in capture_threads:
        del capture_threads[camera_id]
    if camera_id in detect_threads:
        del detect_threads[camera_id]
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

    stop_flags[camera_id] = False
    start_camera_threads()
    return redirect(url_for("index"))

@app.route("/shutdown_threads", methods=["POST"])
def shutdown_threads():
    cams = load_cameras_from_db()
    for cam in cams:
        cid = cam["id"]
        stop_flags[cid] = True
        recording_status[cid] = False
        with recording_locks.setdefault(cid, threading.Lock()):
            close_writer(cid)
    time.sleep(2)
    return "All threads stopped", 200

@app.route("/delete_camera/<int:camera_id>", methods=["POST"])
def delete_camera(camera_id: int):
    stop_flags[camera_id] = True
    recording_status[camera_id] = False
    with recording_locks.setdefault(camera_id, threading.Lock()):
        close_writer(camera_id)
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM cctv WHERE id = %s", (camera_id,))
        conn.commit()
    except Exception as e:
        app.logger.error(f"Failed to delete camera {camera_id}: {e}")
        flash(f"Failed to delete camera: {e}", "error")
    finally:
        cursor.close()
        conn.close()

    if camera_id in capture_threads:
        del capture_threads[camera_id]
    if camera_id in detect_threads:
        del detect_threads[camera_id]

    flash(f"Camera {camera_id} deleted successfully.", "success")
    return redirect(url_for("index"))

@app.route("/events_json")
def events_json():
    try:
        events = get_recent_events(limit=50)
        out = []
        for e in events:
            et = e.get("exit_time")
            if hasattr(et, "strftime"):
                et = et.strftime("%Y-%m-%d %H:%M:%S")
            out.append({
                "id": e.get("id"),
                "camera_id": e.get("camera_id"),
                "camera_name": e.get("camera_name"),
                "exit_time": et,
                "description": e.get("description"),
            })
        return jsonify(out)
    except Exception as e:
        app.logger.error(f"Gagal ambil events_json: {e}")
        return jsonify([]), 500

@app.route("/add_camera", methods=["GET", "POST"])
def add_camera():
    if request.method == "POST":
        name = request.form.get("name")
        rtsp_url = request.form.get("rtsp_url")

        if not name or not rtsp_url:
            flash("Camera name and RTSP URL are required.", "error")
            return redirect(url_for("add_camera"))

        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO cctv (name, rtsp_url, is_active) VALUES (%s, %s, TRUE)",
                (name, rtsp_url),
            )
            conn.commit()
            flash("Camera added successfully!", "success")
        except Exception as e:
            conn.rollback()
            app.logger.error(f"Error adding camera: {str(e)}")
            flash(f"Error adding camera: {str(e)}", "error")
        finally:
            cursor.close()
            conn.close()

        start_camera_threads()
        return redirect(url_for("index"))

    return render_template("add_camera.html")

@app.route("/edit_zone/<int:camera_id>")
def edit_zone(camera_id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name, zone FROM cctv WHERE id = %s", (camera_id,))
    camera = cursor.fetchone()
    cursor.close()
    conn.close()

    if not camera:
        flash("Camera not found.", "error")
        return redirect(url_for("index"))

    zone_data = camera.get("zone")
    parsed_zone = []
    if zone_data:
        try:
            parsed_zone = json.loads(zone_data)
        except (json.JSONDecodeError, TypeError):
            parsed_zone = []

    return render_template("edit_zone.html", camera=camera, current_zone=parsed_zone)

# Endpoint baru untuk menyimpan zona ke DB
@app.route("/set_zone/<int:camera_id>", methods=["POST"])
def set_zone(camera_id):
    zone_data = request.form.get("zone_coordinates")
    if not zone_data:
        flash("No zone data received.", "error")
        return redirect(url_for("edit_zone", camera_id=camera_id))
    
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "UPDATE cctv SET zone = %s WHERE id = %s",
            (zone_data, camera_id)
        )
        conn.commit()
        flash("Zone saved successfully!", "success")
        
        # Muat ulang konfigurasi zona untuk thread deteksi
        start_camera_threads() 

    except Exception as e:
        conn.rollback()
        app.logger.error(f"Error saving zone for camera {camera_id}: {str(e)}")
        flash(f"Error saving zone: {str(e)}", "error")
    finally:
        cursor.close()
        conn.close()
    
    return redirect(url_for("index"))

# -------------------- Main --------------------
if __name__ == "__main__":
    start_camera_threads()
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)