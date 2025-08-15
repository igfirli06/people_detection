import os
import time
import threading
from typing import Tuple, List, Dict, Any
import numpy as np
from datetime import datetime
import cv2
import random

from flask import (
    Flask,
    Response,
    render_template,
    redirect,
    url_for,
    jsonify,
    request,
    flash,
)

from ultralytics import YOLO
from database import load_cameras_from_db, get_connection

MODEL_PATH = os.environ.get("YOLO_MODEL", "yolov8n.pt")
RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)
DETECTION_SKIP = int(os.environ.get("DETECTION_SKIP", "3"))
DETECTION_SIZE = (960, 540)
TARGET_RECORD_FPS = float(os.environ.get("TARGET_RECORD_FPS", "25.0"))
SLOW_FACTOR = float(os.environ.get("SLOW_FACTOR", "1.25"))
RTSP_OPEN_RETRY_SECONDS = float(os.environ.get("RTSP_RETRY", "2.0"))
EXPORTS_DIR = os.environ.get("EXPORTS_DIR", "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "supersecret")


def ensure_schema() -> None:
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cctv (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                rtsp_url TEXT,
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
        conn.commit()
    except Exception as e:
        app.logger.warning(f"ensure_schema() warning: {e}")
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass


def save_event(camera_id: int, description: str = "Orang meninggalkan ruangan") -> None:
    """
    Simpan satu baris event ke tabel person_exit.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO person_exit (camera_id, exit_time, description) VALUES (%s, NOW(), %s)",
            (camera_id, description),
        )
        conn.commit()
    except Exception as e:
        app.logger.error(f"Gagal save_event cam {camera_id}: {e}")
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass


def get_recent_events(limit: int = 15) -> List[Dict[str, Any]]:
    """
    Ambil event terbaru join dengan nama kamera.
    Return list of dicts: camera_name, exit_time (datetime), description
    """
    rows: List[Dict[str, Any]] = []
    try:
        conn = get_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            f"""
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
    """
    Export event ke Excel (xlsx). Mengembalikan path file.
    """
    try:
        import pandas as pd
    except Exception as e:
        app.logger.error(f"Pandas tidak tersedia untuk export: {e}")
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

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    app.logger.error(f"Gagal load YOLO model di {MODEL_PATH}: {e}")
    model = None

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
tracked_people: Dict[int, Any] = {}
last_people_count: Dict[int, int] = {}  

try:
    ensure_schema()
except Exception as _:
    app.logger.warning("ensure_schema() gagal/skip; lanjut jalan.")

def init_camera_data(cams: List[Dict[str, Any]]) -> None:
    for cam in cams:
        cid = cam["id"]
        latest_frame.setdefault(cid, None)
        annotated_frame.setdefault(cid, None)
        people_count.setdefault(cid, 0)
        writer_info.setdefault(
            cid, {"fps": TARGET_RECORD_FPS, "size": None, "filename": None}
        )
        recording_status.setdefault(cid, False)
        stop_flags.setdefault(cid, False)
        recording_locks.setdefault(cid, threading.Lock())
        thread_locks.setdefault(cid, threading.Lock())
        last_people_count.setdefault(cid, 0)

def init_writer_if_needed(cam_id: int, frame, fps=TARGET_RECORD_FPS):
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

def capture_thread_fn(cam_id: int, rtsp_url: str):
    cap = None
    backoff = RTSP_OPEN_RETRY_SECONDS
    app.logger.info(f"Starting capture thread for camera {cam_id}")

    try:
        while not stop_flags.get(cam_id, False):
            try:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass

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
                grabbed = cap.grab()
                if not grabbed:
                    raise RuntimeError("Frame grab failed")
                ret, frame = cap.retrieve()
                if not ret:
                    raise RuntimeError("Frame retrieve failed")

                with frames_lock:
                    latest_frame[cam_id] = frame
                    if writer_info.get(cam_id, {}).get("size") is None:
                        writer_info[cam_id]["size"] = (frame.shape[1], frame.shape[0])
                time.sleep(0.001)

            except Exception as e:
                app.logger.error(f"Error capture camera {cam_id}: {str(e)}")
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 30.0)

    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        app.logger.info(f"Capture thread stopped for camera {cam_id}")

def process_detection(frame, cam_id: int, resize_for_det: Tuple[int, int] = (1280, 720)):
    try:
        if frame is None or model is None:
            return frame, 0
        small = cv2.resize(frame, resize_for_det)
        results = model.predict(
            small,
            classes=[0],
            verbose=False,
            conf=0.1,   
            iou=0.2,    
            max_det=1000
        )

        annotated = frame.copy()
        count = 0

        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                try:
                    cls_id = int(box.cls.item())
                except Exception:
                    cls_id = 0
                if cls_id != 0:
                    continue

                count += 1
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                fw, fh = frame.shape[1], frame.shape[0]
                rw, rh = resize_for_det
                scale_x = fw / rw
                scale_y = fh / rh
                ox1 = int(max(0, min(fw, x1 * scale_x)))
                oy1 = int(max(0, min(fh, y1 * scale_y)))
                ox2 = int(max(0, min(fw, x2 * scale_x)))
                oy2 = int(max(0, min(fh, y2 * scale_y)))
                cv2.rectangle(annotated, (ox1, oy1), (ox2, oy2), (0, 255, 0), 6)
        cv2.putText(
            annotated, f"People: {count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
        )

        return annotated, count

    except Exception as e:
        app.logger.error(f"Error in process_detection camera {cam_id}: {str(e)}")
        return frame, 0

def detect_and_record_thread_fn(cam_id: int):
    app.logger.info(f"Starting detection/record thread for camera {cam_id}")
    last_people_count.setdefault(cam_id, 0)
    fps_asli = TARGET_RECORD_FPS or 25.0
    fps_slow = fps_asli / max(6.0, SLOW_FACTOR)
    base_interval = 1.0 / fps_slow
    next_write_ts = time.monotonic() + base_interval
    counter = 0
    try:
        while not stop_flags.get(cam_id, False):
            with frames_lock:
                frame = latest_frame.get(cam_id)
            if frame is None:
                time.sleep(0.01)
                continue
            run_det = (counter % max(1, DETECTION_SKIP) == 0)
            if run_det:
                annotated, current_count = process_detection(frame, cam_id, DETECTION_SIZE)
                try:
                    if last_people_count[cam_id] > 0 and current_count == 0:
                        app.logger.info(f"[EVENT] Camera {cam_id}: Orang meninggalkan kamera")
                        try:
                            save_event(cam_id, "Orang meninggalkan ruangan")
                        except Exception as e:
                            app.logger.error(f"Gagal simpan event cam {cam_id}: {e}")
                        try:
                            export_path = os.path.join(EXPORTS_DIR, "events.xlsx")
                            export_events_to_excel(export_path)
                            app.logger.info(f"Events diexport ke {export_path}")
                        except Exception as e:
                            app.logger.error(f"Gagal export Excel: {e}")
                finally:
                    last_people_count[cam_id] = current_count

                with frames_lock:
                    annotated_frame[cam_id] = annotated
                    people_count[cam_id] = current_count
            else:
                with frames_lock:
                    annotated = annotated_frame.get(cam_id, frame)

            if recording_status.get(cam_id, False):
                with recording_locks.setdefault(cam_id, threading.Lock()):
                    if writers.get(cam_id) is None:
                        init_writer_if_needed(cam_id, annotated, fps=fps_slow)

                    vw = writers.get(cam_id)
                    if vw is not None:
                        now = time.monotonic()
                        if now >= next_write_ts:
                            try:
                                vw.write(annotated)
                            except Exception as e:
                                app.logger.error(f"Write failed camera {cam_id}: {str(e)}")
                                close_writer(cam_id)

                            next_write_ts += base_interval
                            if now - next_write_ts > 2 * base_interval:
                                next_write_ts = now + base_interval
                        else:
                            time.sleep(min(0.003, max(0.0, next_write_ts - now)))
            else:
                with recording_locks.setdefault(cam_id, threading.Lock()):
                    close_writer(cam_id)
                next_write_ts = time.monotonic() + base_interval
            counter += 1
            time.sleep(0.001)
    except Exception as e:
        app.logger.error(f"Fatal error in detection thread {cam_id}: {str(e)}")
    finally:
        with recording_locks.setdefault(cam_id, threading.Lock()):
            close_writer(cam_id)
        app.logger.info(f"Detection thread {cam_id} stopped")

def generate_stream(cam_id: int):
    while True:
        with frames_lock:
            ann = annotated_frame.get(cam_id)
            raw = latest_frame.get(cam_id)
        frame = ann if ann is not None else raw
        if frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder, "NO SIGNAL", (150, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3
            )
            ok, buf = cv2.imencode('.jpg', placeholder)
            if not ok:
                time.sleep(0.05)
                continue
            jpg = buf.tobytes()
        else:
            ok, buf = cv2.imencode('.jpg', frame)
            if not ok:
                time.sleep(0.05)
                continue
            jpg = buf.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        time.sleep(0.03)

# Thread Manager
def start_camera_threads():
    cams = load_cameras_from_db()
    active_cams = [c for c in cams if c.get("is_active", True)]
    init_camera_data(active_cams)

    for cam in active_cams:
        cid = cam["id"]
        rtsp = cam.get("rtsp_url") or cam.get("url") or cam.get("rtsp") or ""
        stop_flags[cid] = False
        tcap = capture_threads.get(cid)
        if tcap is None or not tcap.is_alive():
            tcap = threading.Thread(target=capture_thread_fn, args=(cid, rtsp), daemon=True)
            capture_threads[cid] = tcap
            tcap.start()
        tdet = detect_threads.get(cid)
        if tdet is None or not tdet.is_alive():
            tdet = threading.Thread(target=detect_and_record_thread_fn, args=(cid,), daemon=True)
            detect_threads[cid] = tdet
            tdet.start()

    # bersihkan thread kamera yang dihapus
    all_cam_ids = {cam["id"] for cam in cams}
    for cid in list(capture_threads.keys()):
        if cid not in all_cam_ids:
            stop_flags[cid] = True
            if cid in capture_threads:
                del capture_threads[cid]
            if cid in detect_threads:
                del detect_threads[cid]


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


# Routes
@app.route("/")
def index():
    cams = load_cameras_from_db()
    start_camera_threads()
    # ambil recent events untuk ditampilkan sebagai notifikasi/log di index.html
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


# route download excel
@app.route("/download_events")
def download_events():
    try:
        export_path = os.path.join(EXPORTS_DIR, "events.xlsx")
        export_events_to_excel(export_path)  # refresh sebelum download
        return send_file(export_path, as_attachment=True)
    except Exception as e:
        app.logger.error(f"Gagal download events: {e}")
        return "Gagal download events", 500


@app.route("/toggle_record/<int:camera_id>", methods=["POST"])
def toggle_record(camera_id: int):
    recording_locks.setdefault(camera_id, threading.Lock())

    # FIX: hilangkan typo cam_idera_id
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
    # Hentikan thread dan rekaman dulu
    stop_flags[camera_id] = True
    recording_status[camera_id] = False
    with recording_locks.setdefault(camera_id, threading.Lock()):
        close_writer(camera_id)

    # Hapus data kamera dari database
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
        # Pastikan exit_time berupa string agar jsonify tidak error
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

if __name__ == "__main__":
    start_camera_threads()
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
