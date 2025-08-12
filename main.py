import os
import time
import threading
from datetime import datetime
from typing import Tuple
import numpy as np

import cv2
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
DETECTION_SKIP = int(os.environ.get("DETECTION_SKIP", "3"))
DEFAULT_RECORD_FPS = float(os.environ.get("DEFAULT_RECORD_FPS", "15.0"))
RTSP_OPEN_RETRY_SECONDS = float(os.environ.get("RTSP_RETRY", "2.0"))
MAX_MEASURE_FPS = 30.0
MIN_MEASURE_FPS = 5.0
os.makedirs(RECORDINGS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "supersecret")

# Load YOLO model
model = YOLO(MODEL_PATH)

# Global states
capture_threads = {}
detect_threads = {}
stop_flags = {}
thread_locks = {}

latest_frame = {}
annotated_frame = {}
people_count = {}
frames_lock = threading.Lock()
recording_locks = {}
writers = {}
writer_info = {}
recording_status = {}
DEFAULT_RECORD_FPS = 30.0 


def init_camera_data(cams):
    for cam in cams:
        cid = cam["id"]
        latest_frame.setdefault(cid, None)
        annotated_frame.setdefault(cid, None)
        people_count.setdefault(cid, 0)
        writer_info.setdefault(cid, {"fps": DEFAULT_RECORD_FPS, "size": None, "filename": None})
        recording_status.setdefault(cid, False)
        stop_flags.setdefault(cid, False)
        recording_locks.setdefault(cid, threading.Lock())
        thread_locks.setdefault(cid, threading.Lock())

def measure_fps(cap: cv2.VideoCapture, sample_frames: int = 20, timeout: float = 3.0) -> float:
    start = time.time()
    count = 0
    for _ in range(sample_frames):
        if time.time() - start > timeout:
            break
        try:
            ok = cap.grab()
        except Exception:
            ok = False
        if not ok:
            break
        count += 1
    elapsed = time.time() - start
    if elapsed > 0 and count > 0:
        fps = count / elapsed
        fps = max(MIN_MEASURE_FPS, min(fps, MAX_MEASURE_FPS))
        return fps
    return DEFAULT_RECORD_FPS

def init_writer_if_needed(cam_id: int, frame):
    if writers.get(cam_id) is None and frame is not None:
        size = (frame.shape[1], frame.shape[0])  
        fps = DEFAULT_RECORD_FPS
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        filename = os.path.join(RECORDINGS_DIR, f"cam{cam_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        vw = cv2.VideoWriter(filename, fourcc, fps, size)
        if not vw.isOpened():
            app.logger.error(f"VideoWriter gagal dibuka untuk kamera {cam_id}")
            writers[cam_id] = None
        else:
            writers[cam_id] = vw
            writer_info[cam_id] = {"filename": filename, "fps": fps, "size": size}
            app.logger.info(f"Rekaman mulai kamera {cam_id} ke {filename}")

def close_writer(cam_id: int):
    w = writers.get(cam_id)
    if w is not None:
        w.release()
        app.logger.info(f"Rekaman dihentikan kamera {cam_id}")
    writers[cam_id] = None
    writer_info.pop(cam_id, None)


def capture_thread_fn(cam_id: int, rtsp_url: str):
    cap = None
    backoff = RTSP_OPEN_RETRY_SECONDS
    app.logger.info(f"Starting capture thread for camera {cam_id}")
    try:
        while not stop_flags.get(cam_id, False):
            try:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(rtsp_url)
                    if not cap.isOpened():
                        app.logger.warning(f"Camera {cam_id} cannot open, retry in {backoff}s...")
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, 30.0)
                        continue
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                    except Exception:
                        pass
                    measured = measure_fps(cap)
                    with thread_locks[cam_id]:
                        writer_info.setdefault(cam_id, {})
                        writer_info[cam_id]["fps"] = measured
                    app.logger.info(f"Camera {cam_id} opened, measured FPS: {measured:.1f}")
                    backoff = RTSP_OPEN_RETRY_SECONDS

                grabbed = cap.grab()
                if not grabbed:
                    raise RuntimeError("Frame grab failed")
                ret, frame = cap.retrieve()
                if not ret:
                    raise RuntimeError("Frame retrieve failed")

                with frames_lock:
                    latest_frame[cam_id] = frame.copy()
                    if writer_info.get(cam_id, {}).get("size") is None:
                        writer_info[cam_id]["size"] = (frame.shape[1], frame.shape[0])

                time.sleep(1 / max(writer_info[cam_id]["fps"], 5.0))

            except Exception as e:
                app.logger.error(f"Error in capture thread {cam_id}: {str(e)}")
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
                cap = None
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 30.0)
    except Exception as e:
        app.logger.error(f"Fatal error in capture thread {cam_id}: {str(e)}")
    finally:
        try:
            if cap is not None and cap.isOpened():
                cap.release()
        except Exception:
            pass
        app.logger.info(f"Capture thread {cam_id} stopped")

def scale_box(xy, frame_shape: Tuple[int, int], resize_shape: Tuple[int, int]):
    fw, fh = frame_shape[1], frame_shape[0]
    rw, rh = resize_shape
    scale_x = fw / rw
    scale_y = fh / rh
    x1 = int(max(0, min(fw, xy[0] * scale_x)))
    y1 = int(max(0, min(fh, xy[1] * scale_y)))
    x2 = int(max(0, min(fw, xy[2] * scale_x)))
    y2 = int(max(0, min(fh, xy[3] * scale_y)))
    return x1, y1, x2, y2
def process_detection(frame, cam_id: int, resize_for_det: Tuple[int, int] = (640, 360)):
    try:
        if frame is None:
            return None, 0
        
        small = cv2.resize(frame, resize_for_det)
        results = model.track(small, persist=True, classes=[0], verbose=False)

        annotated = frame.copy()
        count = 0
        
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                cls_id = box.cls.item()
                conf = box.conf.item()
                # DEBUG print
                print(f"Detected class {cls_id} with confidence {conf:.2f}")
                if cls_id != 0:
                    continue
                
                # Hitung orang terdeteksi
                count += 1
                
                # Dapatkan koordinat xyxy (float)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Scale kotak dari ukuran resize ke ukuran asli frame
                fw, fh = frame.shape[1], frame.shape[0]
                rw, rh = resize_for_det
                scale_x = fw / rw
                scale_y = fh / rh
                
                orig_x1 = int(max(0, min(fw, x1 * scale_x)))
                orig_y1 = int(max(0, min(fh, y1 * scale_y)))
                orig_x2 = int(max(0, min(fw, x2 * scale_x)))
                orig_y2 = int(max(0, min(fh, y2 * scale_y)))
                
                # Gambar bounding box hijau
                cv2.rectangle(annotated, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 2)
                
                # Label confidence
                cv2.putText(
                    annotated,
                    f"Person {conf:.2f}",
                    (orig_x1, max(16, orig_y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Overlay total orang terdeteksi di pojok kiri atas
        cv2.putText(
            annotated,
            f"People: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        return annotated, count

    except Exception as e:
        app.logger.error(f"Error in process_detection camera {cam_id}: {str(e)}")
        return frame, 0
    
def detect_and_record_thread_fn(cam_id: int):
    counter = 0
    resize_for_det = (640, 360)
    app.logger.info(f"Starting detection thread for camera {cam_id}")
    try:
        while not stop_flags.get(cam_id, False):
            with frames_lock:
                frame = latest_frame.get(cam_id)
            
            if frame is None:
                time.sleep(0.1)
                continue

            counter += 1
            annotated = None
            current_count = 0
            
            if (counter % DETECTION_SKIP) == 0:
                annotated, current_count = process_detection(frame, cam_id, resize_for_det)
                app.logger.info(f"[Detect Thread] Camera {cam_id}: detected {current_count} people")
                with frames_lock:
                    annotated_frame[cam_id] = annotated
                    people_count[cam_id] = current_count
            else:
                with frames_lock:
                    annotated = annotated_frame.get(cam_id)
                    current_count = people_count.get(cam_id, 0)
                if annotated is None:
                    annotated = frame.copy()

            # Rekaman video
            if recording_status.get(cam_id, False):
                with recording_locks.setdefault(cam_id, threading.Lock()):
                    if writers.get(cam_id) is None:
                        init_writer_if_needed(cam_id, annotated)
                    if writers.get(cam_id) is not None:
                        try:
                            writers[cam_id].write(annotated)
                        except Exception as e:
                            app.logger.error(f"Write failed camera {cam_id}: {str(e)}")
                            close_writer(cam_id)
            else:
                # Jika rekaman off, pastikan writer ditutup
                with recording_locks.setdefault(cam_id, threading.Lock()):
                    close_writer(cam_id)

            fps = float(writer_info.get(cam_id, {}).get("fps") or DEFAULT_RECORD_FPS)
            delay = max(0.001, 1.0 / max(5.0, min(fps, 30.0)))
            time.sleep(delay)
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
            cv2.putText(placeholder, "NO SIGNAL", (150, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            _, buf = cv2.imencode('.jpg', placeholder)
            jpg = buf.tobytes()
        else:
            try:
                _, buf = cv2.imencode('.jpg', frame)
                jpg = buf.tobytes()
            except Exception:
                time.sleep(0.05)
                continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
        time.sleep(0.03)

def start_camera_threads():
    cams = load_cameras_from_db()
    active_cams = [c for c in cams if c.get("is_active", True)]
    init_camera_data(active_cams)

    for cam in active_cams:
        cid = cam["id"]
        rtsp = cam.get("rtsp_url") or cam.get("url") or cam.get("rtsp")

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
            
    all_cam_ids = {cam["id"] for cam in cams}
    for cid in list(capture_threads.keys()):
        if cid not in all_cam_ids:
            stop_flags[cid] = True
            del capture_threads[cid]
            del detect_threads[cid]

@app.route("/")
def index():
    cams = load_cameras_from_db()
    start_camera_threads()
    return render_template(
        "index.html",
        cameras=cams,
        recording_status=recording_status,
        people_count=people_count,
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



@app.route("/toggle_record/<int:camera_id>", methods=["POST"])
def toggle_record(camera_id: int):
    # Pastikan lock tersedia
    recording_locks.setdefault(camera_id, threading.Lock())
    
    # Toggle status rekaman
    new_status = not recording_status.get(camera_id, False)
    recording_status[camera_id] = new_status
    app.logger.info(f"Toggle record cam {camera_id} -> {new_status}")

    # Jika dimatikan, segera tutup writer
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

@app.route("/add_camera", methods=["GET", "POST"])
def add_camera():
    if request.method == "POST":
        name = request.form.get("name")
        rtsp_url = request.form.get("rtsp_url")
        
        if not name or not rtsp_url:
            flash("Name and RTSP URL are required", "error")
            return redirect(url_for("add_camera"))

        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO cctv (name, rtsp_url, is_active) VALUES (%s, %s, TRUE)",
                (name, rtsp_url),
            )
            conn.commit()
            new_id = cursor.lastrowid
            flash(f"Camera added successfully (ID: {new_id})", "success")
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
