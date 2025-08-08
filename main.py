import cv2
import time
import threading
import os
from datetime import datetime
from flask import Flask, Response, render_template, request, redirect, url_for
from ultralytics import YOLO
from database import get_connection

app = Flask(__name__)
model = YOLO("yolov8n.pt")

camera_streams = {}
recording_status = {}   # status rekaman ON/OFF
video_writers = {}      # object VideoWriter
record_start_time = {}  # waktu mulai rekam

# folder simpan rekaman
if not os.path.exists("recordings"):
    os.makedirs("recordings")

def ensure_table_exists():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cctv (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        rtsp_url TEXT NOT NULL,
        is_active TINYINT(1) DEFAULT 1
    )
    """)
    conn.commit()
    cursor.close()
    conn.close()

def load_cameras():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM cctv WHERE is_active = 1")
    cameras = cursor.fetchall()
    cursor.close()
    conn.close()
    # pastikan setiap kamera punya status default
    for cam in cameras:
        recording_status.setdefault(cam['id'], False)
        video_writers.setdefault(cam['id'], None)
        record_start_time.setdefault(cam['id'], None)
    return cameras

def draw_thick_text(img, text, org, font, font_scale, color, thickness, line_type=cv2.LINE_AA):
    cv2.putText(img, text, org, font, font_scale, (0, 0, 0), thickness + 2, line_type)
    cv2.putText(img, text, org, font, font_scale, color, thickness, line_type)

def generate_frames(rtsp_url, cam_id):
    while True:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            try:
                cap.release()
            except:
                pass
            time.sleep(3)
            continue

        while True:
            success, frame = cap.read()
            if not success or frame is None:
                break

            try:
                results = model(frame)
            except:
                draw_thick_text(frame, "Model error", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue

            person_count = 0
            for r in results:
                boxes = getattr(r, 'boxes', [])
                for box in boxes:
                    try:
                        cls = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
                        label = model.names[cls]
                    except:
                        label = None

                    if label == "person":
                        xy = box.xyxy[0] if hasattr(box, 'xyxy') else None
                        if xy is not None:
                            x1, y1, x2, y2 = map(int, xy)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 7)  # tebal
                            draw_thick_text(frame, "Person", (x1, max(20, y1 - 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                            person_count += 1

            draw_thick_text(frame, f"People: {person_count}", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            # ==== Rekaman ====
            if recording_status[cam_id]:
                if video_writers[cam_id] is None:
                    filename = f"recordings/cam{cam_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    h, w = frame.shape[:2]
                    video_writers[cam_id] = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
                    record_start_time[cam_id] = time.time()

                video_writers[cam_id].write(frame)

                # stop otomatis setelah 60 detik
                if time.time() - record_start_time[cam_id] >= 60:
                    video_writers[cam_id].release()
                    video_writers[cam_id] = None
                    recording_status[cam_id] = False
                    record_start_time[cam_id] = None

            elif video_writers[cam_id] is not None:
                video_writers[cam_id].release()
                video_writers[cam_id] = None

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        try:
            cap.release()
        except:
            pass
        time.sleep(3)

@app.route('/')
def index():
    cameras = load_cameras()
    return render_template('index.html', cameras=cameras, recording_status=recording_status)

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    cameras = load_cameras()
    for cam in cameras:
        if cam['id'] == camera_id:
            return Response(generate_frames(cam['rtsp_url'], cam['id']),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
    return "Camera not found", 404

@app.route('/toggle_record/<int:camera_id>', methods=['POST'])
def toggle_record(camera_id):
    recording_status[camera_id] = not recording_status.get(camera_id, False)
    return redirect(url_for('index'))

@app.route('/add_camera', methods=['GET', 'POST'])
def add_camera():
    if request.method == 'POST':
        name = request.form['name']
        rtsp_url = request.form['rtsp_url']
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO cctv (name, rtsp_url, is_active) VALUES (%s, %s, 1)",
                       (name, rtsp_url))
        conn.commit()
        cursor.close()
        conn.close()
        return redirect(url_for('index'))
    return render_template('add_camera.html')

@app.route('/deactivate/<int:camera_id>', methods=['POST'])
def deactivate(camera_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE cctv SET is_active = 0 WHERE id = %s", (camera_id,))
    conn.commit()
    cursor.close()
    conn.close()
    return redirect(url_for('index'))

if __name__ == "__main__":
    ensure_table_exists()
    app.run(host="0.0.0.0", port=5000, debug=True)
