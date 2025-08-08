# app.py
import cv2
import time
import os
import threading
import pymysql
from flask import Flask, Response, render_template, request, redirect, url_for, flash
from ultralytics import YOLO
from datetime import datetime

# ===== CONFIG =====
DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',      
    'db': 'people_detection',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

RECORDING_FOLDER = 'recordings'
MODEL_PATH = 'yolov8n.pt'
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
FPS = 20.0
FRAME_SKIP = 5

os.makedirs(RECORDING_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'ganti'


model = YOLO(MODEL_PATH)


cameras = {}
cameras_lock = threading.Lock()


def get_active_cameras_from_db():
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, rtsp_url, is_active FROM cctv WHERE is_active=1")
            rows = cur.fetchall()
        return rows
    finally:
        conn.close()

def insert_camera_to_db(name, rtsp_url, is_active=1):
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO cctv (name, rtsp_url, is_active) VALUES (%s, %s, %s)",
                        (name, rtsp_url, is_active))
            conn.commit()
            return cur.lastrowid
    finally:
        conn.close()


def start_camera_thread(cam):
    t = threading.Thread(target=detect_people, args=(cam,), daemon=True)
    cam['thread'] = t
    t.start()

def detect_people(cam):
    rtsp = cam['rtsp']
    name = cam['name']
    safe_name = name.replace(' ', '_')
    print(f"[INFO] Starting detection for camera {name} (id={cam['id']})")

    cap = cv2.VideoCapture(rtsp)
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka RTSP: {rtsp} for {name}")
    
        while not cap.isOpened():
            time.sleep(3)
            cap = cv2.VideoCapture(rtsp)
            if cap.isOpened():
                print(f"[INFO] Reconnected to {name}")
                break

    cam['cap'] = cap
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(RECORDING_FOLDER, f"{safe_name}_{timestamp}.avi")
    out = cv2.VideoWriter(filename, fourcc, FPS, (cam.get('width', FRAME_WIDTH), cam.get('height', FRAME_HEIGHT)))

    frame_counter = 0
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print(f"[WARNING] Tidak dapat membaca frame dari {name}, mencoba reconnect...")
            cap.release()
            cam['cap'] = None
            time.sleep(2)
            cap = cv2.VideoCapture(rtsp)
            if not cap.isOpened():
                time.sleep(3)
                continue
            else:
                cam['cap'] = cap
                continue

      
        frame_resized = cv2.resize(frame, (cam.get('width', FRAME_WIDTH), cam.get('height', FRAME_HEIGHT)))
        frame_counter += 1

        with cam['lock']:
            cam['latest_frame'] = frame_resized.copy()

        if frame_counter % FRAME_SKIP != 0:
            time.sleep(0.005)
            continue

        results = model(frame_resized, verbose=False)
        people = [box for box in results[0].boxes if int(box.cls[0]) == 0]
        current_count = len(people)

       
        for box in people:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_resized, f"People: {current_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        out.write(frame_resized)
        with cam['lock']:
            cam['latest_frame'] = frame_resized.copy()

        if time.time() - start_time >= 60:
            out.release()
            start_time = time.time()
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = os.path.join(RECORDING_FOLDER, f"{safe_name}_{timestamp}.avi")
            out = cv2.VideoWriter(filename, fourcc, FPS, (cam.get('width', FRAME_WIDTH), cam.get('height', FRAME_HEIGHT)))

        time.sleep(0.005)


def gen_frames(camera_id):
    cam_id = int(camera_id)
    while True:
        with cameras_lock:
            cam = cameras.get(cam_id)
        if cam is None:
            time.sleep(0.2)
            continue
        with cam['lock']:
            frame = cam.get('latest_frame')
            if frame is None:
                time.sleep(0.05)
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                time.sleep(0.02)
                continue
            img_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')
        time.sleep(0.01)

@app.route('/')
def index():
    with cameras_lock:
        cams = [{'id': cam_id, 'name': cam['name']} for cam_id, cam in cameras.items()]
    return render_template('index.html', cameras=cams)

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    return Response(gen_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_camera', methods=['GET', 'POST'])
def add_camera():
    if request.method == 'POST':
        name = request.form.get('name')
        rtsp = request.form.get('rtsp')
        if not name or not rtsp:
            flash('Nama dan RTSP wajib diisi', 'danger')
            return redirect(url_for('add_camera'))
        cam_id = insert_camera_to_db(name, rtsp, 1)
        cam = {
            'id': cam_id,
            'name': name,
            'rtsp': rtsp,
            'cap': None,
            'latest_frame': None,
            'lock': threading.Lock(),
            'width': FRAME_WIDTH,
            'height': FRAME_HEIGHT,
            'fps': FPS,
            'thread': None
        }
        with cameras_lock:
            cameras[cam_id] = cam
        start_camera_thread(cam)
        flash('Kamera berhasil ditambahkan dan deteksi dijalankan', 'success')
        return redirect(url_for('index'))
    return render_template('add_camera.html')


def load_cameras_and_start():
    rows = get_active_cameras_from_db()
    for r in rows:
        cam_id = int(r['id'])
        cam = {
            'id': cam_id,
            'name': r['name'],
            'rtsp': r['rtsp_url'],
            'cap': None,
            'latest_frame': None,
            'lock': threading.Lock(),
            'width': FRAME_WIDTH,
            'height': FRAME_HEIGHT,
            'fps': FPS,
            'thread': None
        }
        with cameras_lock:
            cameras[cam_id] = cam
        start_camera_thread(cam)

if __name__ == '__main__':
    load_cameras_and_start()
    app.run(host='0.0.0.0', port=5000, debug=False)
