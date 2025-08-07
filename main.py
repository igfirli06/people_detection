import cv2
import time
import os
from flask import Flask, Response, render_template
from ultralytics import YOLO
import threading
from datetime import datetime

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Folder penyimpanan rekaman
recording_path = 'recordings'
os.makedirs(recording_path, exist_ok=True)

# Daftar kamera RTSP
cameras = [
    {
        "name": "Kamera 1",
        "rtsp": "rtsp://admin:kutaitimber121@192.168.1.64:554/Streaming/Channels/101",
        "cap": None,
        "latest_frame": None,
        "lock": threading.Lock()
    },
    {
        "name": "Kamera 2",
        "rtsp": "rtsp://admin:kutaitimber121@192.168.1.63:554/Streaming/Channels/101",
        "cap": None,
        "latest_frame": None,
        "lock": threading.Lock()
    }
]

def detect_people(camera):
    cap = cv2.VideoCapture(camera["rtsp"])
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka kamera: {camera['name']}")
        return

    camera["cap"] = cap
    frame_skip = 5
    frame_counter = 0

    # Setup video writer pertama
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(recording_path, f"{camera['name'].replace(' ', '_')}_{timestamp}.avi")
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 360))

    while True:
        success, frame = cap.read()
        if not success:
            print(f"[WARNING] Gagal membaca dari {camera['name']}, mencoba lagi...")
            time.sleep(1)
            continue

        frame_counter += 1
        if frame_counter % frame_skip != 0:
            time.sleep(0.01)
            continue

        # Resize & deteksi
        frame_resized = cv2.resize(frame, (640, 360))
        results = model(frame_resized, verbose=False)
        people = [box for box in results[0].boxes if int(box.cls[0]) == 0]
        current_count = len(people)

        # Gambar bounding box
        for box in people:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Tambahkan teks jumlah orang
        cv2.putText(frame_resized, f"People: {current_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Simpan ke video
        out.write(frame_resized)

        # Simpan frame untuk streaming
        with camera["lock"]:
            camera["latest_frame"] = frame_resized.copy()

        # Cek apakah sudah 1 menit, buat file baru
        if time.time() - start_time >= 60:
            out.release()
            start_time = time.time()
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = os.path.join(recording_path, f"{camera['name'].replace(' ', '_')}_{timestamp}.avi")
            out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 360))

        time.sleep(0.01)  # agar CPU tidak penuh

def gen_frames(camera_id):
    camera = cameras[camera_id]
    while True:
        with camera["lock"]:
            if camera["latest_frame"] is None:
                time.sleep(0.05)
                continue
            ret, buffer = cv2.imencode('.jpg', camera["latest_frame"])
            frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)

@app.route('/')
def index():
    return render_template('index.html', cameras=cameras)

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    return Response(gen_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    for camera in cameras:
        t = threading.Thread(target=detect_people, args=(camera,))
        t.daemon = True
        t.start()
    time.sleep(2)
    app.run(host='0.0.0.0', port=5000, debug=False)
