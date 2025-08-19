import cv2
import numpy as np
import threading

# Daftar kamera
cams = {
    "Kamera 1": "rtsp://admin:kutaitimber121@192.168.1.64:554/Streaming/Channels/101",
    "Kamera 2": "rtsp://admin:kutaitimber121@192.168.1.63:554/Streaming/Channels/101",
}

# Dictionary untuk simpan zona tiap kamera
zones = {name: [] for name in cams.keys()}

def select_zone(name, url):
    cap = cv2.VideoCapture(url)
    points = []

    def get_coords(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"[{name}] Klik di koordinat: ({x}, {y})")
            points.append((x, y))

    cv2.namedWindow(name)
    cv2.setMouseCallback(name, get_coords)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Gagal baca stream {name}")
            break

        # gambar titik
        for p in points:
            cv2.circle(frame, p, 5, (0, 0, 255), -1)

        # gambar polygon
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow(name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # keluar dari kamera ini
            zones[name] = points.copy()
            break

    cap.release()
    cv2.destroyWindow(name)

for cam_name, cam_url in cams.items():
    print(f"--- Klik zona untuk {cam_name} (tekan 'q' kalau sudah selesai) ---")
    select_zone(cam_name, cam_url)

print("\nHasil zona untuk semua kamera:")
for cam, pts in zones.items():
    print(f"{cam}: {pts}")
