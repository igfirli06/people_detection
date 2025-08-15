import cv2
import time
import csv
from ultralytics import YOLO

# CONFIG
CAMERAS = [
    {"id": 1, "rtsp": "rtsp://admin:kutaitimber121@192.168.1.64:554/Streaming/Channels/101", "log": "cam1_log.csv"},
    {"id": 2, "rtsp": "rtsp://admin:kutaitimber121@192.168.1.63:554/Streaming/Channels/101", "log": "cam2_log.csv"}
]
MODEL_PATH = "validate_yolo"
CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.2
RESIZE_FOR_DET = (1280, 720)

# INIT
caps = [cv2.VideoCapture(cam["rtsp"]) for cam in CAMERAS]
model = YOLO(MODEL_PATH)

# Siapkan CSV untuk log
for cam in CAMERAS:
    with open(cam["log"], mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "people_count"])

# LOOP DETEKSI
while True:
    frames = []
    counts = []

    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {CAMERAS[i]['id']} tidak tersedia")
            frames.append(None)
            counts.append(0)
            continue

        small_frame = cv2.resize(frame, RESIZE_FOR_DET)
        results = model.predict(
            small_frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=[0],
            verbose=False,
            max_det=1000
        )

        count = 0
        annotated_frame = frame.copy()

        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                if cls_id != 0:
                    continue
                count += 1
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                fw, fh = frame.shape[1], frame.shape[0]
                rw, rh = RESIZE_FOR_DET
                scale_x = fw / rw
                scale_y = fh / rh
                ox1 = int(max(0, min(fw, x1 * scale_x)))
                oy1 = int(max(0, min(fh, y1 * scale_y)))
                ox2 = int(max(0, min(fw, x2 * scale_x)))
                oy2 = int(max(0, min(fh, y2 * scale_y)))

                cv2.rectangle(annotated_frame, (ox1, oy1), (ox2, oy2), (0, 255, 0), 4)
                cv2.putText(
                    annotated_frame,
                    f"Person {conf:.2f}",
                    (ox1, max(16, oy1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        cv2.putText(
            annotated_frame,
            f"People: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

        frames.append(annotated_frame)
        counts.append(count)

        # Simpan log
        with open(CAMERAS[i]["log"], mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), count])

    # Tampilkan kedua kamera di window terpisah
    for i, frame in enumerate(frames):
        if frame is not None:
            cv2.imshow(f"Camera {CAMERAS[i]['id']}", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
print("Log tersimpan untuk kedua kamera")
