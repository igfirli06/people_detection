import pytest
import numpy as np
import threading

# Import fungsi dari main.py (pastikan main.py ada di folder yang sama)
from main import (
    process_detection,
    init_camera_data,
    latest_frame,
    annotated_frame,
    frames_lock,
    generate_stream,
    DETECTION_SIZE
)

# fixture kamera dummy
@pytest.fixture
def dummy_camera():
    cam_id = 1
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with frames_lock:
        latest_frame[cam_id] = frame
        annotated_frame[cam_id] = None
    init_camera_data([{"id": cam_id}])
    return cam_id, frame

# Test process_detection
def test_process_detection_on_blank_frame(dummy_camera):
    cam_id, frame = dummy_camera
    annotated, count = process_detection(frame, cam_id, DETECTION_SIZE)
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == frame.shape
    # Blank frame seharusnya tidak ada orang
    assert count == 0

# Test generate_stream
def test_generate_stream_output(dummy_camera):
    cam_id, frame = dummy_camera

    # Panggil generator
    gen = generate_stream(cam_id)
    frame_bytes = next(gen)  # ambil satu frame dari stream

    assert isinstance(frame_bytes, bytes)
    assert b'Content-Type: image/jpeg' in frame_bytes

# Test thread-safe update frame
def test_thread_safe_frame_update(dummy_camera):
    cam_id, frame = dummy_camera

    def update_frame():
        for _ in range(5):
            with frames_lock:
                latest_frame[cam_id] = np.ones_like(frame)
    
    threads = [threading.Thread(target=update_frame) for _ in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()

    with frames_lock:
        assert latest_frame[cam_id] is not None
        assert latest_frame[cam_id].shape == frame.shape
