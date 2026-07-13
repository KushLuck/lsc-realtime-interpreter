# test_fps.py
import time
import cv2
from mediapipe.python.solutions.holistic import Holistic
from src.features.mediapipe_holistic import mediapipe_detection
from src.features.feature_schema import CAMERA_WIDTH, CAMERA_HEIGHT

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("No pude abrir la cámara.")

print("Midiendo FPS... haz señas normales. Q para salir.\n")

frame_count = 0
t0 = time.time()
fps = 0.0

try:
    with Holistic(model_complexity=0) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Procesa igual que en captura/inferencia real
            results = mediapipe_detection(frame, holistic)

            frame_count += 1
            elapsed = time.time() - t0

            # Actualiza el promedio cada segundo
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t0 = time.time()

            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )
            cv2.imshow("FPS Test", frame)

            if cv2.waitKey(10) & 0xFF in [ord("q"), ord("Q")]:
                break
finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nÚltimo FPS medido: {fps:.1f}")
