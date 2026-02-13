# src/capture/capture_samples.py
import os
import json
import cv2
import numpy as np
from datetime import datetime

import mediapipe as mp
from mediapipe.python.solutions.holistic import Holistic

from src.features.mediapipe_holistic import (
    mediapipe_detection,
    there_hand,
    extract_v1_pose_hands,
)
from src.features.normalize import resample_sequence
from src.features.feature_schema import (
    MODEL_FRAMES,
    MIN_LENGTH_FRAMES,
    MARGIN_FRAMES,
    DELAY_FRAMES,
)

# MediaPipe drawing helpers (para visualizar landmarks)
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_words(words_json_path: str):
    with open(words_json_path, "r", encoding="utf-8") as f:
        return json.load(f)["word_ids"]


def draw_landmarks_v1(frame, results):
    """
    Dibuja SOLO lo que usamos en v1: pose + manos.
    (No dibuja cara para evitar carga innecesaria.)
    """
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
        )


def capture_word(
    word_id: str,
    out_root: str,
    save_debug_frames: bool = False,
    show_landmarks: bool = False,
):
    """
    Guarda:
      - data/keypoints_v1/<word_id>/<sample_id>.npy  shape=(MODEL_FRAMES, 258)
      - data/metadata/<word_id>/<sample_id>.json
      - (opcional) data/raw_frames/<word_id>/<sample_id>/frame_XXX.jpg

    Controles:
      - Q: salir
      - L: toggle mostrar landmarks (pose+manos)
    """
    kp_dir = os.path.join(out_root, "keypoints_v1", word_id)
    md_dir = os.path.join(out_root, "metadata", word_id)
    rf_dir = os.path.join(out_root, "raw_frames", word_id)

    ensure_dir(kp_dir)
    ensure_dir(md_dir)
    if save_debug_frames:
        ensure_dir(rf_dir)

    count_frame = 0
    fix_frames = 0
    recording = False

    kp_seq = []
    raw_frames = []

    cap = cv2.VideoCapture(0)
    # (opcional) pedir 720p a la cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ventana grande y redimensionable
    WINDOW_NAME = f"LSC Capture | {word_id}"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1100, 700)

# fullscreen (opcional para demo)
# cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if not cap.isOpened():
        raise RuntimeError("No pude abrir la cámara (VideoCapture(0)).")

    with Holistic() as holistic:
        print(f'\n➡️  Listo para capturar: "{word_id}" | Q = salir | L = landmarks on/off\n')
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = mediapipe_detection(frame, holistic)

            if there_hand(results) or recording:
                recording = False
                count_frame += 1
                if count_frame > MARGIN_FRAMES:
                    kp_seq.append(extract_v1_pose_hands(results))
                    if save_debug_frames:
                        raw_frames.append(frame.copy())

                cv2.putText(
                    frame,
                    "CAPTURANDO...",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 80, 255),
                    2,
                )

            else:
                # cierre de seña
                if len(kp_seq) >= (MIN_LENGTH_FRAMES + MARGIN_FRAMES):
                    fix_frames += 1
                    if fix_frames < DELAY_FRAMES:
                        recording = True
                    else:
                        # recortar margen final + delay
                        cut = (MARGIN_FRAMES + DELAY_FRAMES)
                        if cut > 0 and len(kp_seq) > cut:
                            kp_seq = kp_seq[:-cut]
                            if save_debug_frames and len(raw_frames) > cut:
                                raw_frames = raw_frames[:-cut]

                        # guardar muestra
                        sample_id = datetime.now().strftime("%y%m%d%H%M%S%f")
                        kp_arr = np.array(kp_seq, dtype=np.float32)  # (T, 258)
                        kp_arr = resample_sequence(kp_arr, MODEL_FRAMES)  # (MODEL_FRAMES, 258)

                        np.save(os.path.join(kp_dir, f"{sample_id}.npy"), kp_arr)

                        meta = {
                            "word_id": word_id,
                            "sample_id": sample_id,
                            "schema": "v1_pose_hands",
                            "n_features": int(kp_arr.shape[1]),
                            "model_frames": int(MODEL_FRAMES),
                            "captured_frames_raw": int(len(kp_seq)),
                        }
                        with open(os.path.join(md_dir, f"{sample_id}.json"), "w", encoding="utf-8") as f:
                            json.dump(meta, f, ensure_ascii=False, indent=2)

                        if save_debug_frames:
                            sample_frame_dir = os.path.join(rf_dir, sample_id)
                            ensure_dir(sample_frame_dir)
                            for i, fr in enumerate(raw_frames, start=1):
                                cv2.imwrite(
                                    os.path.join(sample_frame_dir, f"frame_{i:03}.jpg"),
                                    fr,
                                )

                        print(
                            f"✅ Guardada muestra {sample_id} | frames_raw={meta['captured_frames_raw']} -> {MODEL_FRAMES}"
                        )

                        # reset
                        kp_seq = []
                        raw_frames = []
                        count_frame = 0
                        fix_frames = 0
                        recording = False

                else:
                    # idle
                    cv2.putText(
                        frame,
                        "LISTO PARA CAPTURAR...",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 180, 0),
                        2,
                    )
                    kp_seq = []
                    raw_frames = []
                    count_frame = 0
                    fix_frames = 0
                    recording = False

            # Overlay de estado
            if show_landmarks:
                draw_landmarks_v1(frame, results)
                cv2.putText(
                    frame,
                    "LANDMARKS: ON (L)",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "LANDMARKS: OFF (L)",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                )

            cv2.imshow(WINDOW_NAME, frame)


            key = cv2.waitKey(10) & 0xFF
            if key in [ord("q"), ord("Q")]:
                break
            if key in [ord("l"), ord("L")]:
                show_landmarks = not show_landmarks

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    words = load_words(os.path.join("models", "words.json"))
    # prueba rápida con una palabra:
    capture_word("adios", out_root="data", save_debug_frames=False, show_landmarks=True)

    print("Palabras:", words)
    print(
        "Ejecuta: python -m src.capture.capture_samples y cambia la palabra en el código o llama capture_word(...)"
    )
