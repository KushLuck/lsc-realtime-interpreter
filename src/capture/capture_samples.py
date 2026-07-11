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
    """Dibuja SOLO lo que usamos en v1: pose + manos (sin cara)."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )


def _reset_state():
    """Estado inicial de captura."""
    return dict(kp_seq=[], raw_frames=[], count_frame=0, fix_frames=0, recording=False)


def capture_word(
    word_id: str,
    out_root: str,
    save_debug_frames: bool = False,
    show_landmarks: bool = True,
):
    """
    Captura muestras de keypoints para una palabra de LSC y las guarda en disco.

    Guarda:
      - data/keypoints_v1/<word_id>/<sample_id>.npy  shape=(MODEL_FRAMES, 258)
      - data/metadata/<word_id>/<sample_id>.json

    Controles:
      - Q: salir
      - L: toggle mostrar landmarks

    Diseno (integra lo probado del repo ronvidev/modelo_lstm_lsp)
    -------------------------------------------------------------
    - Maquina de estados de captura con periodo de gracia estilo original:
      cuando se pierden las manos pero la sena ya es valida, se espera
      DELAY_FRAMES antes de cerrar. Se usa `continue` en la rama de gracia
      (como el original) para saltar al siguiente frame sin cerrar.
    - MEJORA sobre el original: reset de fix_frames cuando la mano reaparece,
      para que solo una ausencia CONTINUA de manos cierre la sena. Esto hace
      la captura robusta al parpadeo de deteccion de MediaPipe.
    - Deteccion estable: model_complexity=1 + min_tracking_confidence=0.3.
      complexity=0 daba mas FPS pero desestabilizaba la deteccion (parpadeo).
      complexity=1 con resolucion 640x480 es el punto dulce para CPU.
    - Normalizacion espacial heredada de extract_v1_pose_hands (invarianza
      a posicion y escala). NO se toca aqui; ocurre dentro de esa funcion.
    - Print de diagnostico al descartar senas cortas.
    - try/finally para liberar camara y ventana siempre.

    IMPORTANTE: los parametros de camara y MediaPipe (resolucion,
    model_complexity, confidences) DEBEN coincidir EXACTAMENTE con live.py,
    o reaparece el gap train-vivo.
    """
    kp_dir = os.path.join(out_root, "keypoints_v1", word_id)
    md_dir = os.path.join(out_root, "metadata", word_id)
    rf_dir = os.path.join(out_root, "raw_frames", word_id)

    ensure_dir(kp_dir)
    ensure_dir(md_dir)
    if save_debug_frames:
        ensure_dir(rf_dir)

    # Verificar camara ANTES de crear ventana o configurar resolucion
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No pude abrir la camara (VideoCapture(0)).")

    # 640x480: mitad de la ganancia de FPS SIN costo de precision
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    WINDOW_NAME = f"LSC Capture | {word_id}"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1100, 700)

    state = _reset_state()

    try:
<<<<<<< HEAD
        # ✅ model_complexity=0 para acelerar (debe coincidir con live.py)
        with Holistic(model_complexity=1) as holistic:
            print(f'\n➡️  Listo para capturar: "{word_id}" | Q = salir | L = landmarks on/off\n')
=======
        # complexity=1 (deteccion estable) + tracking bajo (menos parpadeo).
        # DEBE coincidir con live.py.
        with Holistic(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
        ) as holistic:
            print(f'\n>> Listo para capturar: "{word_id}" | Q = salir | L = landmarks on/off\n')
>>>>>>> 7feb2c932cc126039fe7b9b04d014e6c50ad8c31

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = mediapipe_detection(frame, holistic)
                hand_detected = there_hand(results)

                # ----------------------------------------------------------
                # RAMA A: hay mano O estamos en periodo de gracia
                # ----------------------------------------------------------
                if hand_detected or state["recording"]:
                    # La mano reaparecio: sal de gracia y REINICIA el conteo
                    # de cierre. Asi solo una ausencia CONTINUA cierra la sena.
                    if hand_detected:
                        state["recording"] = False
                        state["fix_frames"] = 0

                    state["count_frame"] += 1
                    if state["count_frame"] > MARGIN_FRAMES:
                        state["kp_seq"].append(extract_v1_pose_hands(results))
                        if save_debug_frames:
                            state["raw_frames"].append(frame.copy())

                    cv2.putText(
                        frame, "CAPTURANDO...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 80, 255), 2,
                    )

                # ----------------------------------------------------------
                # RAMA B: no hay mano y no estamos en gracia
                # ----------------------------------------------------------
                else:
                    if len(state["kp_seq"]) >= (MIN_LENGTH_FRAMES + MARGIN_FRAMES):
                        state["fix_frames"] += 1

                        # Periodo de gracia: espera DELAY_FRAMES antes de cerrar.
                        # `continue` (como el original) salta al siguiente frame.
                        if state["fix_frames"] < DELAY_FRAMES:
                            state["recording"] = True
                            cv2.imshow(WINDOW_NAME, frame)
                            key = cv2.waitKey(10) & 0xFF
                            if key in [ord("q"), ord("Q")]:
                                break
                            if key in [ord("l"), ord("L")]:
                                show_landmarks = not show_landmarks
                            continue

                        # ---- Guardar muestra ----
                        cut = MARGIN_FRAMES + DELAY_FRAMES
                        if cut > 0 and len(state["kp_seq"]) > cut:
                            state["kp_seq"] = state["kp_seq"][:-cut]
                            if save_debug_frames and len(state["raw_frames"]) > cut:
                                state["raw_frames"] = state["raw_frames"][:-cut]

                        sample_id = datetime.now().strftime("%y%m%d%H%M%S%f")
                        kp_arr = np.array(state["kp_seq"], dtype=np.float32)   # (T, 258)
                        kp_arr = resample_sequence(kp_arr, MODEL_FRAMES)        # (MODEL_FRAMES, 258)

                        np.save(os.path.join(kp_dir, f"{sample_id}.npy"), kp_arr)

                        meta = {
                            "word_id": word_id,
                            "sample_id": sample_id,
                            "schema": "v1_pose_hands",
                            "n_features": int(kp_arr.shape[1]),
                            "model_frames": int(MODEL_FRAMES),
                            "captured_frames_raw": int(len(state["kp_seq"])),
                        }
                        with open(
                            os.path.join(md_dir, f"{sample_id}.json"), "w", encoding="utf-8"
                        ) as f:
                            json.dump(meta, f, ensure_ascii=False, indent=2)

                        if save_debug_frames:
                            sample_frame_dir = os.path.join(rf_dir, sample_id)
                            ensure_dir(sample_frame_dir)
                            for i, fr in enumerate(state["raw_frames"], start=1):
                                cv2.imwrite(
                                    os.path.join(sample_frame_dir, f"frame_{i:03}.jpg"), fr
                                )

                        print(
                            f">> Guardada muestra {sample_id} "
                            f"| frames_raw={meta['captured_frames_raw']} -> {MODEL_FRAMES}"
                        )
                        state = _reset_state()

                    else:
                        # Sena demasiado corta o idle -> descartar.
                        # Diagnostico: si ves descartes con senas legitimas,
                        # baja MIN_LENGTH_FRAMES o sube DELAY_FRAMES.
                        if len(state["kp_seq"]) > 0:
                            print(
                                f"[!] Descartada: {len(state['kp_seq'])} frames "
                                f"(minimo requerido: {MIN_LENGTH_FRAMES + MARGIN_FRAMES})"
                            )
                        cv2.putText(
                            frame, "LISTO PARA CAPTURAR...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 180, 0), 2,
                        )
                        state = _reset_state()

                # ----------------------------------------------------------
                # Overlay de landmarks
                # ----------------------------------------------------------
                if show_landmarks:
                    draw_landmarks_v1(frame, results)
                    cv2.putText(
                        frame, "LANDMARKS: ON (L)", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    )
                else:
                    cv2.putText(
                        frame, "LANDMARKS: OFF (L)", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
                    )

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(10) & 0xFF
                if key in [ord("q"), ord("Q")]:
                    break
                if key in [ord("l"), ord("L")]:
                    show_landmarks = not show_landmarks

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    words = load_words(os.path.join("models", "words.json"))
    capture_word("hola", out_root="data", save_debug_frames=False, show_landmarks=True)
    print("Palabras:", words)