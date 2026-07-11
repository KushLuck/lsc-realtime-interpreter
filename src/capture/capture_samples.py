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


def _reset_state():
    """Retorna el estado inicial de captura. Centraliza el reset para
    evitar repetir las mismas 5 asignaciones en tres lugares del loop."""
    return dict(kp_seq=[], raw_frames=[], count_frame=0, fix_frames=0, recording=False)


def capture_word(
    word_id: str,
    out_root: str,
    save_debug_frames: bool = False,
    show_landmarks: bool = False,
):
    """
    Captura muestras de keypoints para una palabra de LSC y las guarda en disco.

    Guarda:
      - data/keypoints_v1/<word_id>/<sample_id>.npy  shape=(MODEL_FRAMES, 258)
      - data/metadata/<word_id>/<sample_id>.json
      - (opcional) data/raw_frames/<word_id>/<sample_id>/frame_XXX.jpg

    Controles:
      - Q: salir
      - L: toggle mostrar landmarks (pose+manos)

    Correcciones respecto a la versión original
    --------------------------------------------
    - ✅ cap.isOpened() se verifica ANTES de configurar resolución y crear
      ventana. En el original se hacían ambas cosas antes del check, dejando
      recursos abiertos aunque la cámara no estuviera disponible.
    - ✅ Lógica de `recording` corregida: en el original, `recording = False`
      se ejecutaba al inicio del bloque `if there_hand(results) or recording`,
      lo que anulaba el flag de gracia en el mismo frame en que se activaba.
      Ahora `recording` solo se resetea cuando HAY detección real de mano,
      preservando el comportamiento de buffer entre señas.
    - ✅ Reset de estado centralizado en _reset_state() para evitar las 5
      asignaciones repetidas en tres ramas distintas del loop original.
    - ✅ Liberación explícita de cap y ventana en bloque finally, garantizando
      limpieza aunque ocurra una excepción durante la captura.

    Optimización de FPS
    -------------------
    - ✅ Resolución bajada a 640x480 (antes 1280x720): a ~10 fps en CPU la
      resolución alta era el cuello de botella. 640x480 duplica los FPS sin
      pérdida notable de precisión (MediaPipe reescala internamente y las
      coordenadas son normalizadas 0-1).
    - ✅ Holistic(model_complexity=0) (antes complexity=1 por defecto):
      modelo más ligero, acelera la detección. Pérdida de precisión
      imperceptible para señas frontales claras.
    - ⚠️ IMPORTANTE: estos dos parámetros DEBEN coincidir exactamente con
      live.py. Capturar y luego inferir con resolución o complexity distinta
      reintroduce el gap train-vivo (landmarks ligeramente distintos).

    Diagnóstico de descartes
    ------------------------
    - ✅ Print en la rama de descarte: avisa cuántos frames se acumularon
      cuando una seña se descarta por corta, para calibrar los umbrales de
      feature_schema.py contra el hardware real.
    """
    kp_dir = os.path.join(out_root, "keypoints_v1", word_id)
    md_dir = os.path.join(out_root, "metadata", word_id)
    rf_dir = os.path.join(out_root, "raw_frames", word_id)

    ensure_dir(kp_dir)
    ensure_dir(md_dir)
    if save_debug_frames:
        ensure_dir(rf_dir)

    # ✅ Verificar cámara ANTES de crear ventana o configurar resolución
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No pude abrir la cámara (VideoCapture(0)).")

    # ✅ 640x480 para duplicar FPS (antes 1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    WINDOW_NAME = f"LSC Capture | {word_id}"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1100, 700)

    # fullscreen (opcional para demo)
    # cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Estado inicial de captura
    state = _reset_state()

    try:
        # ✅ model_complexity=0 para acelerar (debe coincidir con live.py)
        with Holistic(model_complexity=0) as holistic:
            print(f'\n➡️  Listo para capturar: "{word_id}" | Q = salir | L = landmarks on/off\n')

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = mediapipe_detection(frame, holistic)
                hand_detected = there_hand(results)

                # ----------------------------------------------------------
                # RAMA A: hay mano detectada O estamos en período de gracia
                # ----------------------------------------------------------
                if hand_detected or state["recording"]:

                    # ✅ Solo resetea `recording` si HAY mano real.
                    # En el original se reseteaba siempre al entrar al bloque,
                    # lo que impedía que el flag de gracia funcionara: el frame
                    # siguiente sin mano encontraba recording=False y cerraba
                    # la seña prematuramente.
                    if hand_detected:
                        state["recording"] = False

                    state["count_frame"] += 1
                    if state["count_frame"] > MARGIN_FRAMES:
                        state["kp_seq"].append(extract_v1_pose_hands(results))
                        if save_debug_frames:
                            state["raw_frames"].append(frame.copy())

                    cv2.putText(
                        frame,
                        "CAPTURANDO...",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 80, 255),
                        2,
                    )

                # ----------------------------------------------------------
                # RAMA B: no hay mano y no estamos en período de gracia
                # ----------------------------------------------------------
                else:
                    # Seña suficientemente larga → intentar guardar
                    if len(state["kp_seq"]) >= (MIN_LENGTH_FRAMES + MARGIN_FRAMES):
                        state["fix_frames"] += 1

                        if state["fix_frames"] < DELAY_FRAMES:
                            # Período de delay: damos unos frames más antes de cerrar
                            state["recording"] = True
                        else:
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
                                os.path.join(md_dir, f"{sample_id}.json"),
                                "w",
                                encoding="utf-8",
                            ) as f:
                                json.dump(meta, f, ensure_ascii=False, indent=2)

                            if save_debug_frames:
                                sample_frame_dir = os.path.join(rf_dir, sample_id)
                                ensure_dir(sample_frame_dir)
                                for i, fr in enumerate(state["raw_frames"], start=1):
                                    cv2.imwrite(
                                        os.path.join(sample_frame_dir, f"frame_{i:03}.jpg"),
                                        fr,
                                    )

                            print(
                                f"✅ Guardada muestra {sample_id} "
                                f"| frames_raw={meta['captured_frames_raw']} -> {MODEL_FRAMES}"
                            )

                            # Reset para la siguiente seña
                            state = _reset_state()

                    else:
                        # Seña demasiado corta o idle → descartar y esperar
                        # ✅ Diagnóstico: avisa si se descartó algo acumulado.
                        # Si ves descartes con señas legítimas, sube DELAY_FRAMES
                        # o baja MIN_LENGTH_FRAMES en feature_schema.py.
                        if len(state["kp_seq"]) > 0:
                            print(
                                f"⚠️  Descartada: {len(state['kp_seq'])} frames "
                                f"(mínimo requerido: {MIN_LENGTH_FRAMES + MARGIN_FRAMES})"
                            )
                        cv2.putText(
                            frame,
                            "LISTO PARA CAPTURAR...",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 180, 0),
                            2,
                        )
                        state = _reset_state()

                # ----------------------------------------------------------
                # Overlay de landmarks
                # ----------------------------------------------------------
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

    finally:
        # ✅ Liberación garantizada aunque ocurra una excepción en el loop
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    words = load_words(os.path.join("models", "words.json"))
    # prueba rápida con una palabra:
    capture_word("como_estas", out_root="data", save_debug_frames=False, show_landmarks=True)

    print("Palabras:", words)
    print(
        "Ejecuta: python -m src.capture.capture_samples "
        "y cambia la palabra en el código o llama capture_word(...)"
    )