# src/infer/live.py
import os
import json
import time

import cv2
import numpy as np
import tensorflow as tf
from mediapipe.python.solutions.holistic import Holistic

from src.infer.tts_gtts import GTTSWorker
from src.app.ui import render_live_overlay

from src.features.mediapipe_holistic import (
    mediapipe_detection,
    there_hand,
    extract_v1_pose_hands,
)
from src.features.normalize import resample_sequence
from src.features.feature_schema import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    MODEL_FRAMES,
    MIN_LENGTH_FRAMES,
    MARGIN_FRAMES,
    DELAY_FRAMES,
)

# Diccionario centralizado de correcciones de acento para gTTS.
ACCENT_FIX = {
    "adios":        "adios",
    "buenos dias":  "buenos dias",
    "gracias":      "gracias",
    "por favor":    "por favor",
}


def load_mapping(path: str = "models/lsc_v0_mapping.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _reset_state() -> dict:
    return dict(kp_seq=[], count_frame=0, fix_frames=0)


def main(
    threshold: float = 0.70,
    cooldown_sec: float = 0.6,
    show_debug: bool = True,
):
    """
    Inferencia en tiempo real: captura senas LSC desde camara, predice con
    el modelo GRU y entrega salida en texto + voz.

    Diseno (consistente con capture_samples.py)
    -------------------------------------------
    - model(tensor, training=False) en lugar de model.predict(): 5-20x mas
      rapido para batch=1, evita congelar el video loop.
    - Maquina de estados con periodo de gracia estilo repo original + reset
      de fix_frames al reaparecer la mano (robusto al parpadeo).
    - complexity=1 + tracking 0.3 + resolucion compartida con
      capture_samples.py para evitar el gap train-vivo.
    - Normalizacion espacial heredada de extract_v1_pose_hands.
    - try/finally garantiza liberacion de camara, ventana y TTS.
    """

    # 1. Mapping y modelo
    mapping_path = "models/lsc_v0_mapping.json"
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"No se encontro el mapping en '{mapping_path}'.\n"
            "Ejecuta primero: python -m src.train.train"
        )
    mapping = load_mapping(mapping_path)
    used_words = mapping["used_words"]

    model_path = os.path.join("models", "lsc_v0.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontro el modelo en '{model_path}'.\n"
            "Ejecuta primero: python -m src.train.train"
        )
    model = tf.keras.models.load_model(model_path)

    # Warm-up: compila el grafo TF para evitar lag en la primera sena
    _ = model(
        tf.zeros((1, MODEL_FRAMES, mapping["n_features"]), dtype=tf.float32),
        training=False,
    )

    # 2. TTS
    tts = GTTSWorker(lang="es", debug=False)

    # 3. Camara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        tts.close()
        raise RuntimeError("No pude abrir la camara.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    WINDOW_NAME = "LSC Live"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, actual_width, actual_height)

    # 4. Estado
    state = _reset_state()
    sentence = []
    last_spoken = None
    last_spoken_time = 0.0
    last_prediction = None
    last_confidence = None

    try:
        # complexity=1 + tracking 0.3: DEBE coincidir con capture_samples.py
        with Holistic(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
        ) as holistic:
            print(
                f"\n>> Live inference ON | {actual_width}x{actual_height} "
                "| Q = salir\n"
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = mediapipe_detection(frame, holistic)
                hand_detected = there_hand(results)
                visual_status = "LISTO"

                # ------------------------------------------------------
                # RAMA A: hay mano
                # ------------------------------------------------------
                if hand_detected:
                    # La mano reaparecio: reinicia el conteo de cierre.
                    state["fix_frames"] = 0

                    state["count_frame"] += 1
                    if state["count_frame"] > MARGIN_FRAMES:
                        state["kp_seq"].append(extract_v1_pose_hands(results))

                    visual_status = "LEYENDO SENA"

                # ------------------------------------------------------
                # RAMA B: no hay mano
                # ------------------------------------------------------
                else:
                    # Tolera parpadeos de MediaPipe tambien antes de llegar
                    # a la longitud minima; no cortes el inicio de la sena.
                    if 0 < len(state["kp_seq"]) < MIN_LENGTH_FRAMES:
                        state["fix_frames"] += 1
                        if state["fix_frames"] < DELAY_FRAMES:
                            view = render_live_overlay(
                                frame, sentence, "PROCESANDO",
                                last_prediction, last_confidence,
                            )
                            cv2.imshow(WINDOW_NAME, view)
                            if cv2.waitKey(10) & 0xFF in [ord("q"), ord("Q")]:
                                break
                            continue

                    if len(state["kp_seq"]) >= MIN_LENGTH_FRAMES:
                        state["fix_frames"] += 1

                        # Gracia: espera DELAY_FRAMES antes de inferir.
                        if state["fix_frames"] < DELAY_FRAMES:
                            # Render + teclas antes de saltar
                            view = render_live_overlay(
                                frame, sentence, "PROCESANDO",
                                last_prediction, last_confidence,
                            )
                            cv2.imshow(WINDOW_NAME, view)
                            if cv2.waitKey(10) & 0xFF in [ord("q"), ord("Q")]:
                                break
                            continue

                        # ---- Inferencia ----
                        kp_arr = np.array(state["kp_seq"], dtype=np.float32)
                        kp_arr = resample_sequence(kp_arr, MODEL_FRAMES)

                        kp_tensor = tf.constant(
                            np.expand_dims(kp_arr, axis=0), dtype=tf.float32
                        )
                        probs = model(kp_tensor, training=False).numpy()[0]

                        idx = int(np.argmax(probs))
                        conf = float(probs[idx])
                        pred_word = used_words[idx]
                        last_prediction = pred_word
                        last_confidence = conf

                        if show_debug:
                            print(f"Pred: {pred_word} | conf={conf:.2f}")

                        now = time.time()
                        if conf >= threshold:
                            if not (
                                pred_word == last_spoken
                                and (now - last_spoken_time) < cooldown_sec
                            ):
                                sentence.insert(0, pred_word)
                                sentence = sentence[:6]

                                speak_text = pred_word.replace("_", " ")
                                speak_text = ACCENT_FIX.get(speak_text.lower(), speak_text)

                                tts.speak(speak_text)
                                last_spoken = pred_word
                                last_spoken_time = now

                        state = _reset_state()

                    else:
                        state = _reset_state()

                # ------------------------------------------------------
                # Overlay: barra de texto con la oracion detectada
                # ------------------------------------------------------
                view = render_live_overlay(
                    frame, sentence, visual_status,
                    last_prediction, last_confidence,
                )
                cv2.imshow(WINDOW_NAME, view)
                if cv2.waitKey(10) & 0xFF in [ord("q"), ord("Q")]:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts.close()


if __name__ == "__main__":
    main()
