# src/infer/live.py
import os
import json
import time

import cv2
import numpy as np
import tensorflow as tf
from mediapipe.python.solutions.holistic import Holistic

from src.infer.tts_gtts import GTTSWorker

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
    return dict(kp_seq=[], count_frame=0, fix_frames=0, recording=False)


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
    - complexity=1 + tracking 0.3 + resolucion 640x480: DEBE coincidir con
      capture_samples.py o reaparece el gap train-vivo.
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

    # 640x480: DEBE coincidir con capture_samples.py
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    WINDOW_NAME = "LSC Live"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1100, 700)

    # 4. Estado
    state = _reset_state()
    sentence = []
    last_spoken = None
    last_spoken_time = 0.0

    try:
<<<<<<< HEAD
        # ✅ model_complexity=0 para acelerar (debe coincidir con capture_samples.py)
        with Holistic(model_complexity=1) as holistic:
            print("\n➡️  Live inference ON | Q = salir\n")
=======
        # complexity=1 + tracking 0.3: DEBE coincidir con capture_samples.py
        with Holistic(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
        ) as holistic:
            print("\n>> Live inference ON | Q = salir\n")
>>>>>>> 7feb2c932cc126039fe7b9b04d014e6c50ad8c31

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = mediapipe_detection(frame, holistic)
                hand_detected = there_hand(results)

                # ------------------------------------------------------
                # RAMA A: hay mano O estamos en gracia
                # ------------------------------------------------------
                if hand_detected or state["recording"]:
                    # La mano reaparecio: sal de gracia y reinicia el conteo.
                    if hand_detected:
                        state["recording"] = False
                        state["fix_frames"] = 0

                    state["count_frame"] += 1
                    if state["count_frame"] > MARGIN_FRAMES:
                        state["kp_seq"].append(extract_v1_pose_hands(results))

                    cv2.putText(
                        frame, "LEYENDO SENA...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 80, 255), 2,
                    )

                # ------------------------------------------------------
                # RAMA B: no hay mano y no estamos en gracia
                # ------------------------------------------------------
                else:
                    if len(state["kp_seq"]) >= (MIN_LENGTH_FRAMES + MARGIN_FRAMES):
                        state["fix_frames"] += 1

                        # Gracia: espera DELAY_FRAMES antes de inferir.
                        if state["fix_frames"] < DELAY_FRAMES:
                            state["recording"] = True
                            # Render + teclas antes de saltar
                            h, w = frame.shape[:2]
                            cv2.rectangle(frame, (0, 0), (w, 85), (0, 0, 0), -1)
                            cv2.putText(
                                frame, " | ".join(sentence), (10, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                            )
                            cv2.imshow(WINDOW_NAME, frame)
                            if cv2.waitKey(10) & 0xFF in [ord("q"), ord("Q")]:
                                break
                            continue

                        # ---- Inferencia ----
                        cut = MARGIN_FRAMES + DELAY_FRAMES
                        if cut > 0 and len(state["kp_seq"]) > cut:
                            state["kp_seq"] = state["kp_seq"][:-cut]

                        kp_arr = np.array(state["kp_seq"], dtype=np.float32)
                        kp_arr = resample_sequence(kp_arr, MODEL_FRAMES)

                        kp_tensor = tf.constant(
                            np.expand_dims(kp_arr, axis=0), dtype=tf.float32
                        )
                        probs = model(kp_tensor, training=False).numpy()[0]

                        idx = int(np.argmax(probs))
                        conf = float(probs[idx])
                        pred_word = used_words[idx]

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
                        cv2.putText(
                            frame, "LISTO...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 180, 0), 2,
                        )
                        state = _reset_state()

                # ------------------------------------------------------
                # Overlay: barra de texto con la oracion detectada
                # ------------------------------------------------------
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 85), (0, 0, 0), -1)
                cv2.putText(
                    frame, " | ".join(sentence), (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                )

                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(10) & 0xFF in [ord("q"), ord("Q")]:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tts.close()


if __name__ == "__main__":
    main()