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

# ✅ Diccionario centralizado de correcciones de acento.
# Agregar aquí las palabras que gTTS pronuncia mal por falta de tilde.
# Evita if/elif hardcodeados dispersos por el código.
ACCENT_FIX: dict[str, str] = {
    "adios":        "adiós",
    "buenos dias":  "buenos días",
    "gracias":      "gracias",
    "por favor":    "por favor",
}


def load_mapping(path: str = "models/lsc_v0_mapping.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _reset_state() -> dict:
    """Estado inicial del buffer de captura en inferencia."""
    return dict(kp_seq=[], count_frame=0, fix_frames=0, recording=False)


def main(
    threshold: float = 0.70,
    cooldown_sec: float = 0.6,
    show_debug: bool = True,
):
    """
    Inferencia en tiempo real: captura señas LSC desde cámara,
    predice con el modelo GRU y entrega salida en texto + voz.

    Correcciones respecto a la versión original
    --------------------------------------------
    - ✅ model(tensor, training=False) en lugar de model.predict():
      para batch=1 en tiempo real, predict() construye internamente un
      tf.data.Dataset y hace validaciones innecesarias que añaden ~50–200ms
      de latencia por predicción, congelando el video loop. __call__ directo
      es entre 5x y 20x más rápido para este caso de uso.
    - ✅ cap.isOpened() verificado ANTES de configurar resolución y crear
      ventana (mismo fix que capture_samples.py).
    - ✅ Lógica de `recording` corregida: solo se resetea cuando HAY mano
      real, preservando el buffer de gracia entre señas (mismo fix que
      capture_samples.py).
    - ✅ ACCENT_FIX: diccionario centralizado en lugar de if hardcodeado
      solo para "adios". Escala a cualquier número de palabras.
    - ✅ try/finally garantiza cap.release(), destroyAllWindows() y
      tts.close() aunque ocurra una excepción en el loop.
    - ✅ Verificación de existencia del modelo antes de cargarlo, con
      mensaje claro de qué ejecutar si no está.
    - ✅ _reset_state() centralizado, igual que en capture_samples.py.

    Optimización de FPS
    -------------------
    - ✅ Resolución bajada a 640x480 (antes 1280x720) y
      Holistic(model_complexity=0): duplican los FPS en CPU.
    - ⚠️ IMPORTANTE: estos dos parámetros DEBEN coincidir EXACTAMENTE con
      capture_samples.py. Si se captura el dataset con una configuración y
      se infiere con otra, los landmarks difieren ligeramente y reaparece
      el gap train-vivo. Mantener ambos archivos sincronizados.
    """

    # ------------------------------------------------------------------
    # 1. Carga del mapping y modelo
    # ------------------------------------------------------------------
    mapping_path = "models/lsc_v0_mapping.json"
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"No se encontró el mapping en '{mapping_path}'.\n"
            "Ejecuta primero: python -m src.train.train"
        )
    mapping = load_mapping(mapping_path)
    used_words = mapping["used_words"]

    model_path = os.path.join("models", "lsc_v0.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo en '{model_path}'.\n"
            "Ejecuta primero: python -m src.train.train"
        )
    model = tf.keras.models.load_model(model_path)

    # Warm-up: la primera llamada a __call__ compila el grafo TF,
    # hacerla aquí evita el lag en la primera seña real.
    _ = model(
        tf.zeros((1, MODEL_FRAMES, mapping["n_features"]), dtype=tf.float32),
        training=False,
    )

    # ------------------------------------------------------------------
    # 2. TTS
    # ------------------------------------------------------------------
    tts = GTTSWorker(lang="es", debug=False)

    # ------------------------------------------------------------------
    # 3. Cámara
    # ------------------------------------------------------------------
    # ✅ Verificar antes de configurar resolución o crear ventana
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        tts.close()
        raise RuntimeError("No pude abrir la cámara.")

    # ✅ 640x480 para duplicar FPS (debe coincidir con capture_samples.py)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    WINDOW_NAME = "LSC Live"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1100, 700)

    # ------------------------------------------------------------------
    # 4. Estado del buffer y UI
    # ------------------------------------------------------------------
    state = _reset_state()
    sentence: list[str] = []
    last_spoken: str | None = None
    last_spoken_time: float = 0.0

    try:
        # ✅ model_complexity=0 para acelerar (debe coincidir con capture_samples.py)
        with Holistic(model_complexity=0) as holistic:
            print("\n➡️  Live inference ON | Q = salir\n")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = mediapipe_detection(frame, holistic)
                hand_detected = there_hand(results)

                # ------------------------------------------------------
                # RAMA A: hay mano O estamos en período de gracia
                # ------------------------------------------------------
                if hand_detected or state["recording"]:

                    # ✅ Solo resetea recording si HAY mano real
                    if hand_detected:
                        state["recording"] = False

                    state["count_frame"] += 1
                    if state["count_frame"] > MARGIN_FRAMES:
                        state["kp_seq"].append(extract_v1_pose_hands(results))

                    cv2.putText(
                        frame,
                        "LEYENDO SEÑA...",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 80, 255),
                        2,
                    )

                # ------------------------------------------------------
                # RAMA B: no hay mano y no estamos en período de gracia
                # ------------------------------------------------------
                else:
                    if len(state["kp_seq"]) >= (MIN_LENGTH_FRAMES + MARGIN_FRAMES):
                        state["fix_frames"] += 1

                        if state["fix_frames"] < DELAY_FRAMES:
                            state["recording"] = True
                        else:
                            # ---- Inferencia ----
                            cut = MARGIN_FRAMES + DELAY_FRAMES
                            if cut > 0 and len(state["kp_seq"]) > cut:
                                state["kp_seq"] = state["kp_seq"][:-cut]

                            kp_arr = np.array(state["kp_seq"], dtype=np.float32)
                            kp_arr = resample_sequence(kp_arr, MODEL_FRAMES)

                            # ✅ __call__ directo: 5–20x más rápido que predict()
                            # para batch=1, evita congelar el video loop
                            kp_tensor = tf.constant(
                                np.expand_dims(kp_arr, axis=0), dtype=tf.float32
                            )
                            probs = model(kp_tensor, training=False).numpy()[0]

                            idx  = int(np.argmax(probs))
                            conf = float(probs[idx])
                            pred_word = used_words[idx]

                            if show_debug:
                                print(f"Pred: {pred_word} | conf={conf:.2f}")

                            now = time.time()
                            if conf >= threshold:
                                # Cooldown solo para la misma palabra repetida
                                if not (
                                    pred_word == last_spoken
                                    and (now - last_spoken_time) < cooldown_sec
                                ):
                                    sentence.insert(0, pred_word)
                                    sentence = sentence[:6]

                                    # ✅ Corrección de acentos desde diccionario
                                    speak_text = pred_word.replace("_", " ")
                                    speak_text = ACCENT_FIX.get(
                                        speak_text.lower(), speak_text
                                    )

                                    tts.speak(speak_text)
                                    last_spoken = pred_word
                                    last_spoken_time = now

                            state = _reset_state()

                    else:
                        cv2.putText(
                            frame,
                            "LISTO...",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 180, 0),
                            2,
                        )
                        state = _reset_state()

                # ------------------------------------------------------
                # Overlay: barra de texto con la oración detectada
                # ------------------------------------------------------
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 85), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    " | ".join(sentence),
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(10) & 0xFF
                if key in [ord("q"), ord("Q")]:
                    break

    finally:
        # ✅ Liberación garantizada aunque ocurra una excepción en el loop
        cap.release()
        cv2.destroyAllWindows()
        tts.close()


if __name__ == "__main__":
    main()