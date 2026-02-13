# src/infer/live.py
import os
import json
import time

import cv2
import numpy as np
import tensorflow as tf
from mediapipe.python.solutions.holistic import Holistic

from src.infer.tts_gtts import GTTSWorker  # ✅ usa gTTS (online) estable

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


def load_mapping(path="models/lsc_v0_mapping.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(
    threshold: float = 0.70,
    cooldown_sec: float = 0.6,
    show_debug: bool = True,
):
    mapping = load_mapping()
    used_words = mapping["used_words"]

    model = tf.keras.models.load_model(os.path.join("models", "lsc_v0.keras"))

    # ✅ TTS gTTS (online) - NO se cuelga como pyttsx3 en Windows
    tts = GTTSWorker(lang="es", debug=False)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    WINDOW_NAME = "LSC Live"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1100, 700)

    if not cap.isOpened():
        raise RuntimeError("No pude abrir la cámara.")

    kp_seq = []
    count_frame = 0
    fix_frames = 0
    recording = False

    sentence = []
    last_spoken = None
    last_spoken_time = 0.0

    with Holistic() as holistic:
        print("\n➡️  Live inference ON | Q = salir\n")

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

                cv2.putText(
                    frame,
                    "LEYENDO SEÑA...",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 80, 255),
                    2,
                )

            else:
                if len(kp_seq) >= (MIN_LENGTH_FRAMES + MARGIN_FRAMES):
                    fix_frames += 1

                    if fix_frames < DELAY_FRAMES:
                        recording = True
                    else:
                        cut = (MARGIN_FRAMES + DELAY_FRAMES)
                        if cut > 0 and len(kp_seq) > cut:
                            kp_seq = kp_seq[:-cut]

                        kp_arr = np.array(kp_seq, dtype=np.float32)
                        kp_arr = resample_sequence(kp_arr, MODEL_FRAMES)

                        probs = model.predict(np.expand_dims(kp_arr, axis=0), verbose=0)[0]
                        idx = int(np.argmax(probs))
                        conf = float(probs[idx])
                        pred_word = used_words[idx]

                        if show_debug:
                            print(f"Pred: {pred_word} | conf={conf:.2f}")

                        now = time.time()
                        if conf >= threshold:
                            # Cooldown SOLO para repetir la MISMA palabra
                            if pred_word == last_spoken and (now - last_spoken_time) < cooldown_sec:
                                pass
                            else:
                                sentence.insert(0, pred_word)
                                sentence = sentence[:6]

                                speak_text = pred_word.replace("_", " ")
                                if speak_text.lower() == "adios":
                                    speak_text = "adiós"

                                tts.speak(speak_text)
                                last_spoken = pred_word
                                last_spoken_time = now

                        kp_seq = []
                        count_frame = 0
                        fix_frames = 0
                        recording = False

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
                    kp_seq = []
                    count_frame = 0
                    fix_frames = 0
                    recording = False

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

    cap.release()
    cv2.destroyAllWindows()
    tts.close()


if __name__ == "__main__":
    main()
