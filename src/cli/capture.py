import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import mediapipe as mp
import numpy as np
import yaml


@dataclass
class CaptureConfig:
    camera_index: int = 0
    seconds_per_sample: float = 2.0
    target_frames: int = 60
    countdown_seconds: int = 3
    max_hands: int = 2
    min_det_conf: float = 0.5
    min_track_conf: float = 0.5


def load_vocab(vocab_path: str) -> List[Dict]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["labels"]


def flatten_hand(hand_landmarks) -> np.ndarray:
    # 21 landmarks, each: (x, y, z) -> 63 features
    arr = np.zeros((21, 3), dtype=np.float32)
    if hand_landmarks is None:
        return arr.reshape(-1)
    for i, lm in enumerate(hand_landmarks.landmark):
        arr[i] = [lm.x, lm.y, lm.z]
    return arr.reshape(-1)


def flatten_pose(pose_landmarks) -> np.ndarray:
    # 33 landmarks, each: (x, y, z, visibility) -> 132 features
    arr = np.zeros((33, 4), dtype=np.float32)
    if pose_landmarks is None:
        return arr.reshape(-1)
    for i, lm in enumerate(pose_landmarks.landmark):
        arr[i] = [lm.x, lm.y, lm.z, lm.visibility]
    return arr.reshape(-1)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def countdown(frame, secs: int) -> None:
    for s in range(secs, 0, -1):
        tmp = frame.copy()
        cv2.putText(
            tmp,
            f"Grabando en: {s}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 255, 0),
            3,
        )
        cv2.imshow("LSC Capture", tmp)
        cv2.waitKey(1000)


def main():
    cfg = CaptureConfig()

    repo_root = Path(__file__).resolve().parents[2]
    vocab_path = repo_root / "configs" / "vocab_v1_30.yaml"
    out_root = repo_root / "data" / "raw" / "samples"

    labels = load_vocab(str(vocab_path))
    idx = 0

    # --- MediaPipe modules ---
    mp_hands = mp.solutions.hands # type: ignore
    mp_pose = mp.solutions.pose # type: ignore
    mp_draw = mp.solutions.drawing_utils # type: ignore

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la cámara index={cfg.camera_index}. Prueba 1 si tienes varias."
        )

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=cfg.max_hands,
        model_complexity=1,
        min_detection_confidence=cfg.min_det_conf,
        min_tracking_confidence=cfg.min_track_conf,
    ) as hands, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Preview (not saving)
            res_h = hands.process(rgb)
            res_p = pose.process(rgb)

            # Draw hands for feedback
            if res_h.multi_hand_landmarks:
                for hand_lm in res_h.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            # UI overlay
            label_name = labels[idx]["name"]
            label_id = labels[idx]["id"]
            cv2.putText(
                frame,
                f"Etiqueta: [{label_id}] {label_name}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "SPACE: grabar | N/P: cambiar | Q: salir",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("LSC Capture", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("n"):
                idx = (idx + 1) % len(labels)
            elif key == ord("p"):
                idx = (idx - 1) % len(labels)
            elif key == 32:  # SPACE
                # Countdown using current frame
                countdown(frame, cfg.countdown_seconds)

                # Record frames
                seq = []
                start = time.time()
                while True:
                    ok2, fr = cap.read()
                    if not ok2:
                        break

                    fr = cv2.flip(fr, 1)
                    rgb2 = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

                    rh = hands.process(rgb2)
                    rp = pose.process(rgb2)

                    # Determine left/right hands consistently if possible
                    left_lm = None
                    right_lm = None
                    if rh.multi_hand_landmarks and rh.multi_handedness:
                        for lm, handed in zip(rh.multi_hand_landmarks, rh.multi_handedness):
                            label = handed.classification[0].label  # 'Left'/'Right'
                            if label == "Left":
                                left_lm = lm
                            elif label == "Right":
                                right_lm = lm

                    # Features: left hand + right hand + pose (NO face)
                    feat = np.concatenate(
                        [
                            flatten_hand(left_lm),
                            flatten_hand(right_lm),
                            flatten_pose(rp.pose_landmarks),
                        ],
                        axis=0,
                    )

                    seq.append(feat)

                    # Live feedback while recording
                    cv2.putText(
                        fr,
                        "REC...",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3,
                    )
                    cv2.imshow("LSC Capture", fr)
                    cv2.waitKey(1)

                    if (time.time() - start) >= cfg.seconds_per_sample:
                        break

                X = np.stack(seq, axis=0)  # [T, F]

                # Pad/trim to target_frames
                T, F = X.shape
                if T < cfg.target_frames:
                    pad = np.zeros((cfg.target_frames - T, F), dtype=np.float32)
                    X = np.concatenate([X, pad], axis=0)
                elif T > cfg.target_frames:
                    X = X[: cfg.target_frames]

                # Save
                ensure_dir(out_root / label_name)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                # Meta placeholders; luego los hacemos interactivos
                person = "p00"
                bg = "bg00"
                light = "light00"
                out_path = out_root / label_name / f"{timestamp}__{person}__{bg}__{light}.npz"

                np.savez_compressed(
                    out_path,
                    X=X.astype(np.float32),
                    y=np.int64(label_id),
                    label_name=label_name,
                    meta=dict(person=person, bg=bg, light=light, seconds=cfg.seconds_per_sample), # type: ignore
                )

                print(f"[OK] Guardado: {out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
