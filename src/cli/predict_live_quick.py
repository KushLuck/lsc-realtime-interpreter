import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import joblib
import yaml


SECONDS = 2.0
TARGET_FRAMES = 60

# Ajusta este umbral según lo que veas (0.70–0.85 suele ir bien)
CONF_THRESHOLD = 0.75


def load_id_to_name(vocab_path: Path) -> dict:
    with open(vocab_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {int(item["id"]): item["name"] for item in cfg["labels"]}


def flatten_hand(hand_landmarks):
    arr = np.zeros((21, 3), dtype=np.float32)
    if hand_landmarks is None:
        return arr.reshape(-1)
    for i, lm in enumerate(hand_landmarks.landmark):
        arr[i] = [lm.x, lm.y, lm.z]
    return arr.reshape(-1)


def flatten_pose(pose_landmarks):
    arr = np.zeros((33, 4), dtype=np.float32)
    if pose_landmarks is None:
        return arr.reshape(-1)
    for i, lm in enumerate(pose_landmarks.landmark):
        arr[i] = [lm.x, lm.y, lm.z, lm.visibility]
    return arr.reshape(-1)


def make_features(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    dX = np.diff(X, axis=0)
    delta_mean = np.abs(dX).mean(axis=0)
    return np.concatenate([mean, std, delta_mean], axis=0)


def main():
    repo_root = Path(__file__).resolve().parents[2]

    vocab_path = repo_root / "configs" / "vocab_v1_30.yaml"
    id_to_name = load_id_to_name(vocab_path)

    model_path = repo_root / "models_artifacts" / "v1" / "quick_lr.pkl"
    model = joblib.load(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara (index 0). Prueba con 1 si tienes varias.")

    mp_hands = mp.solutions.hands # type: ignore
    mp_pose = mp.solutions.pose # type: ignore

    last_pred_id = None
    last_pred_name = None
    last_conf = None

    with mp_hands.Hands(max_num_hands=2) as hands, mp_pose.Pose() as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            # UI
            cv2.putText(
                frame,
                "SPACE: grabar | Q: salir",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            if last_pred_name is not None:
                txt = f"Prediccion: {last_pred_name}"
                if last_conf is not None:
                    txt += f" | conf={last_conf:.2f}"
                cv2.putText(
                    frame,
                    txt,
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0) if (last_conf is not None and last_conf >= CONF_THRESHOLD) else (0, 255, 255),
                    3,
                )

            cv2.imshow("LSC Quick Predictor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == 32:  # SPACE
                seq = []
                start = time.time()

                while True:
                    ok2, fr = cap.read()
                    if not ok2:
                        break

                    fr = cv2.flip(fr, 1)
                    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

                    rh = hands.process(rgb)
                    rp = pose.process(rgb)

                    left, right = None, None
                    if rh.multi_hand_landmarks and rh.multi_handedness:
                        for lm, handed in zip(rh.multi_hand_landmarks, rh.multi_handedness):
                            if handed.classification[0].label == "Left":
                                left = lm
                            else:
                                right = lm

                    feat_frame = np.concatenate(
                        [flatten_hand(left), flatten_hand(right), flatten_pose(rp.pose_landmarks)],
                        axis=0,
                    )
                    seq.append(feat_frame)

                    # feedback mínimo mientras graba
                    cv2.putText(
                        fr,
                        "REC...",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                    )
                    cv2.imshow("LSC Quick Predictor", fr)
                    cv2.waitKey(1)

                    if (time.time() - start) >= SECONDS:
                        break

                X = np.stack(seq)  # [T, F]

                # Pad/trim
                if X.shape[0] < TARGET_FRAMES:
                    pad = np.zeros((TARGET_FRAMES - X.shape[0], X.shape[1]), dtype=np.float32)
                    X = np.vstack([X, pad])
                else:
                    X = X[:TARGET_FRAMES]

                feat = make_features(X).reshape(1, -1)

                # Predicción + confianza
                proba = model.predict_proba(feat)[0]
                pred_id = int(np.argmax(proba))
                conf = float(proba[pred_id])

                if conf < CONF_THRESHOLD:
                    pred_name = "NO SEGURO"
                else:
                    pred_name = id_to_name.get(pred_id, f"id_{pred_id}")

                last_pred_id = pred_id
                last_pred_name = pred_name
                last_conf = conf

                print(f">>> Prediccion: {pred_name} ({pred_id}) | conf={conf:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
