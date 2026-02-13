# src/features/mediapipe_holistic.py
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic

def mediapipe_detection(frame_bgr, model: Holistic):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = model.process(img)
    return results

def there_hand(results) -> bool:
    return (results.left_hand_landmarks is not None) or (results.right_hand_landmarks is not None)

def extract_v1_pose_hands(results) -> np.ndarray:
    # Pose
    if results.pose_landmarks:
        pose = np.array([[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark], dtype=np.float32).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    # Left hand
    if results.left_hand_landmarks:
        lh = np.array([[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    # Right hand
    if results.right_hand_landmarks:
        rh = np.array([[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    return np.concatenate([pose, lh, rh], axis=0)  # (258,)
