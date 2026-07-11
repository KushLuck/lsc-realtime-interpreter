# src/features/mediapipe_holistic.py
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic

from src.features.normalize import normalize_pose_hands


def mediapipe_detection(frame_bgr: np.ndarray, model: Holistic):
    """
    Convierte el frame BGR a RGB, lo pasa por MediaPipe Holistic y retorna
    los resultados. Restaura writeable para no dejar el array bloqueado y
    preserva frame_bgr intacto para que el caller pueda dibujar sobre el.
    """
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = model.process(img)
    img.flags.writeable = True
    return results


def there_hand(results) -> bool:
    """True si MediaPipe detecto al menos una mano (izquierda o derecha)."""
    return (
        results.left_hand_landmarks is not None
        or results.right_hand_landmarks is not None
    )


def extract_v1_pose_hands(results) -> np.ndarray:
    """
    Extrae, normaliza y concatena los keypoints de pose y manos en un vector
    plano.

    Schema v1 (258 features):
      - Pose    : 33 landmarks * 4 (x, y, z, visibility) = 132
      - Mano izq: 21 landmarks * 3 (x, y, z)             =  63
      - Mano der: 21 landmarks * 3 (x, y, z)             =  63

    Normalizacion espacial
    ----------------------
    Tras extraer las coordenadas crudas de MediaPipe (relativas al frame de
    la camara), se aplica normalize_pose_hands() para hacer el vector
    invariante a la POSICION de la persona en el encuadre y a la DISTANCIA
    a la camara. Critico para la robustez en vivo: sin ello el modelo aprende
    posiciones absolutas del set de entrenamiento y falla cuando el usuario
    se para en otra posicion o distancia.

    Esta normalizacion se aplica de forma identica en captura e inferencia,
    porque ambas pasan por esta misma funcion.

    Retorna
    -------
    np.ndarray de shape (258,), dtype float32, normalizado. Rellena con ceros
    las partes no detectadas; los ceros se preservan sin normalizar como
    senal de "ausente".
    """
    # --- Pose (33 * 4 = 132) ---
    if results.pose_landmarks:
        pose = np.array(
            [[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(33 * 4, dtype=np.float32)

    # --- Mano izquierda (21 * 3 = 63) ---
    if results.left_hand_landmarks:
        lh = np.array(
            [[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        lh = np.zeros(21 * 3, dtype=np.float32)

    # --- Mano derecha (21 * 3 = 63) ---
    if results.right_hand_landmarks:
        rh = np.array(
            [[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        rh = np.zeros(21 * 3, dtype=np.float32)

    # --- Normalizacion espacial (invarianza a posicion y escala) ---
    pose, lh, rh = normalize_pose_hands(pose, lh, rh)

    return np.concatenate([pose, lh, rh])  # (258,) normalizado