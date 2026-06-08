# src/features/mediapipe_holistic.py
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic


def mediapipe_detection(frame_bgr: np.ndarray, model: Holistic):
    """
    Convierte el frame BGR a RGB, lo pasa por MediaPipe Holistic
    y retorna los resultados.

    Parámetros
    ----------
    frame_bgr : np.ndarray
        Frame capturado por OpenCV en formato BGR, shape (H, W, 3).
    model : Holistic
        Instancia activa de MediaPipe Holistic.

    Retorna
    -------
    results : NamedTuple de MediaPipe con los landmarks detectados.

    Correcciones respecto a la versión original
    --------------------------------------------
    - Se restaura img.flags.writeable = True después de model.process().
      En la versión original el array RGB quedaba bloqueado como no-escribible
      de forma permanente. Aunque no causaba error inmediato (porque
      capture_samples.py y live.py operan sobre `frame` BGR por separado),
      cualquier refactor que reutilice `img` lanzaría ValueError sin razón
      aparente.
    - La conversión BGR→RGB se hace sobre una vista del frame original;
      se preserva `frame_bgr` intacto para que el caller pueda seguir
      dibujando sobre él sin side-effects.
    """
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # MediaPipe no necesita que el array sea escribible durante process();
    # desactivarlo evita una copia interna innecesaria (pequeña ganancia de memoria).
    img.flags.writeable = False
    results = model.process(img)

    # ✅ Restaurar writeable para no dejar el array bloqueado
    img.flags.writeable = True

    return results


def there_hand(results) -> bool:
    """
    Retorna True si MediaPipe detectó al menos una mano (izquierda o derecha).
    """
    return (
        results.left_hand_landmarks is not None
        or results.right_hand_landmarks is not None
    )


def extract_v1_pose_hands(results) -> np.ndarray:
    """
    Extrae y concatena los keypoints de pose y manos en un vector plano.

    Schema v1:
      - Pose   : 33 landmarks × 4 valores (x, y, z, visibility) = 132
      - Mano izq: 21 landmarks × 3 valores (x, y, z)             =  63
      - Mano der: 21 landmarks × 3 valores (x, y, z)             =  63
      ---------------------------------------------------------------
      Total    : 258 features

    Retorna
    -------
    np.ndarray de shape (258,), dtype float32.
    Rellena con ceros las partes no detectadas (pose o manos ausentes).
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

    return np.concatenate([pose, lh, rh])  # (258,)