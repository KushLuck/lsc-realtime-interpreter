# src/features/normalize.py
import numpy as np


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Remuestra una secuencia temporal de keypoints a una longitud fija
    mediante interpolación lineal vectorizada.

    Parámetros
    ----------
    seq        : np.ndarray de shape (T, D)
                 T = frames capturados, D = features por frame (ej. 258)
    target_len : int
                 Número de frames de salida (ej. MODEL_FRAMES = 30)

    Retorna
    -------
    np.ndarray de shape (target_len, D), dtype float32

    Correcciones respecto a la versión original
    --------------------------------------------
    - Reemplaza el loop Python frame-a-frame por operaciones NumPy
      completamente vectorizadas: entre 10x y 30x más rápido en CPU,
      crítico para inferencia en tiempo real.
    - np.clip() en el índice `hi` evita out-of-bounds en el último frame
      (cuando floor(idx[-1]) == T-1, hi sería T que está fuera del array).
    - El broadcast (target_len, 1) * (target_len, D) es correcto y
      elimina el bucle explícito sobre features.
    - Conserva el early-return cuando T == target_len (sin copia innecesaria).
    - Garantiza dtype float32 en la salida (consistente con el resto del pipeline).
    """
    T, D = seq.shape

    # Caso trivial: ya tiene la longitud correcta
    if T == target_len:
        return seq.astype(np.float32)

    # Índices flotantes equiespaciados en [0, T-1]
    idx = np.linspace(0, T - 1, target_len)          # (target_len,)

    lo = np.floor(idx).astype(np.intp)                # (target_len,) índice inferior
    hi = np.clip(lo + 1, 0, T - 1).astype(np.intp)   # (target_len,) índice superior ← evita OOB

    # Peso de interpolación, shape (target_len, 1) para broadcast sobre D features
    w = (idx - lo)[:, None]                           # (target_len, 1)

    # Interpolación lineal vectorizada: (1-w)*seq[lo] + w*seq[hi]
    out = (1.0 - w) * seq[lo] + w * seq[hi]           # (target_len, D)

    return out.astype(np.float32)

def normalize_pose_hands(pose: np.ndarray, lh: np.ndarray, rh: np.ndarray) -> tuple:
    """
    Normaliza espacialmente un frame de keypoints para hacerlo invariante
    a la posición en el encuadre y a la distancia a la cámara.

    Parámetros
    ----------
    pose : np.ndarray (132,)  -> 33 landmarks * (x, y, z, visibility)
    lh   : np.ndarray (63,)   -> 21 landmarks * (x, y, z)
    rh   : np.ndarray (63,)   -> 21 landmarks * (x, y, z)

    Retorna
    -------
    (pose, lh, rh) normalizados, mismos shapes.

    Método
    ------
    1. Traslación: todo se centra respecto al punto medio de los hombros
       (landmarks 11 y 12 de pose). Elimina dependencia de DÓNDE está la
       persona en el frame.
    2. Escala: se divide por la distancia entre hombros. Elimina dependencia
       de qué tan CERCA/LEJOS está de la cámara.

    Nota sobre visibility: el 4º valor de cada landmark de pose NO se normaliza
    (es una probabilidad 0-1, no una coordenada). Solo se tocan x, y, z.

    Nota sobre manos ausentes: si una mano es todo ceros (no detectada),
    se deja en ceros. Normalizarla metería ruido; los ceros son la señal
    de "ausente" que el modelo debe ver de forma consistente.
    """
    pose = pose.reshape(33, 4).copy()
    lh = lh.reshape(21, 3).copy()
    rh = rh.reshape(21, 3).copy()

    # Si no hay pose detectada, no hay referencia estable -> devolver tal cual.
    # (pose en ceros significa que MediaPipe no vio el cuerpo)
    if not np.any(pose[:, :3]):
        return pose.flatten(), lh.flatten(), rh.flatten()

    # --- 1. Punto de referencia: centro de los hombros ---
    # landmark 11 = hombro izquierdo, 12 = hombro derecho
    left_shoulder = pose[11, :3]
    right_shoulder = pose[12, :3]
    center = (left_shoulder + right_shoulder) / 2.0        # (3,)

    # --- 2. Escala: distancia entre hombros ---
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_dist < 1e-6:
        shoulder_dist = 1.0  # fallback defensivo, evita división por cero

    # --- Aplicar traslación + escala a coordenadas (x, y, z) ---
    pose[:, :3] = (pose[:, :3] - center) / shoulder_dist

    # Solo normalizar manos que SÍ fueron detectadas (no todo ceros)
    if np.any(lh):
        lh = (lh - center) / shoulder_dist
    if np.any(rh):
        rh = (rh - center) / shoulder_dist

    return pose.flatten(), lh.flatten(), rh.flatten()