# src/features/normalize.py
import numpy as np


def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Remuestrea una secuencia temporal de keypoints a una longitud fija
    mediante interpolacion lineal vectorizada.

    Parametros
    ----------
    seq        : np.ndarray de shape (T, D)
                 T = frames capturados, D = features por frame (258)
    target_len : int
                 Numero de frames de salida (MODEL_FRAMES)

    Retorna
    -------
    np.ndarray de shape (target_len, D), dtype float32
    """
    T, D = seq.shape

    # Caso trivial: ya tiene la longitud correcta
    if T == target_len:
        return seq.astype(np.float32)

    # Caso borde: secuencia vacia o de un solo frame
    if T <= 1:
        if T == 0:
            return np.zeros((target_len, D), dtype=np.float32)
        return np.repeat(seq.astype(np.float32), target_len, axis=0)

    # Indices flotantes equiespaciados en [0, T-1]
    idx = np.linspace(0, T - 1, target_len)          # (target_len,)

    lo = np.floor(idx).astype(np.intp)                # indice inferior
    hi = np.clip(lo + 1, 0, T - 1).astype(np.intp)    # indice superior (evita OOB)

    # Peso de interpolacion, shape (target_len, 1) para broadcast sobre D
    w = (idx - lo)[:, None]                            # (target_len, 1)

    # Interpolacion lineal vectorizada: (1-w)*seq[lo] + w*seq[hi]
    out = (1.0 - w) * seq[lo] + w * seq[hi]            # (target_len, D)

    return out.astype(np.float32)


def normalize_pose_hands(pose: np.ndarray, lh: np.ndarray, rh: np.ndarray):
    """
    Normaliza espacialmente un frame de keypoints para hacerlo invariante
    a la posicion en el encuadre y a la distancia a la camara.

    Parametros
    ----------
    pose : np.ndarray (132,)  -> 33 landmarks * (x, y, z, visibility)
    lh   : np.ndarray (63,)   -> 21 landmarks * (x, y, z)
    rh   : np.ndarray (63,)   -> 21 landmarks * (x, y, z)

    Retorna
    -------
    (pose, lh, rh) normalizados, mismos shapes.

    Metodo
    ------
    1. Traslacion: todo se centra respecto al punto medio de los hombros
       (landmarks 11 y 12 de pose). Elimina dependencia de DONDE esta la
       persona en el frame.
    2. Escala: se divide por la distancia entre hombros. Elimina dependencia
       de que tan CERCA/LEJOS esta de la camara.

    Nota sobre visibility: el 4o valor de cada landmark de pose NO se normaliza
    (es una probabilidad 0-1, no una coordenada). Solo se tocan x, y, z.

    Nota sobre manos ausentes: si una mano es todo ceros (no detectada),
    se deja en ceros. Normalizarla meteria ruido; los ceros son la senal
    de "ausente" que el modelo debe ver de forma consistente.

    IMPORTANTE: esta normalizacion debe aplicarse de forma identica en
    captura e inferencia. Como ambas llaman a extract_v1_pose_hands (que a
    su vez llama a esta funcion), la consistencia queda garantizada.
    """
    pose = pose.reshape(33, 4).copy()
    lh = lh.reshape(21, 3).copy()
    rh = rh.reshape(21, 3).copy()

    # Si no hay pose detectada, no hay referencia estable -> devolver tal cual.
    if not np.any(pose[:, :3]):
        return pose.flatten(), lh.flatten(), rh.flatten()

    # 1. Punto de referencia: centro de los hombros
    #    landmark 11 = hombro izquierdo, 12 = hombro derecho
    left_shoulder = pose[11, :3]
    right_shoulder = pose[12, :3]
    center = (left_shoulder + right_shoulder) / 2.0        # (3,)

    # 2. Escala: distancia entre hombros
    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_dist < 1e-6:
        shoulder_dist = 1.0  # fallback defensivo, evita division por cero

    # Aplicar traslacion + escala a coordenadas (x, y, z) de la pose
    pose[:, :3] = (pose[:, :3] - center) / shoulder_dist

    # Solo normalizar manos que SI fueron detectadas (no todo ceros)
    if np.any(lh):
        lh = (lh - center) / shoulder_dist
    if np.any(rh):
        rh = (rh - center) / shoulder_dist

    return pose.flatten(), lh.flatten(), rh.flatten()