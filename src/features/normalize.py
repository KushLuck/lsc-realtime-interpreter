# src/features/normalize.py
import numpy as np

def resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    seq: (T, D)
    return: (target_len, D)
    """
    T, D = seq.shape
    if T == target_len:
        return seq

    # indices float equiespaciados
    idx = np.linspace(0, T - 1, target_len)
    out = np.zeros((target_len, D), dtype=np.float32)

    for i, x in enumerate(idx):
        lo = int(np.floor(x))
        hi = int(np.ceil(x))
        w = x - lo
        if lo == hi:
            out[i] = seq[lo]
        else:
            out[i] = (1 - w) * seq[lo] + w * seq[hi]
    return out
