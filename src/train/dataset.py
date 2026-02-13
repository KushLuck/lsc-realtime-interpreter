# src/train/dataset.py
import os
import json
import glob
import numpy as np
from typing import List, Tuple, Dict


def load_words(words_json_path: str) -> List[str]:
    with open(words_json_path, "r", encoding="utf-8") as f:
        return json.load(f)["word_ids"]


def find_available_words(data_root: str, word_ids: List[str]) -> List[str]:
    """Devuelve solo las palabras que tienen al menos 1 muestra .npy."""
    available = []
    for w in word_ids:
        pattern = os.path.join(data_root, "keypoints_v1", w, "*.npy")
        if len(glob.glob(pattern)) > 0:
            available.append(w)
    return available


def load_dataset(
    data_root: str = "data",
    words_json_path: str = os.path.join("models", "words.json"),
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, int]]:
    """
    Carga:
      X: (N, T, D) float32
      y: (N,) int
      used_words: lista de clases usadas (solo las que tienen datos)
      word_to_idx: dict palabra -> índice
    """
    all_words = load_words(words_json_path)
    used_words = find_available_words(data_root, all_words)

    if len(used_words) < 2:
        raise RuntimeError(
            "Necesitas al menos 2 palabras con muestras para entrenar.\n"
            "Verifica que existan archivos en data/keypoints_v1/<palabra>/*.npy"
        )

    word_to_idx = {w: i for i, w in enumerate(used_words)}

    X_list, y_list = [], []
    for w in used_words:
        files = sorted(glob.glob(os.path.join(data_root, "keypoints_v1", w, "*.npy")))
        for fpath in files:
            arr = np.load(fpath)  # (T, D)
            X_list.append(arr)
            y_list.append(word_to_idx[w])

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, used_words, word_to_idx
