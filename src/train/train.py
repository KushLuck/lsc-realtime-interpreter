# src/train/train.py
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.train.dataset import load_dataset
from src.train.model_gru import build_gru_model


def main():
    X, y, used_words, word_to_idx = load_dataset()

    N, T, D = X.shape
    n_classes = len(used_words)

    print(f"✅ Dataset cargado: N={N}, T={T}, D={D}, clases={n_classes}")
    print("Clases usadas:", used_words)

    # split estratificado
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.15,
        random_state=42,
        stratify=y
    )

    model = build_gru_model(T, D, n_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    os.makedirs("models", exist_ok=True)

    model_path = os.path.join("models", "lsc_v0.keras")
    model.save(model_path)
    print(f"✅ Modelo guardado en: {model_path}")

    # Guardamos el mapping real usado en el entrenamiento
    mapping = {
        "used_words": used_words,
        "word_to_idx": word_to_idx,
        "timesteps": int(T),
        "n_features": int(D),
    }
    with open(os.path.join("models", "lsc_v0_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print("✅ Mapping guardado en: models/lsc_v0_mapping.json")


if __name__ == "__main__":
    main()
