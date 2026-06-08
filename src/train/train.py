# src/train/train.py
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.train.dataset import load_dataset
from src.train.model_gru import build_gru_model


def main():
    """
    Entrena el modelo GRU para reconocimiento de LSC.

    Correcciones respecto a la versión original
    --------------------------------------------
    - ✅ ModelCheckpoint: guarda el mejor modelo durante el entrenamiento.
      En el original, si el proceso se interrumpía o EarlyStopping restauraba
      pesos pero el script fallaba al guardar, se perdía todo el trabajo.
    - ✅ Se guardan los índices del split (train/val) en
      models/split_indices.json para que eval.py use exactamente la misma
      partición, incluso si se agregan nuevas muestras entre runs.
    - ✅ Se guarda el historial de entrenamiento en models/history.json
      para poder graficar curvas de aprendizaje sin reentrenar.
    - ✅ Verificación mínima de muestras por clase antes de entrenar,
      con mensaje claro si alguna clase tiene muy pocas muestras para
      el split estratificado (sklearn lanza un error críptico en ese caso).
    - ✅ Resumen del modelo impreso antes de entrenar.
    - ✅ Reporte final con val_accuracy y val_loss del mejor epoch.
    """

    # ------------------------------------------------------------------
    # 1. Carga del dataset
    # ------------------------------------------------------------------
    X, y, used_words, word_to_idx = load_dataset()

    N, T, D = X.shape
    n_classes = len(used_words)

    print(f"\n✅ Dataset cargado: N={N}, T={T}, D={D}, clases={n_classes}")
    print(f"   Clases: {used_words}\n")

    # Verificación: cada clase debe tener al menos 2 muestras para
    # que train_test_split estratificado no falle
    counts = np.bincount(y)
    clases_escasas = [used_words[i] for i, c in enumerate(counts) if c < 2]
    if clases_escasas:
        raise RuntimeError(
            f"Las siguientes clases tienen menos de 2 muestras y no pueden "
            f"entrar al split estratificado: {clases_escasas}\n"
            f"Captura más muestras antes de entrenar."
        )

    # Distribución por clase
    print("   Muestras por clase:")
    for w, c in zip(used_words, counts):
        print(f"     {w:<20} {c}")
    print()

    # ------------------------------------------------------------------
    # 2. Split train / validación
    # ------------------------------------------------------------------
    idx_all = np.arange(N)
    idx_train, idx_val = train_test_split(
        idx_all,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )

    X_train, y_train = X[idx_train], y[idx_train]
    X_val,   y_val   = X[idx_val],   y[idx_val]

    print(f"   Train: {len(X_train)} muestras | Val: {len(X_val)} muestras\n")

    # ------------------------------------------------------------------
    # 3. Construcción del modelo
    # ------------------------------------------------------------------
    model = build_gru_model(T, D, n_classes)
    model.summary()
    print()

    # ------------------------------------------------------------------
    # 4. Directorio de salida y rutas
    # ------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)

    model_path       = os.path.join("models", "lsc_v0.keras")
    best_model_path  = os.path.join("models", "lsc_v0_best.keras")
    mapping_path     = os.path.join("models", "lsc_v0_mapping.json")
    split_path       = os.path.join("models", "split_indices.json")
    history_path     = os.path.join("models", "history.json")

    # ------------------------------------------------------------------
    # 5. Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        # ✅ Guarda el mejor modelo por val_accuracy durante el entrenamiento.
        # Si el proceso se interrumpe, lsc_v0_best.keras conserva el mejor
        # checkpoint alcanzado hasta ese momento.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-6,        # ✅ evita que lr baje indefinidamente
            verbose=1,
        ),
    ]

    # ------------------------------------------------------------------
    # 6. Entrenamiento
    # ------------------------------------------------------------------
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 7. Guardado del modelo final
    # ------------------------------------------------------------------
    model.save(model_path)
    print(f"\n✅ Modelo final guardado en:  {model_path}")
    print(f"✅ Mejor checkpoint guardado en: {best_model_path}")

    # ------------------------------------------------------------------
    # 8. Mapping (clases usadas en este entrenamiento)
    # ------------------------------------------------------------------
    mapping = {
        "used_words":   used_words,
        "word_to_idx":  word_to_idx,
        "timesteps":    int(T),
        "n_features":   int(D),
        "n_classes":    int(n_classes),
    }
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"✅ Mapping guardado en:          {mapping_path}")

    # ------------------------------------------------------------------
    # 9. ✅ Índices del split — para que eval.py use la misma partición
    # ------------------------------------------------------------------
    split_info = {
        "idx_train": idx_train.tolist(),
        "idx_val":   idx_val.tolist(),
        "test_size": 0.15,
        "random_state": 42,
        "N_total": int(N),
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    print(f"✅ Índices del split guardados en: {split_path}")

    # ------------------------------------------------------------------
    # 10. ✅ Historial de entrenamiento — para graficar curvas después
    # ------------------------------------------------------------------
    history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)
    print(f"✅ Historial guardado en:          {history_path}")

    # ------------------------------------------------------------------
    # 11. Reporte final
    # ------------------------------------------------------------------
    best_epoch   = int(np.argmax(history.history["val_accuracy"]))
    best_val_acc = history.history["val_accuracy"][best_epoch]
    best_val_loss = history.history["val_loss"][best_epoch]

    print(f"\n{'='*50}")
    print(f"  Mejor epoch   : {best_epoch + 1}")
    print(f"  Val accuracy  : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Val loss      : {best_val_loss:.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()