# src/train/eval.py
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from src.train.dataset import load_dataset


def main():
    """
    Evalúa el modelo entrenado sobre el split de validación.

    Correcciones respecto a la versión original
    --------------------------------------------
    - ❌ Original: evaluaba sobre TODO el dataset (X completo), incluyendo
      los datos de entrenamiento que el modelo ya memorizó. Las métricas
      resultantes eran optimistas y no reflejaban la capacidad real de
      generalización del modelo.

    - ✅ Corregido: se replica el mismo split estratificado usado en train.py
      (test_size=0.15, random_state=42, stratify=y) para evaluar únicamente
      sobre X_val / y_val — datos que el modelo nunca vio durante el
      entrenamiento.

    Nota: la forma correcta a largo plazo es guardar los índices del split
    en train.py (ej. en models/split_indices.json) y cargarlos aquí,
    así se garantiza que ambos scripts usen exactamente la misma partición
    incluso si se agregan nuevas muestras entre runs. Por ahora replicar
    los parámetros del split es suficiente para el estado actual del proyecto.
    """
    # Carga el dataset completo (igual que en train.py)
    X, y, used_words, _ = load_dataset()

    N, T, D = X.shape
    print(f"Dataset total: N={N}, T={T}, D={D}, clases={len(used_words)}")
    print(f"Clases: {used_words}\n")

    # ✅ Mismo split estratificado que train.py — evalúa solo sobre validación
    _, X_val, _, y_val = train_test_split(
        X, y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )

    print(f"Muestras de validación: {len(X_val)} ({len(X_val)/N*100:.1f}% del total)\n")

    # Carga el modelo entrenado
    model_path = os.path.join("models", "lsc_v0.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo en '{model_path}'.\n"
            "Ejecuta primero: python -m src.train.train"
        )

    model = tf.keras.models.load_model(model_path)

    # Inferencia sobre el split de validación
    probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # --- Reporte de clasificación ---
    print("=== Classification report (validación) ===")
    print(
        classification_report(
            y_val,
            y_pred,
            target_names=used_words,
            zero_division=0,  # evita warning si alguna clase no aparece en val
        )
    )

    # --- Matriz de confusión ---
    cm = confusion_matrix(y_val, y_pred)
    print("=== Confusion matrix (filas=real, columnas=predicho) ===")

    # Encabezado con nombres de clases
    col_width = max(len(w) for w in used_words) + 2
    header = " " * col_width + "  ".join(f"{w:>{col_width}}" for w in used_words)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>{col_width}}" for v in row)
        print(f"{used_words[i]:<{col_width}}{row_str}")

    # --- Accuracy global ---
    acc = np.sum(y_pred == y_val) / len(y_val)
    print(f"\nVal accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # --- Clases con peor desempeño (útil para saber qué palabras reentrenar) ---
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\n=== Accuracy por clase ===")
    for word, acc_c in sorted(zip(used_words, per_class_acc), key=lambda x: x[1]):
        bar = "█" * int(acc_c * 20)
        print(f"  {word:<20} {acc_c:.2f}  {bar}")


if __name__ == "__main__":
    main()