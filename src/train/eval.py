# src/train/eval.py
import os
import json
import numpy as np
import tensorflow as tf
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

    - ❌ Segunda versión: replicaba el split con train_test_split(X, y, ...)
      usando el mismo random_state que train.py. PERO train.py particiona
      un array de ÍNDICES (train_test_split(idx_all, ...)), mientras que
      esta versión particionaba X, y directamente. Con firmas distintas,
      sklearn NO garantiza la misma permutación aunque el random_state
      coincida → el "set de validación" podía incluir muestras de
      entrenamiento, reintroduciendo el sesgo optimista que se quería evitar.

    - ✅ Corregido: se cargan los índices EXACTOS guardados por train.py en
      models/split_indices.json. Esto garantiza que eval.py evalúe sobre
      exactamente las mismas muestras que train.py apartó como validación,
      sin ambigüedad de permutación.

      Requisito: los .npy no deben cambiar entre el entrenamiento y la
      evaluación (dataset.py carga con sorted(glob(...)), así que el orden
      es determinista mientras los archivos sean los mismos). Flujo correcto:
      recapturar → train → eval, en ese orden.
    """
    # ------------------------------------------------------------------
    # 1. Carga del dataset completo (mismo orden determinista que train.py)
    # ------------------------------------------------------------------
    X, y, used_words, _ = load_dataset()

    N, T, D = X.shape
    print(f"Dataset total: N={N}, T={T}, D={D}, clases={len(used_words)}")
    print(f"Clases: {used_words}\n")

    # ------------------------------------------------------------------
    # 2. ✅ Carga de los índices EXACTOS del split guardados por train.py
    # ------------------------------------------------------------------
    split_path = os.path.join("models", "split_indices.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"No se encontró '{split_path}'.\n"
            "Ejecuta primero: python -m src.train.train"
        )
    with open(split_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    # Chequeo defensivo: si el número total de muestras cambió respecto al
    # entrenamiento, los índices ya no son válidos (se agregaron/borraron .npy).
    if split_info.get("N_total") != N:
        raise RuntimeError(
            f"El dataset cambió desde el entrenamiento "
            f"(N actual={N}, N en split={split_info.get('N_total')}).\n"
            "Los índices del split ya no son válidos. Reentrena antes de evaluar:\n"
            "  python -m src.train.train"
        )

    idx_val = np.array(split_info["idx_val"], dtype=np.intp)
    X_val, y_val = X[idx_val], y[idx_val]

    print(f"Muestras de validación: {len(X_val)} ({len(X_val)/N*100:.1f}% del total)\n")

    # ------------------------------------------------------------------
    # 3. Carga del modelo entrenado
    # ------------------------------------------------------------------
    model_path = os.path.join("models", "lsc_v0.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo en '{model_path}'.\n"
            "Ejecuta primero: python -m src.train.train"
        )

    model = tf.keras.models.load_model(model_path)

    # ------------------------------------------------------------------
    # 4. Inferencia sobre el split de validación
    # ------------------------------------------------------------------
    probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # ------------------------------------------------------------------
    # 5. Reporte de clasificación
    # ------------------------------------------------------------------
    print("=== Classification report (validación) ===")
    print(
        classification_report(
            y_val,
            y_pred,
            target_names=used_words,
            zero_division=0,  # evita warning si alguna clase no aparece en val
        )
    )

    # ------------------------------------------------------------------
    # 6. Matriz de confusión
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_val, y_pred, labels=np.arange(len(used_words)))
    print("=== Confusion matrix (filas=real, columnas=predicho) ===")

    col_width = max(len(w) for w in used_words) + 2
    header = " " * col_width + "  ".join(f"{w:>{col_width}}" for w in used_words)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>{col_width}}" for v in row)
        print(f"{used_words[i]:<{col_width}}{row_str}")

    # ------------------------------------------------------------------
    # 7. Accuracy global
    # ------------------------------------------------------------------
    acc = np.sum(y_pred == y_val) / len(y_val)
    print(f"\nVal accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # ------------------------------------------------------------------
    # 8. Accuracy por clase (útil para saber qué palabras reentrenar)
    # ------------------------------------------------------------------
    # np.errstate evita el warning de división por cero si una clase no
    # aparece en el set de validación (su fila de la matriz suma 0).
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
    per_class_acc = np.nan_to_num(per_class_acc)  # NaN -> 0 para clases ausentes

    print("\n=== Accuracy por clase ===")
    for word, acc_c in sorted(zip(used_words, per_class_acc), key=lambda x: x[1]):
        bar = "█" * int(acc_c * 20)
        print(f"  {word:<20} {acc_c:.2f}  {bar}")


if __name__ == "__main__":
    main()