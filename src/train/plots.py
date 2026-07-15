"""Genera gráficas y métricas reproducibles del modelo entrenado.

Usa el historial, el split de validación y el modelo existentes. No entrena ni
modifica el modelo. Los resultados se guardan en ``results/``.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.decomposition import PCA

from src.train.dataset import load_dataset


MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODEL_PATH = MODELS_DIR / "lsc_v0.keras"
HISTORY_PATH = MODELS_DIR / "history.json"
SPLIT_PATH = MODELS_DIR / "split_indices.json"


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()


def _plot_training(history: dict, best_epoch: int) -> None:
    epochs = np.arange(1, len(history["loss"]) + 1)

    plt.figure(figsize=(9, 5.2))
    plt.plot(epochs, history["accuracy"], label="Entrenamiento", linewidth=2)
    plt.plot(epochs, history["val_accuracy"], label="Validación", linewidth=2)
    plt.axvline(best_epoch, color="black", linestyle="--", alpha=0.55,
                label=f"Mejor epoch: {best_epoch}")
    plt.title("Accuracy durante el entrenamiento")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.03)
    plt.grid(alpha=0.25)
    plt.legend()
    _save_figure(RESULTS_DIR / "training_accuracy.png")

    plt.figure(figsize=(9, 5.2))
    plt.plot(epochs, history["loss"], label="Entrenamiento", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Validación", linewidth=2)
    plt.axvline(best_epoch, color="black", linestyle="--", alpha=0.55,
                label=f"Mejor epoch: {best_epoch}")
    plt.title("Pérdida durante el entrenamiento")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    _save_figure(RESULTS_DIR / "training_loss.png")


def _plot_confusion_matrix(cm: np.ndarray, words: list[str]) -> None:
    labels = [word.replace("_", " ") for word in words]
    fig, ax = plt.subplots(figsize=(16, 14))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title="Matriz de confusión — validación",
        xlabel="Palabra predicha",
        ylabel="Palabra real",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2 if cm.size else 0
    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            value = int(cm[row, column])
            if value:
                ax.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=8,
                )
    _save_figure(RESULTS_DIR / "confusion_matrix.png")


def _plot_class_accuracy(cm: np.ndarray, words: list[str]) -> np.ndarray:
    totals = cm.sum(axis=1)
    per_class = np.divide(
        cm.diagonal(), totals, out=np.zeros_like(totals, dtype=float), where=totals != 0
    )
    order = np.argsort(per_class)
    labels = [words[index].replace("_", " ") for index in order]
    values = per_class[order] * 100

    plt.figure(figsize=(10, 8))
    bars = plt.barh(labels, values, color="#3478b8")
    plt.xlim(0, 105)
    plt.xlabel("Accuracy por clase (%)")
    plt.title("Rendimiento por palabra — validación")
    plt.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(value + 0.7, bar.get_y() + bar.get_height() / 2,
                 f"{value:.1f}%", va="center", fontsize=9)
    _save_figure(RESULTS_DIR / "accuracy_per_class.png")
    return per_class


def _plot_sample_distribution(y: np.ndarray, words: list[str]) -> np.ndarray:
    counts = np.bincount(y, minlength=len(words))
    labels = [word.replace("_", " ") for word in words]

    plt.figure(figsize=(12, 6.5))
    bars = plt.bar(labels, counts, color="#3b9b78")
    plt.title(f"Distribución del dataset — {len(y)} muestras")
    plt.xlabel("Palabra")
    plt.ylabel("Número de muestras")
    plt.xticks(rotation=55, ha="right")
    plt.grid(axis="y", alpha=0.25)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, count + 1, str(int(count)),
                 ha="center", va="bottom", fontsize=8)
    _save_figure(RESULTS_DIR / "samples_per_class.png")
    return counts


def _draw_flow(
    title: str,
    steps: list[tuple[str, str]],
    filename: str,
    colors: list[str],
) -> None:
    """Dibuja un flujo horizontal limpio para usar directamente en diapositivas."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=22, weight="bold", pad=20)

    count = len(steps)
    box_width = min(0.15, 0.80 / count)
    gap = (0.92 - count * box_width) / (count - 1)
    x_positions = [0.04 + index * (box_width + gap) for index in range(count)]
    for index, ((heading, detail), x) in enumerate(zip(steps, x_positions)):
        box = FancyBboxPatch(
            (x, 0.34), box_width, 0.34,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor=colors[index], edgecolor="white", linewidth=2,
        )
        ax.add_patch(box)
        ax.text(x + box_width / 2, 0.55, heading, ha="center", va="center",
                color="white", fontsize=13, weight="bold")
        ax.text(x + box_width / 2, 0.43, detail, ha="center", va="center",
                color="white", fontsize=9.5, wrap=True)
        if index < count - 1:
            next_x = x_positions[index + 1]
            ax.annotate(
                "", xy=(next_x - 0.008, 0.51), xytext=(x + box_width + 0.008, 0.51),
                arrowprops=dict(arrowstyle="-|>", color="#354052", lw=2.2),
            )
    _save_figure(RESULTS_DIR / filename)


def _plot_pipeline() -> None:
    steps = [
        ("CÁMARA", "Video 1280 × 720"),
        ("MEDIAPIPE", "Pose + manos"),
        ("KEYPOINTS", "258 características"),
        ("SECUENCIA", "15 frames"),
        ("RED GRU", "Clasificación"),
        ("SALIDA", "20 palabras + voz"),
    ]
    colors = ["#2c5282", "#2b6cb0", "#3182ce", "#319795", "#2f855a", "#276749"]
    _draw_flow("Pipeline del intérprete LSC", steps, "pipeline.png", colors)


def _plot_architecture(model: tf.keras.Model) -> None:
    steps = [
        ("ENTRADA", "15 × 258"),
        ("GRU", "128 unidades"),
        ("BN + DROPOUT", "0.30"),
        ("GRU", "64 unidades"),
        ("BN + DROPOUT", "0.30"),
        ("DENSE", "64 + ReLU"),
        ("SOFTMAX", "20 clases"),
    ]
    colors = ["#4a5568", "#553c9a", "#6b46c1", "#805ad5", "#6b46c1", "#d53f8c", "#c53030"]
    _draw_flow("Arquitectura de la red neuronal GRU", steps, "model_architecture.png", colors)

    # Añade el número total de parámetros como pie de imagen sin sobrecargarla.
    path = RESULTS_DIR / "model_architecture.png"
    image = plt.imread(path)
    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.imshow(image)
    ax.axis("off")
    ax.text(0.5, 0.02, f"Parámetros totales: {model.count_params():,}",
            transform=ax.transAxes, ha="center", fontsize=12, color="#354052")
    _save_figure(path)


def _plot_confidence(probabilities: np.ndarray, y_val: np.ndarray, y_pred: np.ndarray) -> None:
    confidence = probabilities.max(axis=1)
    correct = y_pred == y_val
    bins = np.linspace(0, 1, 21)

    plt.figure(figsize=(10, 5.8))
    plt.hist(confidence[correct], bins=bins, alpha=0.82, color="#2f855a",
             label=f"Correctas ({correct.sum()})")
    if np.any(~correct):
        plt.hist(confidence[~correct], bins=bins, alpha=0.9, color="#c53030",
                 label=f"Incorrectas ({(~correct).sum()})")
    plt.axvline(0.70, color="#2d3748", linestyle="--", linewidth=2.2,
                label="Threshold del live: 0.70")
    plt.title("Confianza de las predicciones — validación")
    plt.xlabel("Probabilidad de la clase predicha")
    plt.ylabel("Número de muestras")
    plt.xlim(0, 1.01)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    _save_figure(RESULTS_DIR / "confidence_distribution.png")


def _plot_pca(model: tf.keras.Model, X_val: np.ndarray, y_val: np.ndarray,
              words: list[str]) -> None:
    # La capa Dense de 64 unidades contiene la representación aprendida antes
    # del clasificador Softmax.
    embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    embeddings = embedding_model.predict(X_val, verbose=0)
    pca = PCA(n_components=2)
    points = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum() * 100

    plt.figure(figsize=(14, 8))
    palette = plt.cm.tab20(np.linspace(0, 1, len(words)))
    for index, word in enumerate(words):
        mask = y_val == index
        plt.scatter(points[mask, 0], points[mask, 1], s=45, alpha=0.82,
                    color=palette[index], label=word.replace("_", " "),
                    edgecolors="white", linewidths=0.35)
    plt.title("Mapa PCA de las representaciones aprendidas")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.grid(alpha=0.18)
    plt.legend(ncol=1, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.figtext(0.43, 0.01, f"Varianza representada en 2D: {explained:.1f}%",
                ha="center", fontsize=9, color="#4a5568")
    _save_figure(RESULTS_DIR / "learned_features_pca.png")


def main() -> None:
    for required in (MODEL_PATH, HISTORY_PATH, SPLIT_PATH):
        if not required.exists():
            raise FileNotFoundError(f"No se encontró el artefacto requerido: {required}")

    RESULTS_DIR.mkdir(exist_ok=True)
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    X, y, words, _ = load_dataset()

    if split.get("N_total") != len(X):
        raise RuntimeError("El dataset cambió desde el entrenamiento; reentrena antes de graficar.")

    idx_val = np.asarray(split["idx_val"], dtype=np.intp)
    X_val, y_val = X[idx_val], y[idx_val]
    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    labels = np.arange(len(words))
    cm = confusion_matrix(y_val, y_pred, labels=labels)

    best_index = int(np.argmin(history["val_loss"]))
    best_epoch = best_index + 1
    _plot_training(history, best_epoch)
    _plot_confusion_matrix(cm, words)
    per_class_accuracy = _plot_class_accuracy(cm, words)
    sample_counts = _plot_sample_distribution(y, words)
    _plot_pipeline()
    _plot_architecture(model)
    _plot_confidence(probabilities, y_val, y_pred)
    _plot_pca(model, X_val, y_val, words)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, labels=labels, average="macro", zero_division=0
    )
    report = classification_report(
        y_val, y_pred, labels=labels, target_names=words,
        output_dict=True, zero_division=0,
    )
    metrics = {
        "dataset_samples": int(len(X)),
        "validation_samples": int(len(X_val)),
        "classes": len(words),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "best_epoch": best_epoch,
        "best_val_loss": float(history["val_loss"][best_index]),
        "best_val_accuracy": float(history["val_accuracy"][best_index]),
        "classification_report": report,
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (RESULTS_DIR / "metrics_per_class.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["word", "dataset_samples", "validation_samples", "accuracy"])
        for index, word in enumerate(words):
            writer.writerow([
                word,
                int(sample_counts[index]),
                int(cm[index].sum()),
                f"{per_class_accuracy[index]:.6f}",
            ])

    print(f"Resultados generados en: {RESULTS_DIR.resolve()}")
    print(f"Accuracy: {metrics['accuracy']:.4%} | Macro F1: {metrics['macro_f1']:.4%}")


if __name__ == "__main__":
    main()
