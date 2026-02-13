from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def load_npz_samples(samples_root: Path):
    X_list, y_list, paths = [], [], []
    for npz_path in samples_root.rglob("*.npz"):
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"]  # [T, F]
        y = int(data["y"])

        # Features mejoradas:
        # - mean: información promedio
        # - std: variabilidad
        # - delta_mean: "energía" de movimiento (cambio entre frames)
        mean = X.mean(axis=0)
        std = X.std(axis=0)

        dX = np.diff(X, axis=0)  # [T-1, F]
        delta_mean = np.abs(dX).mean(axis=0)

        feat = np.concatenate([mean, std, delta_mean], axis=0)

        X_list.append(feat)
        y_list.append(y)
        paths.append(str(npz_path))

    return np.stack(X_list), np.array(y_list), paths


def main():
    repo_root = Path(__file__).resolve().parents[2]
    samples_root = repo_root / "data" / "raw" / "samples"
    out_dir = repo_root / "models_artifacts" / "v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, paths = load_npz_samples(samples_root)

    if len(np.unique(y)) < 2:
        raise RuntimeError("Necesitas al menos 2 clases distintas (por ejemplo hola y adios) para entrenar.")

    # Split simple
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy (test): {acc:.3f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Guardar modelo
    model_path = out_dir / "quick_lr.pkl"
    joblib.dump(clf, model_path)
    print(f"[OK] Modelo guardado en: {model_path}")


if __name__ == "__main__":
    main()
