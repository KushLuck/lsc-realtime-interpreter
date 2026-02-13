from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from src.models.gru import GRUClassifier


def stratified_split(paths, labels, val_ratio=0.25, seed=42):
    rng = np.random.default_rng(seed)
    paths = np.array(paths)
    labels = np.array(labels)

    train_idx, val_idx = [], []
    for cls in np.unique(labels):
        idx_cls = np.where(labels == cls)[0]
        rng.shuffle(idx_cls)
        n_val = max(1, int(len(idx_cls) * val_ratio))
        val_idx.extend(idx_cls[:n_val].tolist())
        train_idx.extend(idx_cls[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def compute_norm_stats(paths):
    """
    Calcula mean/std por feature usando SOLO train.
    X: [T, F] -> juntamos todos los frames de todos los samples.
    """
    all_frames = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        X = d["X"].astype(np.float32)  # [T, F]
        all_frames.append(X)

    Z = np.concatenate(all_frames, axis=0)  # [Nframes, F]
    mean = Z.mean(axis=0)
    std = Z.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)  # evita división por 0
    return mean.astype(np.float32), std.astype(np.float32)


class SubsetDataset(Dataset):
    def __init__(self, paths, mean=None, std=None):
        self.paths = paths
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        d = np.load(self.paths[i], allow_pickle=True)
        X = d["X"].astype(np.float32)  # [T, F]
        y = int(d["y"])

        if self.mean is not None and self.std is not None:
            X = (X - self.mean) / self.std

        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)


def main():
    repo_root = Path(__file__).resolve().parents[2]
    samples_root = repo_root / "data" / "raw" / "samples"
    out_dir = repo_root / "models_artifacts" / "v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_ids = {0, 1}  # hola/adios

    all_paths, all_labels = [], []
    for p in samples_root.rglob("*.npz"):
        y = int(np.load(p, allow_pickle=True)["y"])
        if y in allowed_ids:
            all_paths.append(p)
            all_labels.append(y)

    if len(all_paths) < 20:
        raise RuntimeError("Muy pocas muestras para GRU. Graba al menos 10 por clase (20 total).")

    tr_idx, val_idx = stratified_split(all_paths, all_labels, val_ratio=0.25, seed=42)
    train_paths = [all_paths[i] for i in tr_idx]
    val_paths = [all_paths[i] for i in val_idx]

    y_tr = [int(np.load(p, allow_pickle=True)["y"]) for p in train_paths]
    y_va = [int(np.load(p, allow_pickle=True)["y"]) for p in val_paths]
    print(f"Train size={len(train_paths)} | class_counts={np.bincount(y_tr)}")
    print(f"Val   size={len(val_paths)} | class_counts={np.bincount(y_va)}")

    # ✅ Normalización (solo con train)
    mean, std = compute_norm_stats(train_paths)

    train_ds = SubsetDataset(train_paths, mean=mean, std=std)
    val_ds = SubsetDataset(val_paths, mean=mean, std=std)

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

    X0, _ = train_ds[0]
    T, F = X0.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GRUClassifier(input_dim=F, hidden_dim=128, num_layers=1, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)  # un toque más alto ayuda

    best_val = -1.0
    best_path = out_dir / "gru.pt"

    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0

        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()

        val_acc = correct / max(total, 1)
        print(f"Epoch {epoch:02d} | train_loss={total_loss/len(train_dl):.4f} | val_acc={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "F": F,
                    "T": T,
                    "mean": mean,
                    "std": std,
                },
                best_path,
            )

    print(f"[OK] Mejor val_acc={best_val:.3f} | Modelo guardado en {best_path}")


if __name__ == "__main__":
    main()
