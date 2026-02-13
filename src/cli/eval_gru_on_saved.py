from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from src.models.gru import GRUClassifier

def main():
    repo_root = Path(__file__).resolve().parents[2]
    samples_root = repo_root / "data" / "raw" / "samples"
    ckpt_path = repo_root / "models_artifacts" / "v1" / "gru.pt"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    F_dim = int(ckpt["F"])
    T_dim = int(ckpt["T"])
    mean = ckpt["mean"].astype(np.float32)
    std = ckpt["std"].astype(np.float32)

    model = GRUClassifier(input_dim=F_dim, hidden_dim=128, num_layers=1, num_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    confs = {0: [], 1: []}
    for p in samples_root.rglob("*.npz"):
        d = np.load(p, allow_pickle=True)
        y = int(d["y"])
        if y not in (0, 1):
            continue

        X = d["X"].astype(np.float32)
        X = X[:T_dim] if X.shape[0] >= T_dim else np.vstack([X, np.zeros((T_dim - X.shape[0], X.shape[1]), np.float32)])
        X = (X - mean) / std

        Xt = torch.from_numpy(X).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(model(Xt), dim=1).numpy()[0]
        confs[y].append(float(probs[y]))

    print("Conf promedio en datos guardados:")
    for k in (0, 1):
        if confs[k]:
            print(f"  clase {k}: mean={np.mean(confs[k]):.3f}  min={np.min(confs[k]):.3f}  max={np.max(confs[k]):.3f}  n={len(confs[k])}")
        else:
            print(f"  clase {k}: sin datos")

if __name__ == "__main__":
    main()
