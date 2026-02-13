# src/train/eval.py
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from src.train.dataset import load_dataset


def main():
    X, y, used_words, _ = load_dataset()

    model = tf.keras.models.load_model(os.path.join("models", "lsc_v0.keras"))
    probs = model.predict(X, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    print("\n=== Classification report ===")
    print(classification_report(y, y_pred, target_names=used_words))

    cm = confusion_matrix(y, y_pred)
    print("\n=== Confusion matrix (rows=true, cols=pred) ===")
    print(cm)


if __name__ == "__main__":
    main()
