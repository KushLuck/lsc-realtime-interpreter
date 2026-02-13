# src/train/model_gru.py
import tensorflow as tf


def build_gru_model(timesteps: int, n_features: int, n_classes: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(timesteps, n_features))

    x = tf.keras.layers.GRU(128, return_sequences=True)(inp)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.GRU(128)(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
