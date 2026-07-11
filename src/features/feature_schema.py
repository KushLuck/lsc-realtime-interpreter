# src/features/feature_schema.py
from dataclasses import dataclass


@dataclass(frozen=True)
class SchemaV1:
    # Pose (33*4) + Hands (21*3 + 21*3) = 258
    # NOTA: a diferencia del repo de referencia (ronvidev/modelo_lstm_lsp),
    # que incluye los 468 landmarks de cara (1662 features totales), este
    # schema EXCLUYE la cara deliberadamente. Para señas manuales la cara
    # aporta 1404 features de ruido en su mayoría, hace el modelo más pesado
    # y más lento. 258 features = pipeline más liviano y mejor para CPU.
    name: str = "v1_pose_hands"
    n_features: int = 258

<<<<<<< HEAD
# Para v0 recomiendo 30 frames por seña (más robusto que 15)
MODEL_FRAMES = 30
MIN_LENGTH_FRAMES = 8   # mínimo real antes de aceptar muestra (sin contar margen)
MARGIN_FRAMES = 2
DELAY_FRAMES = 10
=======

# ------------------------------------------------------------------
# Parámetros de captura / inferencia
# ------------------------------------------------------------------
# Calibrados tomando como referencia los valores probados del repo original
# (MODEL_FRAMES=15, min_cant_frames=5, margin_frame=1, delay_frames=3), que
# fueron pensados para funcionar en CPU. Se ajustan ligeramente para tolerar
# el parpadeo de detección observado en este hardware (~20 fps).

# Número de frames al que se remuestrea CADA seña antes de entrar al modelo.
# ⬇️ Bajado de 30 a 15 (como el original):
#   - A ~20 fps, 15 frames = ~0.75s de seña: alcanzable sin interpolar de más.
#   - Menos timesteps = modelo más liviano = mejores FPS en entrenamiento e
#     inferencia.
#   - Menos descartes por longitud insuficiente.
# ⚠️ Cambiar este valor CAMBIA la forma del input del modelo (MODEL_FRAMES, 258).
#    Si ya hay un modelo entrenado con otro valor, HAY QUE REENTRENAR.
MODEL_FRAMES = 15

# Longitud mínima real (sin contar margen) para aceptar una muestra.
# ⬇️ Bajado a 5 (como el original): permisivo con señas cortas, reduce descartes.
MIN_LENGTH_FRAMES = 5

# Frames ignorados al inicio (colchón mientras las manos entran al encuadre).
MARGIN_FRAMES = 1

# Frames de gracia tras perder las manos antes de cerrar la seña.
# ⬆️ Subido de 3 (original) a 5: da ~0.25s extra de tolerancia al parpadeo
#    de MediaPipe. Con el reset de fix_frames al reaparecer la mano (ver
#    capture_samples.py / live.py), esto absorbe parpadeos sin cerrar señas.
DELAY_FRAMES = 5
>>>>>>> 7feb2c932cc126039fe7b9b04d014e6c50ad8c31
