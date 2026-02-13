# src/features/feature_schema.py
from dataclasses import dataclass

@dataclass(frozen=True)
class SchemaV1:
    # Pose (33*4) + Hands (21*3 + 21*3) = 258
    name: str = "v1_pose_hands"
    n_features: int = 258

# Para v0 recomiendo 30 frames por seña (más robusto que 15)
MODEL_FRAMES = 30
MIN_LENGTH_FRAMES = 8   # mínimo real antes de aceptar muestra (sin contar margen)
MARGIN_FRAMES = 1
DELAY_FRAMES = 3
