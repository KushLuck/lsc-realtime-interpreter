"""Componentes visuales OpenCV para captura e inferencia.

Este modulo solo dibuja sobre una copia del frame. No contiene logica de
captura, segmentacion, inferencia ni almacenamiento.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np


# Colores BGR. Una paleta unica evita estilos distintos entre pantallas.
NAVY = (31, 24, 18)
PANEL = (48, 39, 30)
WHITE = (245, 245, 245)
MUTED = (190, 181, 170)
GREEN = (124, 205, 72)
ORANGE = (66, 156, 255)
CYAN = (224, 190, 67)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _alpha_rect(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    """Dibuja un panel translucido sin modificar el tamano del frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _status_pill(frame: np.ndarray, label: str, color: tuple[int, int, int]) -> None:
    text_size, _ = cv2.getTextSize(label, FONT, 0.55, 1)
    width = text_size[0] + 42
    cv2.rectangle(frame, (18, 18), (18 + width, 52), PANEL, -1)
    cv2.circle(frame, (34, 35), 6, color, -1, cv2.LINE_AA)
    cv2.putText(frame, label, (48, 41), FONT, 0.55, WHITE, 1, cv2.LINE_AA)


def render_live_overlay(
    frame: np.ndarray,
    sentence: Sequence[str],
    status: str,
    last_word: Optional[str] = None,
    confidence: Optional[float] = None,
) -> np.ndarray:
    """Presenta el estado del interprete sin afectar el frame procesado."""
    view = frame.copy()
    height, width = view.shape[:2]
    status_color = ORANGE if status == "LEYENDO SENA" else GREEN

    _alpha_rect(view, (0, 0), (width, 70), NAVY, 0.88)
    _status_pill(view, status, status_color)
    cv2.putText(
        view, "INTERPRETE LSC", (max(18, width - 188), 41),
        FONT, 0.55, MUTED, 1, cv2.LINE_AA,
    )

    panel_top = max(80, height - 112)
    _alpha_rect(view, (0, panel_top), (width, height), NAVY, 0.90)
    cv2.putText(
        view, "TRADUCCION", (20, panel_top + 27),
        FONT, 0.45, CYAN, 1, cv2.LINE_AA,
    )
    phrase = "  |  ".join(word.replace("_", " ").upper() for word in sentence)
    if not phrase:
        phrase = "Haz una sena para comenzar"
    cv2.putText(
        view, phrase[:52], (20, panel_top + 62),
        FONT, 0.72, WHITE if sentence else MUTED, 2, cv2.LINE_AA,
    )

    detail = "Q  Salir"
    if last_word is not None and confidence is not None:
        detail = f"Ultima: {last_word.replace('_', ' ')}   Confianza: {confidence:.0%}"
    cv2.putText(
        view, detail, (20, panel_top + 91),
        FONT, 0.43, MUTED, 1, cv2.LINE_AA,
    )
    return view


def render_capture_overlay(
    frame: np.ndarray,
    word_id: str,
    status: str,
    show_landmarks: bool,
    captured_frames: int,
) -> np.ndarray:
    """Presenta el estado de recoleccion sin intervenir en sus datos."""
    view = frame.copy()
    height, width = view.shape[:2]
    status_color = ORANGE if status == "CAPTURANDO" else GREEN

    _alpha_rect(view, (0, 0), (width, 70), NAVY, 0.88)
    _status_pill(view, status, status_color)
    cv2.putText(
        view, "CAPTURA LSC", (max(18, width - 155), 41),
        FONT, 0.55, MUTED, 1, cv2.LINE_AA,
    )

    panel_top = max(80, height - 90)
    _alpha_rect(view, (0, panel_top), (width, height), NAVY, 0.90)
    cv2.putText(view, "PALABRA", (20, panel_top + 24), FONT, 0.42, CYAN, 1, cv2.LINE_AA)
    cv2.putText(
        view, word_id.replace("_", " ").upper(), (20, panel_top + 58),
        FONT, 0.78, WHITE, 2, cv2.LINE_AA,
    )
    landmarks = "ON" if show_landmarks else "OFF"
    info = f"Frames: {captured_frames}   |   L  Landmarks {landmarks}   |   Q  Salir"
    cv2.putText(view, info, (20, panel_top + 80), FONT, 0.40, MUTED, 1, cv2.LINE_AA)
    return view
