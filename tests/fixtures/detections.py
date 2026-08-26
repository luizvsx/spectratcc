"""Construtores determinísticos de detecções usados pela suíte."""

import numpy as np
import supervision as sv


def player(
    tracker_id: int,
    x: float,
    y: float = 100,
    *,
    team_id: int = 0,
    width: float = 20,
    height: float = 40,
) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[x, y, x + width, y + height]], dtype=float),
        class_id=np.array([team_id], dtype=int),
        confidence=np.array([0.95], dtype=float),
        tracker_id=np.array([tracker_id], dtype=int),
    )


def ball(center_x: float, center_y: float = 140, *, size: float = 6) -> sv.Detections:
    half = size / 2
    return sv.Detections(
        xyxy=np.array(
            [[center_x - half, center_y - half, center_x + half, center_y + half]],
            dtype=float,
        ),
        class_id=np.array([0], dtype=int),
        confidence=np.array([0.95], dtype=float),
    )


def players(*items: sv.Detections) -> sv.Detections:
    return sv.Detections.merge(list(items)) if items else sv.Detections.empty()


def foot_x(x: float, width: float = 20) -> float:
    """Coordenada horizontal do ponto inferior central da caixa."""
    return x + width / 2
