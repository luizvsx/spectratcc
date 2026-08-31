"""Utilitários para determinar a posse instantânea da bola."""

from typing import Optional, Tuple

import numpy as np
import supervision as sv


def get_ball_possession(
    player_detections: sv.Detections,
    ball_detections: sv.Detections,
    possession_threshold: float,
    referee_class_id: int = 3,
) -> Tuple[Optional[int], Optional[int]]:
    """Retorna o identificador e o time do jogador mais próximo da bola."""
    if len(ball_detections) == 0 or len(player_detections) == 0:
        return None, None

    player_detections = player_detections[
        player_detections.class_id != referee_class_id
    ]
    if len(player_detections) == 0:
        return None, None

    ball_center = ball_detections.get_anchors_coordinates(sv.Position.CENTER)[0]
    player_anchors = player_detections.get_anchors_coordinates(
        sv.Position.BOTTOM_CENTER
    )
    distances = np.linalg.norm(player_anchors - ball_center, axis=1)
    nearest = int(np.argmin(distances))

    if distances[nearest] < possession_threshold:
        return (
            player_detections.tracker_id[nearest],
            player_detections.class_id[nearest],
        )
    return None, None
