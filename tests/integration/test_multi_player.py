import pytest
from unittest.mock import MagicMock

from tcc_common.pass_detector import PassDetector
from tcc_common.possession import get_ball_possession

from tests.fixtures.fake_data import (
    create_player,
    create_ball,
    merge_players
)


def test_multiple_players_nearest_gets_possession():
    """
    Vários jogadores em campo → o mais próximo da bola deve ter posse
    """

    mock_yolo = MagicMock()

    frames = [1, 2]

    detections_per_frame = {
        1: (
            merge_players(
                create_player(1, 100, 100),
                create_player(2, 200, 200),
                create_player(3, 300, 300),
            ),
            create_ball(105, 105),  # mais perto do jogador 1
        ),

        2: (
            merge_players(
                create_player(1, 100, 100),
                create_player(2, 200, 200),
                create_player(3, 300, 300),
            ),
            create_ball(198, 198),  # mais perto do jogador 2
        ),
    }

    def fake_predict(frame):
        return detections_per_frame[frame]

    mock_yolo.predict.side_effect = fake_predict

    detector = PassDetector(
        proximity_limit=80,
        min_frames=1,
        display_duration=10,
        referee_class_id=3,
        possession_grace_frames=2,
        distance_factor=0.7
    )

    possession_history = []

    for frame in frames:
        players, ball = mock_yolo.predict(frame)

        player_id, _ = get_ball_possession(players, ball, 80)
        detector.update(frame, 30, players, ball)

        possession_history.append(player_id)

    assert possession_history[0] == 1
    assert possession_history[1] == 2
