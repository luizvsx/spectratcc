import pytest
from unittest.mock import MagicMock

import supervision as sv

from tcc_common.pass_detector import PassDetector
from tcc_common.possession import get_ball_possession

from tests.fixtures.fake_data import (
    create_player,
    create_ball,
    merge_players
)


def test_pipeline_with_noise():
    """
    Simula ruído real:

    - bola some
    - bola reaparece
    - jogador muda depois

    Esperado:
    - sistema não troca posse erroneamente
    """

    mock_yolo = MagicMock()

    frames = list(range(1, 9))

    detections_per_frame = {
        # jogador 1 com bola
        1: (merge_players(create_player(1, 100, 100)), create_ball(102, 102)),

        # bola continua
        2: (merge_players(create_player(1, 100, 100)), create_ball(105, 105)),

        # 🔥 bola some (ruído)
        3: (merge_players(create_player(1, 100, 100)), None),

        # ainda sem bola
        4: (merge_players(create_player(1, 100, 100)), None),

        # bola volta perto do mesmo jogador
        5: (merge_players(create_player(1, 100, 100)), create_ball(103, 103)),

        # bola se move
        6: (merge_players(create_player(1, 100, 100)), create_ball(150, 150)),

        # jogador 2 recebe
        7: (merge_players(create_player(2, 200, 200)), create_ball(202, 202)),

        # mantém
        8: (merge_players(create_player(2, 200, 200)), create_ball(205, 205)),
    }

    def fake_predict(frame):
        return detections_per_frame[frame]

    mock_yolo.predict.side_effect = fake_predict

    detector = PassDetector(
        proximity_limit=50,
        min_frames=1,
        display_duration=10,
        referee_class_id=3,
        possession_grace_frames=2,  # 🔥 importante aqui
        distance_factor=0.7
    )

    possession_history = []
    detector_owner_history = []

    for frame in frames:
        players, ball = mock_yolo.predict(frame)

        # trata None (ruído)
        if ball is None:
            ball = sv.Detections.empty()

        player_id, _ = get_ball_possession(players, ball, 50)
        detector.update(frame, 30, players, ball)

        possession_history.append(player_id)
        detector_owner_history.append(
            detector.posse_atual[0] if detector.posse_atual else None
        )

    # 🔥 validações importantes

    # começa com jogador 1
    assert possession_history[0] == 1

    # NÃO deve trocar durante ausência da bola
    assert possession_history[2:4] == [None, None]
    assert detector_owner_history[2:4] == [1, 1]

    # depois troca corretamente pro jogador 2
    assert possession_history[-1] == 2
