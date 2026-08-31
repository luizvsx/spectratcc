import pytest
from unittest.mock import MagicMock
import random

from tcc_common.pass_detector import PassDetector
from tcc_common.possession import get_ball_possession

from tests.fixtures.fake_data import (
    create_player,
    create_ball,
    merge_players
)


def test_pipeline_stress():
    """
    Simula muitos frames (tipo jogo longo)

    Esperado:
    - não quebrar
    - sempre retornar valores válidos
    """

    random.seed(42)
    mock_yolo = MagicMock()

    num_frames = 100

    def fake_predict(frame):
        # jogador aleatório
        player_id = random.randint(1, 5)

        player = create_player(player_id, 100 + player_id * 50, 100)

        # pequena variação da bola
        ball = create_ball(100 + player_id * 50 + random.randint(-5, 5), 100)

        return merge_players(player), ball

    mock_yolo.predict.side_effect = fake_predict

    detector = PassDetector(
        proximity_limit=60,
        min_frames=1,
        display_duration=10,
        referee_class_id=3,
        possession_grace_frames=2,
        distance_factor=0.7
    )

    for frame in range(1, num_frames + 1):
        players, ball = mock_yolo.predict(frame)

        player_id, _ = get_ball_possession(players, ball, 60)
        detector.update(frame, 30, players, ball)

        # 🔥 validações básicas de sanidade
        assert player_id is None or player_id >= 0

    summary = detector.possession_summary()
    assert summary["frames_total"] == num_frames
    assert len(detector.passes_list) + len(detector.shots_list) <= num_frames - 1
