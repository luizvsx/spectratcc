import pytest
from unittest.mock import MagicMock

from tcc_common.pass_detector import PassDetector
from tcc_common.possession import get_ball_possession

from tests.fixtures.fake_data import (
    create_player,
    create_ball,
    merge_players
)


def test_full_game_simulation():
    """
    Simula uma sequência maior de frames (mini jogo)

    Cenário:
    - Jogador 1 começa com a bola
    - Bola se movimenta
    - Jogador 2 recebe
    - Jogador 2 mantém posse
    - Jogador 3 recebe depois

    Esperado:
    - Sistema acompanha mudanças corretamente
    """

    mock_yolo = MagicMock()

    # 🔥 sequência mais longa
    frames = list(range(1, 11))

    detections_per_frame = {
        # Jogador 1 com bola
        1: (merge_players(create_player(1, 100, 100)), create_ball(102, 102)),
        2: (merge_players(create_player(1, 100, 100)), create_ball(105, 105)),

        # bola em movimento
        3: (merge_players(create_player(1, 100, 100)), create_ball(140, 140)),

        # jogador 2 recebe
        4: (merge_players(create_player(2, 200, 200)), create_ball(202, 202)),
        5: (merge_players(create_player(2, 200, 200)), create_ball(205, 205)),

        # mantém posse
        6: (merge_players(create_player(2, 200, 200)), create_ball(207, 207)),

        # nova movimentação
        7: (merge_players(create_player(2, 200, 200)), create_ball(250, 250)),

        # jogador 3 recebe
        8: (merge_players(create_player(3, 300, 300)), create_ball(302, 302)),
        9: (merge_players(create_player(3, 300, 300)), create_ball(305, 305)),

        # mantém posse
        10: (merge_players(create_player(3, 300, 300)), create_ball(307, 307)),
    }

    def fake_predict(frame):
        return detections_per_frame[frame]

    mock_yolo.predict.side_effect = fake_predict

    detector = PassDetector(
        proximity_limit=50,
        min_frames=1,
        display_duration=10,
        referee_class_id=3,
        possession_grace_frames=2,
        distance_factor=0.7
    )

    possession_history = []

    for frame in frames:
        players, ball = mock_yolo.predict(frame)

        player_id, _ = get_ball_possession(players, ball, 50)
        detector.update(frame, 30, players, ball)

        possession_history.append(player_id)

    # 🔥 validações importantes

    # começa com jogador 1
    assert possession_history[0] == 1

    # em algum momento muda pra 2
    assert 2 in possession_history

    # depois muda pra 3
    assert 3 in possession_history

    # último frame deve ser jogador 3
    assert possession_history[-1] == 3
