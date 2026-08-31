import pytest
from unittest.mock import MagicMock

from tcc_common.pass_detector import PassDetector
from tcc_common.possession import get_ball_possession

from tests.fixtures.fake_data import (
    create_player,
    create_ball,
    merge_players
)


def test_pipeline_with_mocked_yolo():
    """
    Testa pipeline completo simulando YOLO.

    Fluxo:
    Frame 1 → Jogador A com bola
    Frame 2 → Bola em movimento
    Frame 3 → Jogador B recebe

    Esperado:
    → troca de posse (proxy de passe)
    """

    # 🔥 Mock do modelo YOLO (não usamos ele de verdade)
    mock_yolo = MagicMock()

    # Sequência de frames simulados
    frames = [1, 2, 3]

    # Detecções fake por frame
    detections_per_frame = {
        1: (merge_players(create_player(1, 100, 100)),
            create_ball(102, 102)),

        2: (merge_players(create_player(1, 100, 100)),
            create_ball(160, 160)),  # bola "livre"

        3: (merge_players(create_player(2, 200, 200)),
            create_ball(202, 202)),
    }

    # Mock do retorno do YOLO
    def fake_predict(frame):
        return detections_per_frame[frame]

    mock_yolo.predict.side_effect = fake_predict

    # Inicializa componentes reais
    detector = PassDetector(
        proximity_limit=50,
        min_frames=1,
        display_duration=10,
        referee_class_id=3,
        possession_grace_frames=2,
        distance_factor=0.7
    )

    current_player = None

    # 🔥 pipeline simulado
    for frame in frames:
        players, ball = mock_yolo.predict(frame)

        # posse
        player_id, _ = get_ball_possession(players, ball, 50)

        # detector usa detections reais
        detector.update(frame, 30, players, ball)

        if player_id is not None:
            current_player = player_id

    # ✔ valida comportamento final (não estrutura interna!)
    assert current_player == 2
