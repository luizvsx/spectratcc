import supervision as sv

from tcc_common.pass_detector import PassDetector
from tcc_common.possession import get_ball_possession
from tests.fixtures.fake_data import create_player, create_ball, merge_players


def test_full_pipeline_basic_flow():
    """
    Simula pipeline completo:

    Frame 1 → jogador A com bola
    Frame 2 → bola em movimento
    Frame 3 → jogador B recebe

    Validação:
    ✔ pipeline roda completo
    ✔ posse muda ao longo do tempo
    """

    detector = PassDetector(
        proximity_limit=50,
        min_frames=1,
        display_duration=10,
        referee_class_id=3,
        possession_grace_frames=2,
        distance_factor=0.7
    )

    last_player = None

    # Frame 1
    player_a = create_player(1, 100, 100)
    ball = create_ball(102, 102)

    players = merge_players(player_a)

    player_id, team_id = get_ball_possession(players, ball, 50)
    detector.update(1, 30, players, ball)

    assert player_id == 1
    last_player = player_id

    # Frame 2 (bola solta)
    ball = create_ball(160, 160)

    player_id, team_id = get_ball_possession(players, ball, 50)
    detector.update(2, 30, players, ball)

    # pode perder posse
    # não validamos valor fixo aqui
    last_player = player_id

    # Frame 3 (novo jogador)
    player_b = create_player(2, 200, 200)
    players = merge_players(player_b)

    ball = create_ball(202, 202)

    player_id, team_id = get_ball_possession(players, ball, 50)
    detector.update(3, 30, players, ball)

    # ✔ valida que houve mudança de posse
    assert player_id == 2


def test_pipeline_handles_no_ball():
    """
    Pipeline deve funcionar mesmo sem bola temporariamente
    """

    detector = PassDetector(
        proximity_limit=50,
        min_frames=1,
        display_duration=10,
        referee_class_id=3,
        possession_grace_frames=2,
        distance_factor=0.7
    )

    player = create_player(1, 100, 100)
    players = merge_players(player)

    # Frame 1
    ball = create_ball(102, 102)
    get_ball_possession(players, ball, 50)
    detector.update(1, 30, players, ball)

    # Frame 2 → sem bola
    empty_ball = sv.Detections.empty()
    get_ball_possession(players, empty_ball, 50)
    detector.update(2, 30, players, empty_ball)

    assert detector.posse_atual is not None
    assert detector.posse_atual[0] == 1
    assert detector.ball_missing == 1
    assert detector.possession_summary()["frames_total"] == 2


def test_pipeline_multiple_players():
    """
    Vários jogadores → escolhe o mais próximo da bola
    """

    detector = PassDetector(
        proximity_limit=50,
        min_frames=1,
        display_duration=10,
        referee_class_id=3,
        possession_grace_frames=2,
        distance_factor=0.7
    )

    player_a = create_player(1, 100, 100)
    player_b = create_player(2, 300, 300)

    players = merge_players(player_a, player_b)

    ball = create_ball(102, 102)

    player_id, _ = get_ball_possession(players, ball, 50)
    detector.update(1, 30, players, ball)

    # ✔ jogador mais próximo ganha posse
    assert player_id == 1


def test_pipeline_ignores_referee():
    """
    Árbitro não deve ter posse
    """

    detector = PassDetector(
        proximity_limit=50,
        min_frames=1,
        display_duration=10,
        referee_class_id=3,
        possession_grace_frames=2,
        distance_factor=0.7
    )

    player = create_player(1, 100, 100)
    referee = create_player(99, 102, 102, team_id=3)

    players = merge_players(player, referee)

    ball = create_ball(102, 102)

    player_id, _ = get_ball_possession(players, ball, 50)
    detector.update(1, 30, players, ball)

    # ✔ jogador, não árbitro
    assert player_id == 1


def test_possession_helper_rejects_referee_alone():
    referee = create_player(99, 100, 100, team_id=3)
    ball = create_ball(102, 102)

    player_id, team_id = get_ball_possession(referee, ball, 50)

    assert player_id is None
    assert team_id is None
