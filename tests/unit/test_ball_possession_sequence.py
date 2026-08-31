import supervision as sv

from tcc_common.possession import get_ball_possession
from tests.fixtures.fake_data import create_player, create_ball, merge_players


def test_possession_sequence_changes():
    """
    Jogador A começa com a bola, depois jogador B assume.
    """
    player_a = create_player(1, 100, 100)
    player_b = create_player(2, 300, 300)

    players = merge_players(player_a, player_b)

    # Frame 1 → bola com A
    ball = create_ball(102, 102)
    id1, _ = get_ball_possession(players, ball, 50)

    # Frame 2 → bola ainda com A
    ball = create_ball(105, 105)
    id2, _ = get_ball_possession(players, ball, 50)

    # Frame 3 → bola vai pra B
    ball = create_ball(302, 302)
    id3, _ = get_ball_possession(players, ball, 50)

    assert id1 == 1
    assert id2 == 1
    assert id3 == 2


def test_possession_lost_when_ball_far():
    """
    Jogador tinha posse e perde porque bola fica longe.
    """
    player = create_player(1, 100, 100)

    # Frame 1 → com posse
    ball = create_ball(102, 102)
    id1, _ = get_ball_possession(player, ball, 50)

    # Frame 2 → bola longe
    ball = create_ball(500, 500)
    id2, _ = get_ball_possession(player, ball, 50)

    assert id1 == 1
    assert id2 is None


def test_possession_with_multiple_frames_noise():
    """
    Bola some (simula falha de detecção), mas não deve atribuir posse errada.
    """
    player_a = create_player(1, 100, 100)
    player_b = create_player(2, 200, 200)

    players = merge_players(player_a, player_b)

    # Frame 1 → A com bola
    ball = create_ball(102, 102)
    id1, _ = get_ball_possession(players, ball, 50)

    # Frame 2 → sem bola (usa empty)
    empty_ball = sv.Detections.empty()
    id2, _ = get_ball_possession(players, empty_ball, 50)

    # Frame 3 → bola volta perto de A
    ball = create_ball(104, 104)
    id3, _ = get_ball_possession(players, ball, 50)

    assert id1 == 1
    assert id2 is None
    assert id3 == 1


def test_no_false_switch_between_players():
    """
    Bola fica entre dois jogadores, mas mais próxima de A → não deve trocar.
    """
    player_a = create_player(1, 100, 100)
    player_b = create_player(2, 130, 130)

    players = merge_players(player_a, player_b)

    # Bola mais perto de A
    ball = create_ball(110, 110)

    id_, _ = get_ball_possession(players, ball, 50)

    assert id_ == 1
