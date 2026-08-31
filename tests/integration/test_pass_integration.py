import supervision as sv

from tcc_common.pass_detector import PassDetector
from tests.fixtures.fake_data import create_player, create_ball, merge_players


def test_pass_flow_records_event():
    """
    Fluxo completo de passe:
    A → bola livre → B

    Validação atual:
    ✔ não deve crashar
    ✔ pipeline executa completo
    """

    detector = PassDetector(
        proximity_limit=50,
        min_frames=1,
        display_duration=10,
        referee_class_id=3,
        possession_grace_frames=2,
        distance_factor=0.7
    )

    # Frame 1
    player_a = create_player(1, 100, 100)
    ball = create_ball(102, 102)
    detector.update(1, 30, merge_players(player_a), ball)

    # Frame 2
    ball = create_ball(160, 160)
    detector.update(2, 30, merge_players(player_a), ball)

    # Frame 3
    player_b = create_player(2, 200, 200)
    ball = create_ball(202, 202)
    detector.update(3, 30, merge_players(player_b), ball)

    assert len(detector.passes_list) == 1
    event = detector.passes_list[0]
    assert event["de"] == 1
    assert event["para"] == 2
    assert event["tipo"] == "PASSE_CURTO"


def test_same_player_does_not_create_pass():
    detector = PassDetector(
        proximity_limit=50,
        min_frames=1,
        display_duration=10,
        referee_class_id=3,
        possession_grace_frames=2,
        distance_factor=0.7
    )

    player = create_player(1, 100, 100)

    ball = create_ball(102, 102)
    detector.update(1, 30, player, ball)

    ball = create_ball(105, 105)
    detector.update(2, 30, player, ball)

    assert detector.posse_atual[0] == 1
    assert detector.owner_frames == 2
    assert detector.passes_list == []


def test_ball_missing_inside_grace_keeps_owner():
    detector = PassDetector(
        proximity_limit=50,
        min_frames=1,
        display_duration=10,
        referee_class_id=3,
        possession_grace_frames=1,
        distance_factor=0.7
    )

    player = create_player(1, 100, 100)

    ball = create_ball(102, 102)
    detector.update(1, 30, player, ball)

    detector.update(2, 30, player, sv.Detections.empty())

    assert detector.posse_atual[0] == 1
    assert detector.ball_missing == 1
    assert detector.possession_summary()["frames_total"] == 2


def test_different_teams_is_not_a_pass():
    detector = PassDetector(
        proximity_limit=50,
        min_frames=1,
        display_duration=10,
        referee_class_id=3,
        possession_grace_frames=2,
        distance_factor=0.7
    )

    player_a = create_player(1, 100, 100, team_id=0)
    ball = create_ball(102, 102)
    detector.update(1, 30, player_a, ball)

    player_b = create_player(2, 200, 200, team_id=1)
    ball = create_ball(202, 202)
    detector.update(2, 30, player_b, ball)

    assert detector.posse_atual[0] == 2
    assert detector.passes_list == []
    assert detector.shots_list == []
