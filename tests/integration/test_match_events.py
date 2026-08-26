import supervision as sv

from tcc_common.pass_detector import PassDetector
from tests.fixtures.detections import ball, foot_x, player, players


def make_detector(**overrides):
    options = {
        "proximity_limit": 25,
        "distance_factor": 0.5,
        "min_frames": 2,
        "possession_grace_frames": 2,
        "short_pass_threshold": 150,
        "shot_threshold": 500,
    }
    options.update(overrides)
    return PassDetector(**options)


def give_ball(detector, frame, x, detections):
    detector.update(frame, 25, detections, ball(foot_x(x)))


def test_sequence_produces_short_pass_long_pass_and_shot():
    detector = make_detector()
    detections = players(
        player(1, 100, team_id=0),
        player(2, 200, team_id=0),
        player(3, 450, team_id=0),
        player(4, 1050, team_id=0),
    )

    sequence = [100, 100, 200, 200, 450, 450, 1050]
    for frame, owner_x in enumerate(sequence, start=1):
        give_ball(detector, frame, owner_x, detections)

    assert [event["tipo"] for event in detector.passes_list] == [
        "PASSE_CURTO",
        "PASSE_LONGO",
    ]
    assert [event["tipo"] for event in detector.shots_list] == ["CHUTE"]
    assert [(event["de"], event["para"]) for event in detector.passes_list] == [
        (1, 2),
        (2, 3),
    ]
    assert (detector.shots_list[0]["de"], detector.shots_list[0]["para"]) == (3, 4)


def test_detection_gap_inside_grace_does_not_break_pass():
    detector = make_detector(possession_grace_frames=2)
    detections = players(player(1, 100, team_id=0), player(2, 200, team_id=0))

    give_ball(detector, 1, 100, detections)
    detector.update(2, 25, detections, sv.Detections.empty())
    detector.update(3, 25, detections, sv.Detections.empty())
    give_ball(detector, 4, 200, detections)

    assert len(detector.passes_list) == 1
    assert detector.passes_list[0]["de"] == 1
    assert detector.passes_list[0]["para"] == 2


def test_interception_is_not_reported_as_pass_or_shot():
    detector = make_detector(min_frames=1)
    detections = players(
        player(1, 100, team_id=0),
        player(9, 700, team_id=1),
    )

    give_ball(detector, 1, 100, detections)
    give_ball(detector, 2, 700, detections)

    assert detector.posse_atual[0:2] == (9, "Time 1")
    assert detector.passes_list == []
    assert detector.shots_list == []
    assert detector.possession_summary()["frames_time_0"] == 1
    assert detector.possession_summary()["frames_time_1"] == 1
