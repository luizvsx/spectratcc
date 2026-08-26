import numpy as np
import supervision as sv

import main
from main import Mode, get_ball_possession, resolve_goalkeepers_team_id, validate_model_files
from tests.fixtures.detections import ball, foot_x, player, players


def test_nearest_player_receives_possession_and_team():
    detections = players(
        player(10, 100, team_id=0),
        player(20, 300, team_id=1),
    )

    owner_id, team_id = get_ball_possession(
        detections,
        ball(foot_x(300)),
        possession_threshold=30,
    )

    assert (owner_id, team_id) == (20, 1)


def test_possession_is_empty_without_ball_or_players():
    one_player = player(10, 100)

    assert get_ball_possession(one_player, sv.Detections.empty(), 30) == (None, None)
    assert get_ball_possession(sv.Detections.empty(), ball(110), 30) == (None, None)


def test_ball_at_threshold_is_not_possession():
    detection = player(10, 100)

    assert get_ball_possession(detection, ball(140), 30) == (None, None)


def test_goalkeepers_are_assigned_to_nearest_team_centroid():
    field_players = players(
        player(1, 90, team_id=0),
        player(2, 110, team_id=0),
        player(3, 490, team_id=1),
        player(4, 510, team_id=1),
    )
    goalkeepers = players(player(5, 50), player(6, 550))

    result = resolve_goalkeepers_team_id(
        field_players,
        field_players.class_id,
        goalkeepers,
        sv.Detections.empty(),
    )

    np.testing.assert_array_equal(result, [0, 1])


def test_goalkeeper_fallbacks_are_deterministic():
    goalkeeper = player(5, 50)

    no_players = resolve_goalkeepers_team_id(
        sv.Detections.empty(), np.array([], dtype=int), goalkeeper, sv.Detections.empty()
    )
    no_goalkeepers = resolve_goalkeepers_team_id(
        player(1, 100), np.array([0]), sv.Detections.empty(), sv.Detections.empty()
    )

    np.testing.assert_array_equal(no_players, [0])
    assert no_goalkeepers.dtype == int
    assert no_goalkeepers.size == 0


def test_missing_models_report_actionable_paths(tmp_path, monkeypatch):
    missing = tmp_path / "football-ball-detection.pt"
    monkeypatch.setattr(main, "BALL_DETECTION_MODEL_PATH", str(missing))

    with np.testing.assert_raises_regex(FileNotFoundError, "Modelos necessários"):
        validate_model_files(Mode.BALL_DETECTION)


def test_available_model_passes_validation(tmp_path, monkeypatch):
    model = tmp_path / "football-ball-detection.pt"
    model.touch()
    monkeypatch.setattr(main, "BALL_DETECTION_MODEL_PATH", str(model))

    validate_model_files(Mode.BALL_DETECTION)
