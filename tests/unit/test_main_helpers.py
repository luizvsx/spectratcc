import numpy as np
import supervision as sv

from main import resolve_goalkeepers_team_id
from tests.fixtures.fake_data import create_player, merge_players


def test_no_goalkeepers_returns_empty_array():
    players = merge_players(create_player(1, 100, 100, team_id=0))
    result = resolve_goalkeepers_team_id(
        players, players.class_id, sv.Detections.empty()
    )
    assert result.dtype == int
    assert result.size == 0


def test_goalkeeper_defaults_to_team_zero_without_players():
    goalkeeper = create_player(10, 10, 10, team_id=1)
    result = resolve_goalkeepers_team_id(
        sv.Detections.empty(), np.array([], dtype=int), goalkeeper
    )
    assert result.tolist() == [0]


def test_goalkeepers_are_assigned_to_nearest_team_centroid():
    players = merge_players(
        create_player(1, 20, 20, team_id=0),
        create_player(2, 400, 20, team_id=1),
    )
    goalkeepers = merge_players(
        create_player(10, 30, 20, team_id=1),
        create_player(11, 390, 20, team_id=0),
    )

    result = resolve_goalkeepers_team_id(
        players, players.class_id, goalkeepers
    )

    assert result.tolist() == [0, 1]
