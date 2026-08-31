import numpy as np
import supervision as sv
import pytest

from tcc_common.possession import get_ball_possession
from tests.fixtures.fake_data import create_player, create_ball, merge_players


def test_player_closest_gets_possession():
    player_a = create_player(1, 100, 100)
    player_b = create_player(2, 300, 300)

    players = merge_players(player_a, player_b)
    ball = create_ball(105, 105)

    player_id, team_id = get_ball_possession(players, ball, possession_threshold=50)

    assert player_id == 1


def test_no_ball_no_possession():
    player = create_player(1, 100, 100)
    empty_ball = sv.Detections.empty()

    player_id, team_id = get_ball_possession(player, empty_ball, possession_threshold=50)

    assert player_id is None
    assert team_id is None

def test_ball_far_no_possession():
    player = create_player(1, 100, 100)
    ball = create_ball(500, 500)

    player_id, team_id = get_ball_possession(player, ball, possession_threshold=30)

    assert player_id is None
    assert team_id is None


def test_correct_team_id_returned():
    player = create_player(1, 100, 100, team_id=0)
    ball = create_ball(102, 102)

    player_id, team_id = get_ball_possession(player, ball, possession_threshold=50)

    assert player_id == 1
    assert team_id == 0


def test_multiple_players_choose_nearest():
    player_a = create_player(1, 100, 100)
    player_b = create_player(2, 120, 120)
    player_c = create_player(3, 300, 300)

    players = merge_players(player_a, player_b, player_c)
    ball = create_ball(118, 118)

    player_id, _ = get_ball_possession(players, ball, possession_threshold=50)

    assert player_id == 2


def test_empty_players():
    empty = sv.Detections.empty()
    ball = create_ball(100, 100)

    player_id, team_id = get_ball_possession(empty, ball, possession_threshold=50)

    assert player_id is None
    assert team_id is None


def test_referee_never_gets_possession():
    referee = create_player(99, 100, 100, team_id=3)
    ball = create_ball(102, 102)

    player_id, team_id = get_ball_possession(
        referee, ball, possession_threshold=50
    )

    assert player_id is None
    assert team_id is None
