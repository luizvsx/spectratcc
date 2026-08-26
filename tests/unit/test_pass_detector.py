import numpy as np
import pytest
import supervision as sv

from tcc_common.pass_detector import PassDetector, PossessionCounter
from tests.fixtures.detections import ball, foot_x, player, players


def detector(**overrides) -> PassDetector:
    options = {
        "proximity_limit": 25,
        "distance_factor": 0.5,
        "min_frames": 2,
        "display_duration": 5,
        "possession_grace_frames": 2,
        "short_pass_threshold": 150,
        "shot_threshold": 500,
    }
    options.update(overrides)
    return PassDetector(**options)


def update_owner(det: PassDetector, frame: int, owner_x: float, all_players, fps=25):
    det.update(frame, fps, all_players, ball(foot_x(owner_x)))


class TestPossessionCounter:
    def test_summary_counts_contested_and_unowned_frames(self):
        counter = PossessionCounter()
        for team_id in (0, 0, 1, None):
            counter.update(team_id)

        assert counter.summary() == {
            "frames_total": 4,
            "frames_sem_posse": 1,
            "frames_time_0": 2,
            "frames_time_1": 1,
            "pct_time_0": 66.7,
            "pct_time_1": 33.3,
        }

    def test_unknown_team_is_counted_as_no_possession(self):
        counter = PossessionCounter()
        counter.update(7)

        assert counter.summary()["frames_sem_posse"] == 1
        assert counter.percentage(0) == 0


class TestPossessionState:
    def test_first_owner_and_continuity_are_recorded(self):
        det = detector()
        detections = player(1, 100, team_id=0)

        update_owner(det, 1, 100, detections)
        update_owner(det, 2, 100, detections)

        assert det.posse_atual[0:2] == (1, "Time 0")
        assert det.owner_frames == 2

    def test_referee_cannot_receive_possession(self):
        det = detector()

        det.update(1, 25, player(99, 100, team_id=3), ball(foot_x(100)))

        assert det.posse_atual is None
        assert det.possession_summary()["frames_sem_posse"] == 1

    def test_loose_ball_preserves_last_owner(self):
        det = detector()
        detections = player(1, 100, team_id=0)
        update_owner(det, 1, 100, detections)

        det.update(2, 25, detections, ball(400))

        assert det.posse_atual[0] == 1
        assert det.possession_summary()["frames_time_0"] == 2

    def test_missing_ball_uses_last_position_only_during_grace(self):
        det = detector(possession_grace_frames=2)
        detections = player(1, 100, team_id=0)
        update_owner(det, 1, 100, detections)

        det.update(2, 25, detections, sv.Detections.empty())
        det.update(3, 25, detections, sv.Detections.empty())
        assert det.posse_atual[0] == 1

        det.update(4, 25, detections, sv.Detections.empty())
        assert det.posse_atual is None
        assert det.owner_frames == 0

    def test_empty_players_do_not_create_owner(self):
        det = detector()

        det.update(1, 25, sv.Detections.empty(), ball(100))

        assert det.posse_atual is None
        assert det.possession_summary()["frames_sem_posse"] == 1


class TestEvents:
    @pytest.mark.parametrize(
        ("receiver_x", "expected_type", "collection"),
        [
            (200, "PASSE_CURTO", "passes_list"),
            (350, "PASSE_LONGO", "passes_list"),
            (700, "CHUTE", "shots_list"),
        ],
    )
    def test_same_team_transfer_is_classified_by_distance(
        self, receiver_x, expected_type, collection
    ):
        det = detector()
        detections = players(
            player(1, 100, team_id=0),
            player(2, receiver_x, team_id=0),
        )
        update_owner(det, 1, 100, detections)
        update_owner(det, 2, 100, detections)

        update_owner(det, 3, receiver_x, detections)

        events = getattr(det, collection)
        assert len(events) == 1
        assert events[0] == {
            "frame": 3,
            "tempo_s": 0.12,
            "de": 1,
            "para": 2,
            "time": "Time 0",
            "distancia_px": pytest.approx(receiver_x - 100),
            "tipo": expected_type,
        }

    def test_unstable_owner_does_not_generate_event(self):
        det = detector(min_frames=3)
        detections = players(player(1, 100), player(2, 200))
        update_owner(det, 1, 100, detections)

        update_owner(det, 2, 200, detections)

        assert det.passes_list == []

    def test_interception_changes_owner_without_counting_pass(self):
        det = detector(min_frames=1)
        detections = players(
            player(1, 100, team_id=0),
            player(2, 200, team_id=1),
        )
        update_owner(det, 1, 100, detections)

        update_owner(det, 2, 200, detections)

        assert det.posse_atual[0:2] == (2, "Time 1")
        assert det.passes_list == []
        assert det.shots_list == []

    def test_summary_separates_teams_and_event_kinds(self):
        det = detector(min_frames=1)
        team_zero = players(player(1, 100, team_id=0), player(2, 200, team_id=0))
        update_owner(det, 1, 100, team_zero)
        update_owner(det, 2, 200, team_zero)

        assert det.events_summary() == {
            "total_passes": 1,
            "total_chutes": 0,
            "passes_time_0": 1,
            "chutes_time_0": 0,
            "dist_media_passes_time_0": 100.0,
            "passes_time_1": 0,
            "chutes_time_1": 0,
            "dist_media_passes_time_1": 0.0,
        }

    def test_annotation_returns_copy_and_expires_old_messages(self):
        det = detector(min_frames=1, display_duration=1)
        detections = players(player(1, 100), player(2, 200))
        update_owner(det, 1, 100, detections)
        update_owner(det, 2, 200, detections)
        frame = np.zeros((180, 400, 3), dtype=np.uint8)

        annotated = det.annotate_frame(frame, frame_num=3)

        assert annotated is not frame
        assert annotated.shape == frame.shape
        assert det.eventos_display == []
