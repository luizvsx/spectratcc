import numpy as np
import supervision as sv

from tcc_common.ball import BallAnnotator, BallTracker, filter_ball_detections_on_pitch
from tests.fixtures.fake_data import create_ball


def test_tracker_returns_empty_detection_when_ball_is_missing():
    tracker = BallTracker(buffer_size=3)
    result = tracker.update(sv.Detections.empty())
    assert len(result) == 0


def test_tracker_selects_candidate_closest_to_recent_trajectory():
    tracker = BallTracker(buffer_size=3)
    tracker.update(create_ball(0, 0))
    candidates = sv.Detections.merge([create_ball(10, 0), create_ball(100, 0)])

    selected = tracker.update(candidates)

    assert len(selected) == 1
    center = selected.get_anchors_coordinates(sv.Position.CENTER)[0]
    assert center[0] == 12.5


def test_annotator_preserves_frame_shape_and_draws_ball():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    annotator = BallAnnotator(radius=6, buffer_size=3)

    result = annotator.annotate(frame.copy(), create_ball(40, 40))

    assert result.shape == frame.shape
    assert np.count_nonzero(result) > 0


def test_pitch_filter_keeps_candidate_surrounded_by_grass():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[:] = (40, 140, 40)

    filtered = filter_ball_detections_on_pitch(frame, create_ball(70, 50))

    assert len(filtered) == 1


def test_pitch_filter_rejects_candidate_in_stands():
    frame = np.full((120, 160, 3), (80, 80, 80), dtype=np.uint8)

    filtered = filter_ball_detections_on_pitch(frame, create_ball(70, 50))

    assert len(filtered) == 0
