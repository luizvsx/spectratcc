import pytest

from tcc_common.shot_detector import ShotDetector


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0, "SHORT_PASS"),
        (119.9, "SHORT_PASS"),
        (120, "LONG_PASS"),
        (279.9, "LONG_PASS"),
        (280, "LONG_PASS"),
        (399.9, "LONG_PASS"),
        (400, "SHOT"),
    ],
)
def test_classification_boundaries(distance, expected):
    result = ShotDetector().classify((0, 0), (distance, 0))

    assert result["event"] == expected
    assert result["distance"] == pytest.approx(distance)


def test_distance_is_euclidean():
    result = ShotDetector().classify((1, 1), (4, 5))

    assert result == {"event": "SHORT_PASS", "distance": pytest.approx(5)}


def test_custom_thresholds_are_respected():
    detector = ShotDetector(
        short_pass_threshold=10,
        long_pass_threshold=20,
        shot_threshold=30,
    )

    assert detector.classify((0, 0), (9, 0))["event"] == "SHORT_PASS"
    assert detector.classify((0, 0), (20, 0))["event"] == "LONG_PASS"
    assert detector.classify((0, 0), (30, 0))["event"] == "SHOT"
