import pytest

from tcc_common.metrics import compute_event_metrics


def test_perfect_detection():
    metrics = compute_event_metrics([(1, 2), (2, 3)], [(1, 2), (2, 3)])
    assert metrics.true_positives == 2
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_partial_detection():
    metrics = compute_event_metrics(
        [(1, 2), (2, 3)],
        [(1, 2), (2, 3), (3, 4)],
    )
    assert metrics.false_negatives == 1
    assert metrics.precision == 1.0
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(0.8)


def test_with_false_positives():
    metrics = compute_event_metrics(
        [(1, 2), (2, 3), (5, 6)],
        [(1, 2), (2, 3)],
    )
    assert metrics.false_positives == 1
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == 1.0
    assert metrics.f1 == pytest.approx(0.8)


def test_no_detections():
    metrics = compute_event_metrics([], [(1, 2)])
    assert metrics.precision == 0
    assert metrics.recall == 0
    assert metrics.f1 == 0


def test_no_ground_truth():
    metrics = compute_event_metrics([(1, 2)], [])
    assert metrics.precision == 0
    assert metrics.recall == 0
    assert metrics.f1 == 0


def test_duplicate_events_are_counted():
    metrics = compute_event_metrics(
        [(1, 2), (1, 2)],
        [(1, 2)],
    )
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.precision == 0.5
