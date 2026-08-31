import pytest

from tcc_common.evaluation import (
    TimedEvent,
    evaluate_timed_events,
    read_ground_truth,
    read_spectra_events,
)
from tcc_common.stats_exporter import save_stats_csv


def event(tipo, tempo, time="time 0"):
    return TimedEvent(tipo=tipo, tempo_s=tempo, time=time)


def test_events_inside_time_tolerance_match():
    metrics = evaluate_timed_events(
        [event("PASSE_CURTO", 10.4)],
        [event("PASSE_CURTO", 10.0)],
        tolerance_s=0.5,
    )
    assert metrics.true_positives == 1
    assert metrics.f1 == 1.0


def test_wrong_type_does_not_match():
    metrics = evaluate_timed_events(
        [event("CHUTE", 10.0)],
        [event("PASSE_LONGO", 10.0)],
    )
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1


def test_one_truth_event_cannot_match_twice():
    metrics = evaluate_timed_events(
        [event("CHUTE", 10.0), event("CHUTE", 10.2)],
        [event("CHUTE", 10.1)],
        tolerance_s=0.5,
    )
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1


def test_negative_tolerance_is_rejected():
    with pytest.raises(ValueError):
        evaluate_timed_events([], [], tolerance_s=-0.1)


def test_read_ground_truth(tmp_path):
    path = tmp_path / "truth.csv"
    path.write_text(
        "tipo,tempo_s,time\nPASSE_CURTO,2.5,Time 0\n",
        encoding="utf-8",
    )
    events = read_ground_truth(str(path))
    assert events == [TimedEvent("PASSE_CURTO", 2.5, "time 0")]


def test_ground_truth_rejects_missing_columns(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text("tipo,tempo_s\nCHUTE,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tipo, tempo_s, time"):
        read_ground_truth(str(path))


def test_read_spectra_events(tmp_path):
    path = tmp_path / "stats.csv"
    save_stats_csv(
        str(path),
        [{
            "tipo": "PASSE_LONGO", "frame": 12, "tempo_s": 1.2,
            "time": "Time 1", "de": 4, "para": 5, "distancia_px": 250,
        }],
        [],
        {},
    )
    events = read_spectra_events(str(path))
    assert events == [TimedEvent("PASSE_LONGO", 1.2, "time 1")]
