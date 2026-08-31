import csv
from pathlib import Path

from app import parse_csv
from tcc_common.stats_exporter import save_stats_csv


def sample_possession():
    return {
        "frames_total": 90,
        "frames_sem_posse": 10,
        "frames_time_0": 50,
        "frames_time_1": 30,
        "pct_time_0": 62.5,
        "pct_time_1": 37.5,
    }


def test_save_stats_csv_creates_parseable_report(tmp_path):
    output = tmp_path / "nested" / "stats.csv"
    passes = [{
        "tipo": "PASSE_CURTO", "frame": 10, "tempo_s": 1.0,
        "time": "Time 0", "de": 7, "para": 8, "distancia_px": 80.0,
    }]
    shots = [{
        "tipo": "CHUTE", "frame": 20, "tempo_s": 2.0,
        "time": "Time 1", "de": 9, "para": "", "distancia_px": 520.0,
    }]

    returned = save_stats_csv(
        str(output), passes, shots, sample_possession(), "partida.mp4", fps=30,
    )

    assert Path(returned) == output.resolve()
    assert output.exists()
    parsed = parse_csv(str(output))
    assert [event["tipo"] for event in parsed["eventos"]] == [
        "PASSE_CURTO", "CHUTE"
    ]
    assert parsed["resumo"]["passes_total"]["total"] == "1"
    assert parsed["resumo"]["chutes"]["total"] == "1"
    assert parsed["resumo"]["duracao_video_s"]["total"] == "3.0"


def test_events_are_sorted_by_frame(tmp_path):
    output = tmp_path / "stats.csv"
    passes = [
        {"tipo": "PASSE_LONGO", "frame": 30, "tempo_s": 3.0,
         "time": "Time 0", "de": 1, "para": 2, "distancia_px": 200},
        {"tipo": "PASSE_CURTO", "frame": 10, "tempo_s": 1.0,
         "time": "Time 0", "de": 2, "para": 3, "distancia_px": 50},
    ]

    save_stats_csv(str(output), passes, [], sample_possession())
    parsed = parse_csv(str(output))

    assert [int(event["frame"]) for event in parsed["eventos"]] == [10, 30]


def test_empty_report_still_has_sections(tmp_path):
    output = tmp_path / "empty.csv"
    save_stats_csv(str(output), [], [], {}, fps=0)

    rows = list(csv.reader(output.open(encoding="utf-8")))
    first_column = [row[0] for row in rows if row]
    assert "## EVENTOS" in first_column
    assert "## RESUMO" in first_column
    assert "duracao_video_s" not in first_column
