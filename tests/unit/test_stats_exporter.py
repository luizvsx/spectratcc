import csv

from tcc_common.stats_exporter import save_stats_csv


def test_export_contains_metadata_events_and_summary(tmp_path):
    output = tmp_path / "nested" / "match_stats.csv"
    passes = [
        {
            "tipo": "PASSE_CURTO",
            "frame": 25,
            "tempo_s": 1.0,
            "time": "Time 0",
            "de": 1,
            "para": 2,
            "distancia_px": 100.0,
        }
    ]
    shots = [
        {
            "tipo": "CHUTE",
            "frame": 50,
            "tempo_s": 2.0,
            "time": "Time 1",
            "de": 9,
            "para": 10,
            "distancia_px": 600.0,
        }
    ]
    possession = {
        "frames_total": 50,
        "frames_sem_posse": 5,
        "frames_time_0": 30,
        "frames_time_1": 15,
        "pct_time_0": 66.7,
        "pct_time_1": 33.3,
    }

    returned = save_stats_csv(
        str(output), passes, shots, possession, video_path="match.mp4", fps=25
    )

    assert returned == str(output)
    rows = list(csv.reader(output.open(encoding="utf-8")))
    assert ["# Vídeo", "match.mp4"] in rows
    assert ["PASSE_CURTO", "25", "1.0", "Time 0", "1", "2", "100.0"] in rows
    assert ["CHUTE", "50", "2.0", "Time 1", "9", "10", "600.0"] in rows
    assert ["passes_total", "1", "0", "1"] in rows
    assert ["chutes", "0", "1", "1"] in rows


def test_export_accepts_empty_event_lists(tmp_path):
    output = tmp_path / "empty.csv"

    save_stats_csv(str(output), [], [], {}, fps=30)

    text = output.read_text(encoding="utf-8")
    assert "## EVENTOS" in text
    assert "passes_total,0,0,0" in text
    assert "chutes,0,0,0" in text
