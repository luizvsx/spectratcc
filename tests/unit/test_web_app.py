import io
from unittest.mock import MagicMock

import pytest

from tcc_common.stats_exporter import save_stats_csv
from web import app as web_app


@pytest.fixture(autouse=True)
def isolated_web_state(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    output_dir = tmp_path / "outputs"
    upload_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.setattr(web_app, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(web_app, "OUTPUT_DIR", output_dir)
    web_app.jobs.clear()
    web_app.app.config.update(TESTING=True)
    yield
    web_app.jobs.clear()


def test_index_serves_dashboard():
    response = web_app.app.test_client().get("/")
    assert response.status_code == 200
    assert b"SPECTRA" in response.data
    assert b"chart-eventos" in response.data


def test_upload_requires_video():
    response = web_app.app.test_client().post("/upload")
    assert response.status_code == 400
    assert response.get_json()["error"] == "Nenhum arquivo enviado"


def test_upload_sanitizes_name_and_starts_background_job(monkeypatch):
    thread = MagicMock()
    monkeypatch.setattr(web_app.threading, "Thread", MagicMock(return_value=thread))
    response = web_app.app.test_client().post(
        "/upload",
        data={"video": (io.BytesIO(b"video"), "../../partida teste.mp4")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    job_id = response.get_json()["job_id"]
    assert web_app.jobs[job_id]["video_original"] == "partida_teste.mp4"
    assert list(web_app.UPLOAD_DIR.glob("*_partida_teste.mp4"))
    thread.start.assert_called_once_with()


def test_status_unknown_job_returns_404():
    assert web_app.app.test_client().get("/status/inexistente").status_code == 404


def test_result_parses_generated_csv(tmp_path):
    csv_path = tmp_path / "stats.csv"
    save_stats_csv(
        str(csv_path),
        passes_list=[{
            "tipo": "PASSE_CURTO", "frame": 25, "tempo_s": 1.0,
            "time": "Time 0", "de": 1, "para": 2, "distancia_px": 100,
        }],
        shots_list=[],
        possession_summary={"pct_time_0": 60, "pct_time_1": 40},
        fps=25,
    )
    web_app.jobs["job"] = {
        "status": "concluido", "video_original": "partida.mp4",
        "csv_path": str(csv_path), "video_path": str(tmp_path / "resultado.mp4"),
        "log": [], "error": None,
    }

    response = web_app.app.test_client().get("/resultado/job")
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["video_original"] == "partida.mp4"
    assert payload["eventos"][0]["tipo"] == "PASSE_CURTO"
    assert payload["resumo"]["passes_total"]["total"] == "1"


def test_pipeline_uses_current_python_and_src_main(tmp_path, monkeypatch):
    csv_path = tmp_path / "result_stats.csv"
    csv_path.touch()
    csv_path.write_text("csv", encoding="utf-8")
    output_video = tmp_path / "result.mp4"
    output_video.write_bytes(b"video")
    process = MagicMock(stdout=["processando\n"], returncode=0)
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(web_app.subprocess, "Popen", popen)
    web_app.jobs["job"] = {"status": "aguardando", "log": [], "error": None}

    web_app.run_pipeline(
        "job", str(tmp_path / "input.mp4"), str(output_video), str(csv_path)
    )

    command = popen.call_args.args[0]
    assert command[0] == web_app.sys.executable
    assert command[1] == str(web_app.PROJECT_ROOT / "src" / "main.py")
    assert command[-1] == "FULL_ANALYSIS"
    assert popen.call_args.kwargs["cwd"] == str(web_app.PROJECT_ROOT)
    assert web_app.jobs["job"]["status"] == "concluido"
    assert web_app.jobs["job"]["log"] == ["processando"]


def test_pipeline_fails_when_annotated_video_is_missing(tmp_path, monkeypatch):
    csv_path = tmp_path / "result_stats.csv"
    csv_path.write_text("csv", encoding="utf-8")
    process = MagicMock(stdout=[], returncode=0)
    monkeypatch.setattr(web_app.subprocess, "Popen", MagicMock(return_value=process))
    web_app.jobs["job"] = {"status": "aguardando", "log": [], "error": None}

    web_app.run_pipeline(
        "job", str(tmp_path / "input.mp4"), str(tmp_path / "missing.mp4"), str(csv_path)
    )

    assert web_app.jobs["job"]["status"] == "erro"
    assert "vídeo anotado" in web_app.jobs["job"]["error"]
