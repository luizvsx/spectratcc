import io

import pytest

import app as app_module
from tcc_common.stats_exporter import save_stats_csv


@pytest.fixture
def client(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    output_dir = tmp_path / "outputs"
    upload_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.setattr(app_module, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(app_module, "OUTPUT_DIR", output_dir)
    app_module.jobs.clear()
    app_module.app.config.update(TESTING=True)
    yield app_module.app.test_client()
    app_module.jobs.clear()


def test_index_is_available(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"SPECTRA" in response.data


def test_upload_rejects_missing_file(client):
    response = client.post("/upload", data={})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Nenhum arquivo enviado"


def test_upload_saves_file_and_schedules_pipeline(client, monkeypatch):
    scheduled = {}

    class FakeThread:
        def __init__(self, target, args, daemon):
            scheduled.update(target=target, args=args, daemon=daemon, started=False)

        def start(self):
            scheduled["started"] = True

    monkeypatch.setattr(app_module.threading, "Thread", FakeThread)

    response = client.post(
        "/upload",
        data={"video": (io.BytesIO(b"fake-video"), "partida.mp4")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    job_id = response.get_json()["job_id"]
    assert app_module.jobs[job_id]["status"] == "aguardando"
    assert scheduled["target"] is app_module.run_pipeline
    assert scheduled["daemon"] is True
    assert scheduled["started"] is True
    assert scheduled["args"][1].endswith(f"{job_id}_partida.mp4")
    assert (app_module.UPLOAD_DIR / f"{job_id}_partida.mp4").read_bytes() == b"fake-video"


def test_status_unknown_job_returns_404(client):
    response = client.get("/status/inexistente")
    assert response.status_code == 404


def test_status_returns_only_last_twenty_log_lines(client):
    app_module.jobs["job"] = {
        "status": "processando", "log": [str(i) for i in range(25)],
        "error": None,
    }
    response = client.get("/status/job")
    body = response.get_json()
    assert response.status_code == 200
    assert body["log"] == [str(i) for i in range(5, 25)]


def test_result_returns_parsed_csv(client, tmp_path):
    csv_path = tmp_path / "stats.csv"
    save_stats_csv(str(csv_path), [], [], {"frames_total": 30}, fps=30)
    app_module.jobs["done"] = {
        "status": "concluido", "video_original": "partida.mp4",
        "csv_path": str(csv_path), "video_path": None, "error": None, "log": [],
    }

    response = client.get("/resultado/done")
    body = response.get_json()

    assert response.status_code == 200
    assert body["video_original"] == "partida.mp4"
    assert body["resumo"]["frames_total"]["total"] == "30"


def test_result_rejects_unfinished_job(client):
    app_module.jobs["pending"] = {
        "status": "processando", "video_original": "partida.mp4",
        "csv_path": None, "video_path": None, "error": None, "log": [],
    }
    response = client.get("/resultado/pending")
    assert response.status_code == 400


def test_download_csv_returns_completed_file(client, tmp_path):
    csv_path = tmp_path / "stats.csv"
    csv_path.write_text("dados", encoding="utf-8")
    app_module.jobs["done"] = {
        "csv_path": str(csv_path), "video_path": None,
    }
    response = client.get("/download/csv/done")
    assert response.status_code == 200
    assert response.data == b"dados"


def test_download_video_returns_404_when_unavailable(client):
    response = client.get("/download/video/missing")
    assert response.status_code == 404


def test_run_pipeline_marks_success_and_collects_log(tmp_path, monkeypatch):
    csv_path = tmp_path / "result_stats.csv"
    csv_path.write_text("ok", encoding="utf-8")
    app_module.jobs["job"] = {"status": "aguardando", "log": []}
    captured = {}

    class FakeProcess:
        stdout = ["linha 1\n", "linha 2\n"]
        returncode = 0

        def wait(self):
            captured["waited"] = True

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(app_module.subprocess, "Popen", fake_popen)
    app_module.run_pipeline(
        "job", "input.mp4", "result.mp4", str(csv_path)
    )

    assert captured["command"][0] == app_module.sys.executable
    assert captured["command"][1] == "main.py"
    assert captured["waited"] is True
    assert app_module.jobs["job"]["status"] == "concluido"
    assert app_module.jobs["job"]["log"] == ["linha 1", "linha 2"]


def test_run_pipeline_marks_subprocess_failure(monkeypatch):
    app_module.jobs["job"] = {"status": "aguardando", "log": []}

    class FailedProcess:
        stdout = []
        returncode = 1

        def wait(self):
            pass

    monkeypatch.setattr(
        app_module.subprocess, "Popen", lambda *args, **kwargs: FailedProcess()
    )
    app_module.run_pipeline("job", "input.mp4", "result.mp4", "missing.csv")

    assert app_module.jobs["job"]["status"] == "erro"
    assert "Pipeline falhou" in app_module.jobs["job"]["error"]
