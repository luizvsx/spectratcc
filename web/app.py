"""
SPECTRA — servidor Flask
Roda no servidor SSH com GPU.

Instalar: pip install flask --break-system-packages
Rodar:    python app.py
Acessar:  http://localhost:5000 (via SSH tunnel)

SSH tunnel no seu computador local:
  ssh -L 5000:localhost:5000 usuario@servidor
"""

import os
import csv
import subprocess
import sys
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask import send_from_directory
from werkzeug.utils import secure_filename

WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parent
STATIC_DIR = WEB_DIR / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR))

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

jobs = {}


def run_pipeline(job_id: str, video_path: str, output_video: str, csv_path: str):
    """Executa o pipeline SPECTRA em background."""
    jobs[job_id]["status"] = "processando"
    jobs[job_id]["log"] = []

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "main.py"),
        "--source_video_path", video_path,
        "--target_video_path", output_video,
        "--device", os.getenv("SPECTRA_DEVICE", "cpu"),
        "--mode", "FULL_ANALYSIS"
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT)
        )

        for line in proc.stdout:
            line = line.strip()
            if line:
                jobs[job_id]["log"].append(line)

        proc.wait()

        video_ready = Path(output_video).is_file() and Path(output_video).stat().st_size > 0
        csv_ready = Path(csv_path).is_file() and Path(csv_path).stat().st_size > 0
        if proc.returncode == 0 and video_ready and csv_ready:
            jobs[job_id]["status"] = "concluido"
            jobs[job_id]["csv_path"] = csv_path
            jobs[job_id]["video_path"] = output_video
        else:
            jobs[job_id]["status"] = "erro"
            missing = []
            if not video_ready:
                missing.append("vídeo anotado")
            if not csv_ready:
                missing.append("CSV")
            detail = f" Arquivos ausentes: {', '.join(missing)}." if missing else ""
            jobs[job_id]["error"] = f"Pipeline falhou.{detail} Veja o log para detalhes."

    except Exception as e:
        jobs[job_id]["status"] = "erro"
        jobs[job_id]["error"] = str(e)


def parse_csv(csv_path: str) -> dict:
    """Lê o CSV do SPECTRA e retorna dict estruturado."""
    eventos = []
    resumo = {}
    section = None

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = None

        for row in reader:
            if not row or not row[0].strip():
                continue
            val = row[0].strip()

            if val.startswith("##"):
                section = val.replace("##", "").strip().upper()
                headers = None
                continue

            if val.startswith("#"):
                continue

            if section == "EVENTOS":
                if headers is None:
                    headers = [h.strip() for h in row]
                    continue
                if len(row) >= len(headers):
                    ev = {headers[i]: row[i].strip() for i in range(len(headers))}
                    eventos.append(ev)

            elif section == "RESUMO":
                if headers is None:
                    headers = [h.strip() for h in row]
                    continue
                if len(row) >= 2:
                    resumo[row[0].strip()] = {
                        "time_0": row[1].strip() if len(row) > 1 else "",
                        "time_1": row[2].strip() if len(row) > 2 else "",
                        "total":  row[3].strip() if len(row) > 3 else "",
                    }

    return {"eventos": eventos, "resumo": resumo}


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    f = request.files["video"]
    filename = secure_filename(f.filename or "")
    if not filename:
        return jsonify({"error": "Nome de arquivo inválido"}), 400

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    video_path  = str(UPLOAD_DIR / f"{job_id}_{filename}")
    output_video = str(OUTPUT_DIR / f"{job_id}_resultado.mp4")
    csv_path    = output_video.replace(".mp4", "_stats.csv")

    f.save(video_path)

    jobs[job_id] = {
        "status": "aguardando",
        "video_original": filename,
        "log": [],
        "csv_path": None,
        "video_path": None,
        "error": None,
    }

    t = threading.Thread(
        target=run_pipeline,
        args=(job_id, video_path, output_video, csv_path),
        daemon=True
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job não encontrado"}), 404
    job = jobs[job_id]
    return jsonify({
        "status": job["status"],
        "log":    job["log"][-20:],
        "error":  job["error"],
    })


@app.route("/resultado/<job_id>")
def resultado(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Job não encontrado"}), 404
    job = jobs[job_id]
    if job["status"] != "concluido":
        return jsonify({"error": "Processamento não concluído"}), 400

    dados = parse_csv(job["csv_path"])
    dados["video_original"] = job["video_original"]
    return jsonify(dados)


@app.route("/download/csv/<job_id>")
def download_csv(job_id):
    if job_id not in jobs or not jobs[job_id]["csv_path"]:
        return jsonify({"error": "Arquivo não disponível"}), 404
    return send_file(jobs[job_id]["csv_path"], as_attachment=True)


@app.route("/download/video/<job_id>")
def download_video(job_id):
    if job_id not in jobs or not jobs[job_id]["video_path"]:
        return jsonify({"error": "Arquivo não disponível"}), 404
    return send_file(jobs[job_id]["video_path"], as_attachment=True)


@app.route("/preview/video/<job_id>")
def preview_video(job_id):
    if job_id not in jobs or not jobs[job_id]["video_path"]:
        return jsonify({"error": "Arquivo não disponível"}), 404
    return send_file(
        jobs[job_id]["video_path"],
        mimetype="video/mp4",
        conditional=True,
    )


if __name__ == "__main__":
    host = os.getenv("SPECTRA_HOST", "127.0.0.1")
    port = int(os.getenv("SPECTRA_PORT", "5000"))
    print(f"SPECTRA rodando em http://{host}:{port}")
    print("No seu computador local, abra o tunnel SSH:")
    print("  ssh -L 5000:localhost:5000 usuario@servidor")
    app.run(host=host, port=port, debug=False)
