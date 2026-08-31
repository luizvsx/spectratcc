"""Teste opcional do pipeline real com modelos e vídeo fornecidos pelo usuário."""

import os
import subprocess
import sys
from pathlib import Path

import cv2
import pytest


PROJECT_ROOT = Path(__file__).parents[2]
TEST_VIDEO = os.environ.get("SPECTRA_TEST_VIDEO")


@pytest.mark.gpu
@pytest.mark.skipif(
    not TEST_VIDEO,
    reason="defina SPECTRA_TEST_VIDEO para executar o pipeline real",
)
def test_real_full_analysis_creates_video_and_csv(tmp_path):
    source = Path(TEST_VIDEO).expanduser().resolve()
    if not source.is_file():
        pytest.fail(f"vídeo de teste não encontrado: {source}")

    target = tmp_path / "resultado.mp4"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--source_video_path", str(source),
        "--target_video_path", str(target),
        "--device", os.environ.get("SPECTRA_TEST_DEVICE", "cuda"),
        "--mode", "FULL_ANALYSIS",
    ]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("SPECTRA_TEST_TIMEOUT", "3600")),
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert target.is_file() and target.stat().st_size > 0

    csv_path = target.with_name("resultado_stats.csv")
    assert csv_path.is_file() and csv_path.stat().st_size > 0
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "## EVENTOS" in csv_text
    assert "## RESUMO" in csv_text

    capture = cv2.VideoCapture(str(target))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert frame_count > 0
