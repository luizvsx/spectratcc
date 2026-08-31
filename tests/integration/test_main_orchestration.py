from types import SimpleNamespace

import numpy as np

import main as spectra_main


def test_full_analysis_mode_writes_every_generated_frame(monkeypatch, tmp_path):
    frames = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.ones((8, 8, 3), dtype=np.uint8),
    ]
    received = {}

    def fake_full_analysis(source_video_path, device, target_video_path):
        received.update(
            source=source_video_path,
            device=device,
            target=target_video_path,
        )
        return iter(frames)

    class FakeSink:
        def __init__(self, target_path, video_info):
            received["sink_target"] = target_path
            received["video_info"] = video_info
            received["written"] = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write_frame(self, frame):
            received["written"].append(frame.copy())

    video_info = SimpleNamespace(total_frames=2, fps=30)
    monkeypatch.setattr(spectra_main, "run_full_analysis", fake_full_analysis)
    monkeypatch.setattr(
        spectra_main.sv.VideoInfo, "from_video_path", lambda path: video_info
    )
    monkeypatch.setattr(spectra_main.sv, "VideoSink", FakeSink)

    target = str(tmp_path / "result.mp4")
    spectra_main.main("input.mp4", target, "cpu", spectra_main.Mode.FULL_ANALYSIS)

    assert received["source"] == "input.mp4"
    assert received["target"] == target
    assert received["device"] == "cpu"
    assert received["sink_target"] == target
    assert len(received["written"]) == 2
    assert np.array_equal(received["written"][1], frames[1])


def test_config_can_be_loaded_outside_project_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = spectra_main.load_config()
    assert config["project"]["name"] == "SPECTRA"
    assert "player" in config["paths"]["models"]
