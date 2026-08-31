"""
Testes para ShotDetector.
Cole em: tests/unit/test_shot_detector.py
"""
import numpy as np
import supervision as sv
import pytest

from tcc_common.shot_detector import ShotDetector
from tests.fixtures.fake_data import create_player, create_ball, merge_players


def make_detector(speed_threshold=25.0, proximity_limit=120.0, cooldown=5, ratio=2.0):
    return ShotDetector(
        speed_threshold=speed_threshold,
        proximity_limit=proximity_limit,
        cooldown_frames=cooldown,
        min_speed_ratio=ratio,
        display_duration=10,
        debug=False,
    )


def make_ball_at(x, y):
    return create_ball(x, y)


class TestBasic:

    def test_no_crash_empty(self):
        det = make_detector()
        det.update(1, 30.0, sv.Detections.empty(), sv.Detections.empty())

    def test_no_shot_without_ball(self):
        det = make_detector()
        players = merge_players(create_player(1, 100, 100, team_id=0))
        det.update(1, 30.0, players, sv.Detections.empty())
        assert len(det.shots_list) == 0

    def test_no_shot_slow_ball(self):
        """Bola se movendo devagar não deve gerar chute."""
        det = make_detector(speed_threshold=50.0)
        players = merge_players(create_player(1, 100, 100, team_id=0))

        # Bola se move 5px por frame — lento
        for i in range(5):
            det.update(i + 1, 30.0, players, make_ball_at(100 + i * 5, 100))

        assert len(det.shots_list) == 0


class TestShotDetection:

    def test_shot_detected_on_fast_ball(self):
        """Bola parada que acelera muito deve gerar chute."""
        det = make_detector(speed_threshold=30.0, proximity_limit=150.0, ratio=2.0)
        player = create_player(1, 100, 100, team_id=0)
        players = merge_players(player)

        # Frame 1-2: bola parada perto do jogador
        det.update(1, 30.0, players, make_ball_at(105, 105))
        det.update(2, 30.0, players, make_ball_at(106, 106))

        # Frame 3: bola acelera bruscamente (chute)
        det.update(3, 30.0, players, make_ball_at(160, 160))

        assert len(det.shots_list) == 1

    def test_shot_attributes(self):
        """Chute deve ter todos os campos esperados."""
        det = make_detector(speed_threshold=30.0, proximity_limit=150.0, ratio=2.0)
        players = merge_players(create_player(1, 100, 100, team_id=0))

        det.update(1, 30.0, players, make_ball_at(105, 105))
        det.update(2, 30.0, players, make_ball_at(106, 106))
        det.update(3, 30.0, players, make_ball_at(160, 160))

        shot = det.shots_list[0]
        assert "frame" in shot
        assert "tempo_s" in shot
        assert "jogador_id" in shot
        assert "time" in shot
        assert "velocidade_px" in shot
        assert shot["jogador_id"] == 1
        assert shot["time"] == "Time 0"

    def test_shot_timestamp(self):
        """Timestamp deve bater com frame/fps."""
        det = make_detector(speed_threshold=30.0, proximity_limit=150.0, ratio=2.0)
        players = merge_players(create_player(1, 100, 100, team_id=0))

        fps = 25.0
        det.update(1, fps, players, make_ball_at(105, 105))
        det.update(2, fps, players, make_ball_at(106, 106))
        det.update(3, fps, players, make_ball_at(160, 160))

        assert abs(det.shots_list[0]["tempo_s"] - 3 / fps) < 0.01

    def test_no_shot_without_nearby_player(self):
        """Bola que acelera sem jogador próximo não deve gerar chute."""
        det = make_detector(speed_threshold=30.0, proximity_limit=50.0, ratio=2.0)
        # Jogador longe da bola
        players = merge_players(create_player(1, 500, 500, team_id=0))

        det.update(1, 30.0, players, make_ball_at(105, 105))
        det.update(2, 30.0, players, make_ball_at(106, 106))
        det.update(3, 30.0, players, make_ball_at(160, 160))

        assert len(det.shots_list) == 0

    def test_referee_ignored(self):
        """Árbitro próximo não deve ser atribuído como chutador."""
        det = make_detector(speed_threshold=30.0, proximity_limit=150.0, ratio=2.0)
        referee = create_player(99, 105, 105, team_id=3)
        players = merge_players(referee)

        det.update(1, 30.0, players, make_ball_at(105, 105))
        det.update(2, 30.0, players, make_ball_at(106, 106))
        det.update(3, 30.0, players, make_ball_at(160, 160))

        assert len(det.shots_list) == 0

    def test_hybrid_candidate_requires_sustained_goalward_trajectory(self):
        detector = ShotDetector(
            speed_threshold=30.0,
            proximity_limit=150.0,
            min_speed_ratio=2.0,
            confirmation_frames=3,
            min_progress_px=45.0,
            direction_consistency=0.8,
            min_horizontal_ratio=1.15,
            event_type="CANDIDATO_CHUTE",
        )
        players = merge_players(create_player(1, 100, 100, team_id=0))

        detector.update(1, 25.0, players, make_ball_at(105, 105), frame_width=640)
        detector.update(2, 25.0, players, make_ball_at(106, 105), frame_width=640)
        detector.update(3, 25.0, players, make_ball_at(160, 105), frame_width=640)
        assert detector.shots_list == []
        detector.update(4, 25.0, players, make_ball_at(205, 105), frame_width=640)
        detector.update(5, 25.0, players, make_ball_at(245, 105), frame_width=640)

        assert len(detector.shots_list) == 1
        assert detector.shots_list[0]["tipo"] == "CANDIDATO_CHUTE"
        assert detector.shots_list[0]["frame_confirmacao"] == 5
        assert 0.0 <= detector.shots_list[0]["confianca"] <= 1.0

    def test_hybrid_candidate_rejects_direction_reversal(self):
        detector = ShotDetector(
            speed_threshold=30.0,
            proximity_limit=150.0,
            min_speed_ratio=2.0,
            confirmation_frames=3,
            min_progress_px=45.0,
            min_horizontal_ratio=1.15,
        )
        players = merge_players(create_player(1, 100, 100, team_id=0))

        detector.update(1, 25.0, players, make_ball_at(105, 105), frame_width=640)
        detector.update(2, 25.0, players, make_ball_at(106, 105), frame_width=640)
        detector.update(3, 25.0, players, make_ball_at(160, 105), frame_width=640)
        detector.update(4, 25.0, players, make_ball_at(120, 105), frame_width=640)
        detector.update(5, 25.0, players, make_ball_at(80, 105), frame_width=640)

        assert detector.shots_list == []


class TestCooldown:

    def test_cooldown_prevents_duplicate(self):
        """Dois chutes em sequência rápida devem contar como um só."""
        det = make_detector(speed_threshold=20.0, proximity_limit=150.0,
                            cooldown=10, ratio=1.5)
        players = merge_players(create_player(1, 100, 100, team_id=0))

        det.update(1, 30.0, players, make_ball_at(105, 105))
        det.update(2, 30.0, players, make_ball_at(106, 106))
        det.update(3, 30.0, players, make_ball_at(150, 150))  # chute 1
        det.update(4, 30.0, players, make_ball_at(151, 151))
        det.update(5, 30.0, players, make_ball_at(200, 200))  # dentro do cooldown

        assert len(det.shots_list) == 1

    def test_second_shot_after_cooldown(self):
        """Após o cooldown, um novo chute deve ser registrado."""
        det = make_detector(speed_threshold=20.0, proximity_limit=150.0,
                            cooldown=3, ratio=1.5)
        players = merge_players(create_player(1, 100, 100, team_id=0))

        # Primeiro chute
        det.update(1, 30.0, players, make_ball_at(100, 100))
        det.update(2, 30.0, players, make_ball_at(101, 101))
        det.update(3, 30.0, players, make_ball_at(150, 150))  # chute 1

        # Cooldown de 3 frames
        det.update(4, 30.0, players, make_ball_at(155, 155))
        det.update(5, 30.0, players, make_ball_at(160, 160))
        det.update(6, 30.0, players, make_ball_at(165, 165))

        # Segundo chute após cooldown
        det.update(7, 30.0, players, make_ball_at(166, 166))
        det.update(8, 30.0, players, make_ball_at(167, 167))
        det.update(9, 30.0, players, make_ball_at(230, 230))  # chute 2

        assert len(det.shots_list) == 2


class TestSummary:

    def test_summary_counts_per_team(self):
        det = make_detector(speed_threshold=20.0, proximity_limit=150.0,
                            cooldown=3, ratio=1.5)
        player_a = create_player(1, 100, 100, team_id=0)
        player_b = create_player(2, 300, 300, team_id=1)

        # Chute do time 0
        players = merge_players(player_a, player_b)
        det.update(1, 30.0, players, make_ball_at(105, 105))
        det.update(2, 30.0, players, make_ball_at(106, 106))
        det.update(3, 30.0, players, make_ball_at(160, 160))

        s = det.shots_summary()
        assert s["total"] == 1
        assert s.get("chutes_time_0", 0) == 1
        assert s.get("chutes_time_1", 0) == 0

    def test_summary_empty(self):
        det = make_detector()
        s = det.shots_summary()
        assert s["total"] == 0


class TestAnnotate:

    def test_annotate_does_not_crash(self):
        det = make_detector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.annotate_frame(frame, frame_num=1)
        assert result.shape == frame.shape
