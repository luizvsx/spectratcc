"""
Testes para PassDetector refatorado + PossessionCounter.
Cole em: tests/unit/test_pass_detector.py
"""
import numpy as np
import supervision as sv
import pytest

from tcc_common.pass_detector import PassDetector, PossessionCounter
from tests.fixtures.fake_data import create_player, create_ball, merge_players


# ── Helpers ─────────────────────────────────────────────────────────────────

def make_detector(min_frames=1, proximity_limit=50, distance_factor=1.5, grace=3, debug=False):
    """
    proximity_limit=50, distance_factor=1.5 → para bboxes de 10px de altura
    (como cria o fake_data), limiar = max(50, 1.5*10) = 50px.
    Bola em (102,102) com jogador em (100,100) → dist ≈ 3px → dentro do limiar.
    """
    return PassDetector(
        proximity_limit=proximity_limit,
        distance_factor=distance_factor,
        min_frames=min_frames,
        display_duration=10,
        referee_class_id=3,
        possession_grace_frames=grace,
        debug=debug,
    )


def run(detector, sequence, players, fps=30.0):
    """Roda uma sequência [(frame_num, ball)] no detector."""
    for frame_num, ball in sequence:
        detector.update(frame_num, fps, players, ball)


# ── PossessionCounter ────────────────────────────────────────────────────────

class TestPossessionCounter:

    def test_starts_zeroed(self):
        pc = PossessionCounter()
        assert pc.percentage(0) == 0.0
        assert pc.percentage(1) == 0.0
        assert pc.total_frames == 0

    def test_single_team_100_percent(self):
        pc = PossessionCounter()
        for _ in range(10):
            pc.update(0)
        assert pc.percentage(0) == 100.0
        assert pc.percentage(1) == 0.0

    def test_equal_split(self):
        pc = PossessionCounter()
        for _ in range(5):
            pc.update(0)
        for _ in range(5):
            pc.update(1)
        assert pc.percentage(0) == 50.0
        assert pc.percentage(1) == 50.0

    def test_none_excluded_from_denominator(self):
        """Frames sem posse não entram no denominador do %."""
        pc = PossessionCounter()
        for _ in range(10):
            pc.update(0)
        for _ in range(10):
            pc.update(None)   # sem posse
        # denominador = 10 (só frames com posse), não 20
        assert pc.percentage(0) == 100.0

    def test_summary_keys(self):
        pc = PossessionCounter()
        pc.update(0)
        s = pc.summary()
        assert "pct_time_0" in s
        assert "pct_time_1" in s
        assert "frames_total" in s

    def test_percentages_sum_100_when_no_none(self):
        pc = PossessionCounter()
        for i in range(7):
            pc.update(0)
        for i in range(3):
            pc.update(1)
        assert abs(pc.percentage(0) + pc.percentage(1) - 100.0) < 0.01


# ── PassDetector básico ──────────────────────────────────────────────────────

class TestPassDetectorBasic:

    def test_update_does_not_crash(self):
        det = make_detector()
        run(det, [(1, create_ball(102, 102))],
            merge_players(create_player(1, 100, 100)))

    def test_first_possession_assigned(self):
        det = make_detector()
        players = merge_players(create_player(1, 100, 100, team_id=0))
        det.update(1, 30.0, players, create_ball(102, 102))
        assert det.posse_atual is not None
        assert det.posse_atual[0] == 1

    def test_empty_players_no_crash(self):
        det = make_detector()
        det.update(1, 30.0, sv.Detections.empty(), create_ball(100, 100))
        assert det.posse_atual is None

    def test_empty_ball_no_crash(self):
        det = make_detector()
        players = merge_players(create_player(1, 100, 100))
        det.update(1, 30.0, players, sv.Detections.empty())


# ── Árbitro ──────────────────────────────────────────────────────────────────

class TestReferee:

    def test_referee_never_gets_possession(self):
        det = make_detector()
        referee = create_player(99, 102, 102, team_id=3)
        player  = create_player(1,  300, 300, team_id=0)
        players = merge_players(referee, player)

        det.update(1, 30.0, players, create_ball(102, 102))

        if det.posse_atual is not None:
            _, team_str, _ = det.posse_atual
            assert "3" not in team_str

    def test_only_referee_no_possession(self):
        det = make_detector()
        referee = create_player(99, 102, 102, team_id=3)
        det.update(1, 30.0, referee, create_ball(102, 102))
        assert det.posse_atual is None


# ── Janela de graça (ball missing) ──────────────────────────────────────────

class TestGraceWindow:

    def test_possession_held_during_grace(self):
        """Posse deve ser mantida enquanto bola ausente dentro da janela."""
        det = make_detector(grace=3)
        players = merge_players(create_player(1, 100, 100, team_id=0))

        det.update(1, 30.0, players, create_ball(102, 102))
        assert det.posse_atual is not None

        for frame in range(2, 5):   # 3 frames sem bola — dentro da graça
            det.update(frame, 30.0, players, sv.Detections.empty())
            assert det.posse_atual is not None, f"posse zerada prematuramente no frame {frame}"

    def test_possession_cleared_after_grace(self):
        """Posse deve ser zerada ao esgotar janela de graça."""
        det = make_detector(grace=2)
        players = merge_players(create_player(1, 100, 100, team_id=0))

        det.update(1, 30.0, players, create_ball(102, 102))

        for frame in range(2, 10):  # muito além da graça
            det.update(frame, 30.0, players, sv.Detections.empty())

        assert det.posse_atual is None

    def test_possession_recovers_after_ball_returns(self):
        """Bola volta → posse deve ser reassignada corretamente."""
        det = make_detector(grace=5)
        players = merge_players(create_player(1, 100, 100, team_id=0))

        det.update(1, 30.0, players, create_ball(102, 102))

        # 2 frames sem bola (dentro da graça)
        det.update(2, 30.0, players, sv.Detections.empty())
        det.update(3, 30.0, players, sv.Detections.empty())

        # Bola volta
        det.update(4, 30.0, players, create_ball(102, 102))
        assert det.posse_atual is not None
        assert det.posse_atual[0] == 1


# ── Detecção de passes ───────────────────────────────────────────────────────

class TestPassDetection:

    def test_pass_detected_same_team(self):
        det = make_detector(min_frames=1)
        player_a = create_player(1, 100, 100, team_id=0)
        player_b = create_player(2, 200, 200, team_id=0)
        players = merge_players(player_a, player_b)

        det.update(1, 30.0, players, create_ball(102, 102))  # A
        det.update(2, 30.0, players, create_ball(202, 202))  # B → passe

        assert len(det.passes_list) == 1
        p = det.passes_list[0]
        assert p["de"] == 1
        assert p["para"] == 2
        assert p["time"] == "Time 0"

    def test_no_pass_between_opponents(self):
        det = make_detector(min_frames=1)
        player_a = create_player(1, 100, 100, team_id=0)
        player_b = create_player(2, 200, 200, team_id=1)
        players = merge_players(player_a, player_b)

        det.update(1, 30.0, players, create_ball(102, 102))
        det.update(2, 30.0, players, create_ball(202, 202))

        assert len(det.passes_list) == 0

    def test_no_pass_when_owner_unstable(self):
        """Com min_frames=3, posse de 1 frame não deve gerar passe."""
        det = make_detector(min_frames=3)
        player_a = create_player(1, 100, 100, team_id=0)
        player_b = create_player(2, 200, 200, team_id=0)
        players = merge_players(player_a, player_b)

        det.update(1, 30.0, players, create_ball(102, 102))  # A — 1 frame
        det.update(2, 30.0, players, create_ball(202, 202))  # B imediato

        assert len(det.passes_list) == 0

    def test_pass_after_stable_possession(self):
        """Com min_frames=3, passe só conta após 3 frames de posse estável."""
        det = make_detector(min_frames=3)
        player_a = create_player(1, 100, 100, team_id=0)
        player_b = create_player(2, 200, 200, team_id=0)
        players = merge_players(player_a, player_b)

        # A mantém posse por 3 frames
        det.update(1, 30.0, players, create_ball(102, 102))
        det.update(2, 30.0, players, create_ball(102, 102))
        det.update(3, 30.0, players, create_ball(102, 102))
        # B assume
        det.update(4, 30.0, players, create_ball(202, 202))

        assert len(det.passes_list) == 1

    def test_pass_distance_positive(self):
        det = make_detector(min_frames=1)
        player_a = create_player(1, 100, 100, team_id=0)
        player_b = create_player(2, 300, 300, team_id=0)
        players = merge_players(player_a, player_b)

        det.update(1, 30.0, players, create_ball(102, 102))
        det.update(2, 30.0, players, create_ball(302, 302))

        assert det.passes_list[0]["distancia_px"] > 0

    def test_long_transfer_is_pass_not_shot(self):
        det = make_detector(min_frames=1)
        players = merge_players(
            create_player(1, 100, 100, team_id=0),
            create_player(2, 700, 100, team_id=0),
        )

        det.update(1, 30.0, players, create_ball(102, 102))
        det.update(2, 30.0, players, create_ball(702, 102))

        assert det.passes_list[0]["tipo"] == "PASSE_LONGO"
        assert det.shots_list == []

    def test_pass_timestamp_correct(self):
        det = make_detector(min_frames=1)
        player_a = create_player(1, 100, 100, team_id=0)
        player_b = create_player(2, 200, 200, team_id=0)
        players = merge_players(player_a, player_b)

        fps = 25.0
        det.update(1,  fps, players, create_ball(102, 102))
        det.update(50, fps, players, create_ball(202, 202))

        assert abs(det.passes_list[0]["tempo_s"] - 50 / fps) < 0.01

    def test_multiple_passes_accumulate(self):
        det = make_detector(min_frames=1)
        player_a = create_player(1, 100, 100, team_id=0)
        player_b = create_player(2, 200, 200, team_id=0)
        players = merge_players(player_a, player_b)

        det.update(1, 30.0, players, create_ball(102, 102))  # A
        det.update(2, 30.0, players, create_ball(202, 202))  # B → passe 1
        det.update(3, 30.0, players, create_ball(102, 102))  # A → passe 2
        det.update(4, 30.0, players, create_ball(202, 202))  # B → passe 3

        assert len(det.passes_list) == 3

    def test_passes_chronological_order(self):
        det = make_detector(min_frames=1)
        player_a = create_player(1, 100, 100, team_id=0)
        player_b = create_player(2, 200, 200, team_id=0)
        players = merge_players(player_a, player_b)

        for i, ball_x in enumerate([102, 202, 102, 202], start=1):
            det.update(i * 10, 30.0, players, create_ball(ball_x, ball_x))

        frames = [p["frame"] for p in det.passes_list]
        assert frames == sorted(frames)


# ── Posse integrada ao PassDetector ─────────────────────────────────────────

class TestIntegratedPossession:

    def test_possession_counter_updates_each_frame(self):
        det = make_detector()
        players = merge_players(create_player(1, 100, 100, team_id=0))

        for i in range(1, 11):
            det.update(i, 30.0, players, create_ball(102, 102))

        s = det.possession_summary()
        assert s["frames_time_0"] == 10
        assert s["pct_time_0"] == 100.0

    def test_possession_split_between_teams(self):
        det = make_detector(min_frames=1)
        player_a = create_player(1, 100, 100, team_id=0)
        player_b = create_player(2, 500, 500, team_id=1)
        players = merge_players(player_a, player_b)

        # 6 frames com A, 4 frames com B
        for i in range(1, 7):
            det.update(i, 30.0, players, create_ball(102, 102))
        for i in range(7, 11):
            det.update(i, 30.0, players, create_ball(502, 502))

        s = det.possession_summary()
        assert s["frames_time_0"] == 6
        assert s["frames_time_1"] == 4
        assert abs(s["pct_time_0"] - 60.0) < 0.1
        assert abs(s["pct_time_1"] - 40.0) < 0.1

    def test_none_frames_excluded_from_percentage(self):
        """Frames sem posse não devem distorcer o percentual."""
        det = make_detector(grace=0)   # grace=0 → zera posse imediatamente sem bola
        players = merge_players(create_player(1, 100, 100, team_id=0))

        det.update(1, 30.0, players, create_ball(102, 102))   # time 0
        det.update(2, 30.0, players, sv.Detections.empty())   # sem posse
        det.update(3, 30.0, players, create_ball(102, 102))   # time 0

        s = det.possession_summary()
        # 2 frames de time 0, 1 frame sem posse → pct deve ser 100%
        assert s["pct_time_0"] == 100.0


# ── annotate_frame ───────────────────────────────────────────────────────────

class TestAnnotateFrame:

    def test_annotate_does_not_crash_empty(self):
        det = make_detector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.annotate_frame(frame, frame_num=1)
        assert result.shape == frame.shape

    def test_annotate_shows_possession_text(self):
        """Frame anotado deve ter pelo menos alguns pixels alterados."""
        det = make_detector()
        players = merge_players(create_player(1, 100, 100, team_id=0))
        det.update(1, 30.0, players, create_ball(102, 102))

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = det.annotate_frame(frame, frame_num=1)

        # Pelo menos um pixel deve ter sido modificado (texto de posse)
        assert not np.array_equal(frame, annotated)
