"""Detecção de chutes a partir da aceleração da bola."""

from typing import Dict, List, Optional

import cv2
import numpy as np
import supervision as sv


class ShotDetector:
    """Registra uma finalização quando a bola acelera perto de um jogador."""

    def __init__(self, speed_threshold: float = 25.0,
                 proximity_limit: float = 120.0, cooldown_frames: int = 5,
                 min_speed_ratio: float = 2.0, display_duration: int = 60,
                 referee_class_id: int = 3, debug: bool = False,
                 confirmation_frames: int = 1, min_progress_px: float = 30.0,
                 direction_consistency: float = 0.75,
                 min_horizontal_ratio: float = 0.0,
                 event_type: str = "CHUTE"):
        self.speed_threshold = speed_threshold
        self.proximity_limit = proximity_limit
        self.cooldown_frames = cooldown_frames
        self.min_speed_ratio = min_speed_ratio
        self.display_duration = display_duration
        self.referee_class_id = referee_class_id
        self.debug = debug
        self.confirmation_frames = max(1, confirmation_frames)
        self.min_progress_px = min_progress_px
        self.direction_consistency = direction_consistency
        self.min_horizontal_ratio = min_horizontal_ratio
        self.event_type = event_type
        self.shots_list: List[Dict] = []
        self._last_ball: Optional[np.ndarray] = None
        self._last_speed = 0.0
        self._last_shot_frame: Optional[int] = None
        self._candidate: Optional[Dict] = None

    def update(self, frame_num: int, fps: float,
               player_detections: sv.Detections,
               ball_detections: sv.Detections,
               frame_width: Optional[int] = None) -> None:
        if len(ball_detections) == 0:
            self._last_ball = None
            self._last_speed = 0.0
            self._candidate = None
            return

        ball = ball_detections.get_anchors_coordinates(sv.Position.CENTER)[0]
        if self._last_ball is None:
            self._last_ball = ball
            return

        speed = float(np.linalg.norm(ball - self._last_ball))

        if self._candidate is not None:
            self._update_candidate(frame_num, fps, ball, frame_width)

        baseline = max(self._last_speed, 1.0)
        accelerated = (speed >= self.speed_threshold
                       and speed >= baseline * self.min_speed_ratio)
        outside_cooldown = (self._last_shot_frame is None or
                            frame_num - self._last_shot_frame > self.cooldown_frames)

        if accelerated and outside_cooldown and len(player_detections) > 0:
            players = player_detections[
                player_detections.class_id != self.referee_class_id
            ]
            if len(players) > 0:
                anchors = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                distances = np.linalg.norm(anchors - self._last_ball, axis=1)
                nearest = int(np.argmin(distances))
                if float(distances[nearest]) <= self.proximity_limit:
                    tracker_id = players.tracker_id[nearest]
                    team_id = int(players.class_id[nearest])
                    self._candidate = {
                        "frame": int(frame_num),
                        "start": self._last_ball.copy(),
                        "last": ball.copy(),
                        "direction": ball - self._last_ball,
                        "initial_speed": speed,
                        "jogador_id": int(tracker_id) if tracker_id is not None else -1,
                        "time": f"Time {team_id}",
                    }
                    if self.confirmation_frames == 1:
                        self._confirm_candidate(frame_num, fps, ball)

        self._last_ball = ball
        self._last_speed = speed

    def _update_candidate(
        self, frame_num: int, fps: float, ball: np.ndarray,
        frame_width: Optional[int],
    ) -> None:
        candidate = self._candidate
        if candidate is None:
            return
        age = frame_num - candidate["frame"]
        displacement = ball - candidate["start"]
        progress = float(np.linalg.norm(displacement))
        initial_direction = candidate["direction"]
        denominator = np.linalg.norm(initial_direction) * max(progress, 1e-8)
        consistency = float(np.dot(initial_direction, displacement) / denominator)
        horizontal = abs(displacement[0]) >= self.min_horizontal_ratio * abs(displacement[1])

        goalward = True
        if frame_width is not None and displacement[0] != 0:
            target_x = float(frame_width if displacement[0] > 0 else 0)
            before = abs(target_x - float(candidate["start"][0]))
            after = abs(target_x - float(ball[0]))
            goalward = after < before

        if consistency < self.direction_consistency or not horizontal or not goalward:
            self._candidate = None
            return
        candidate["last"] = ball.copy()
        candidate["consistency"] = consistency
        if age >= self.confirmation_frames - 1:
            if progress >= self.min_progress_px:
                self._confirm_candidate(frame_num, fps, ball)
            else:
                self._candidate = None

    def _confirm_candidate(self, frame_num: int, fps: float, ball: np.ndarray) -> None:
        candidate = self._candidate
        if candidate is None:
            return
        progress = float(np.linalg.norm(ball - candidate["start"]))
        confidence = min(
            1.0,
            0.4 * min(candidate["initial_speed"] / (2 * self.speed_threshold), 1.0)
            + 0.35 * min(progress / (2 * self.min_progress_px), 1.0)
            + 0.25 * float(candidate.get("consistency", 1.0)),
        )
        shot = {
            "tipo": self.event_type,
            "frame": int(candidate["frame"]),
            "tempo_s": float(candidate["frame"] / fps) if fps else 0.0,
            "jogador_id": candidate["jogador_id"],
            "de": candidate["jogador_id"],
            "para": "",
            "time": candidate["time"],
            "velocidade_px": candidate["initial_speed"],
            "distancia_px": round(progress, 1),
            "frame_confirmacao": int(frame_num),
            "confianca": round(confidence, 3),
        }
        self.shots_list.append(shot)
        self._last_shot_frame = frame_num
        self._candidate = None
        if self.debug:
            print(f"[ShotDetector] candidato confirmado: {shot}")

    def shots_summary(self) -> Dict[str, int]:
        summary: Dict[str, int] = {"total": len(self.shots_list)}
        for shot in self.shots_list:
            team = shot["time"].split()[-1]
            key = f"chutes_time_{team}"
            summary[key] = summary.get(key, 0) + 1
        return summary

    def annotate_frame(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        active = [shot for shot in self.shots_list
                  if 0 <= frame_num - shot["frame"] < self.display_duration]
        if active:
            label = "CANDIDATO A CHUTE" if self.event_type == "CANDIDATO_CHUTE" else "CHUTE!"
            cv2.putText(frame, label, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
                        cv2.LINE_AA)
        return frame
