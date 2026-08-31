import cv2
import numpy as np
import supervision as sv
from typing import Optional, Tuple, Dict, List

REFEREE_CLASS_ID = 3


class PossessionCounter:
    """
    Conta frames de posse por time e calcula percentual acumulado.
    Atualizado a cada frame, independente de passes.
    """

    def __init__(self, team_ids: Tuple[int, int] = (0, 1)):
        self.team_ids = team_ids
        self._frames: Dict[int, int] = {t: 0 for t in team_ids}
        self._frames["none"] = 0
        self.total_frames = 0

    def update(self, team_id: Optional[int]) -> None:
        self.total_frames += 1
        key = team_id if team_id in self.team_ids else "none"
        self._frames[key] += 1

    def percentage(self, team_id: int) -> float:
        contested = sum(self._frames[t] for t in self.team_ids)
        if contested == 0:
            return 0.0
        return 100.0 * self._frames[team_id] / contested

    def summary(self) -> Dict:
        return {
            "frames_total": self.total_frames,
            "frames_sem_posse": self._frames["none"],
            **{f"frames_time_{t}": self._frames[t] for t in self.team_ids},
            **{f"pct_time_{t}": round(self.percentage(t), 1) for t in self.team_ids},
        }


class PassDetector:
    """
    Detecta passes e chutes, rastreia posse de bola frame a frame.

    Classificação de eventos:
    - distancia_px < short_pass_threshold  → PASSE_CURTO
    - distancia_px >= short_pass_threshold → PASSE_LONGO

    Chutes são responsabilidade do ShotDetector, que analisa a trajetória da bola.

    passes_list: contém PASSE_CURTO e PASSE_LONGO
    shots_list:  contém apenas CHUTE
    """

    def __init__(
        self,
        proximity_limit: int = 80,
        distance_factor: float = 1.2,
        min_frames: int = 3,
        display_duration: int = 60,
        referee_class_id: int = REFEREE_CLASS_ID,
        possession_grace_frames: int = 8,
        team_ids: Tuple[int, int] = (0, 1),
        short_pass_threshold: float = 150.0,  # px — abaixo = passe curto
        shot_threshold: float = 500.0,        # px — acima = chute
        debug: bool = False,
    ):
        self.proximity_limit = proximity_limit
        self.distance_factor = distance_factor
        self.min_frames = min_frames
        self.display_duration = display_duration
        self.referee_class_id = referee_class_id
        self.possession_grace_frames = possession_grace_frames
        self.short_pass_threshold = short_pass_threshold
        self.shot_threshold = shot_threshold
        self.debug = debug

        # Estado de posse
        self.posse_atual: Optional[Tuple[int, str, np.ndarray]] = None
        self.owner_frames: int = 0

        # Bola
        self.last_ball_center: Optional[np.ndarray] = None
        self.ball_missing: int = 0

        # Eventos separados
        self.passes_list: List[Dict] = []   # PASSE_CURTO + PASSE_LONGO
        self.shots_list: List[Dict] = []    # CHUTE
        self.eventos_display: List[Dict] = []

        # Contador de posse
        self.possession_counter = PossessionCounter(team_ids=team_ids)

    def _classify_event(self, dist: float) -> str:
        """Classifica o tipo de evento pela distância entre jogadores."""
        if dist < self.short_pass_threshold:
            return "PASSE_CURTO"
        return "PASSE_LONGO"

    def _threshold(self, player_xyxy: np.ndarray) -> float:
        h = float(player_xyxy[3] - player_xyxy[1])
        return max(self.proximity_limit, self.distance_factor * h)

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[PassDetector] {msg}")

    def _resolve_ball(self, ball_detections: sv.Detections) -> Optional[np.ndarray]:
        if len(ball_detections) > 0:
            center = ball_detections.get_anchors_coordinates(sv.Position.CENTER)[0]
            self.last_ball_center = center
            self.ball_missing = 0
            return center

        self.ball_missing += 1

        if self.last_ball_center is not None and self.ball_missing <= self.possession_grace_frames:
            self._log(f"bola ausente {self.ball_missing}/{self.possession_grace_frames} — usando última posição")
            return self.last_ball_center

        self._log("bola ausente além da graça — sem posição confiável")
        return None

    def update(
        self,
        frame_num: int,
        fps: float,
        player_detections: sv.Detections,
        ball_detections: sv.Detections,
    ) -> None:

        # 1. Resolve bola
        bola_coords = self._resolve_ball(ball_detections)

        if bola_coords is None:
            if self.posse_atual is not None:
                self._log(f"frame {frame_num}: bola perdida → zerando posse")
            self.posse_atual = None
            self.owner_frames = 0
            self.possession_counter.update(None)
            return

        # 2. Filtra árbitros — só jogadores e goleiros com class_id 0 ou 1
        mask_valido = player_detections.class_id != self.referee_class_id
        players_only = player_detections[mask_valido]

        if len(players_only) == 0:
            self._log(f"frame {frame_num}: nenhum jogador detectado")
            self.possession_counter.update(None)
            return

        # 3. Distâncias bola → pé do jogador
        player_anchors = players_only.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        xyxy = players_only.xyxy
        dists = np.linalg.norm(player_anchors - bola_coords, axis=1)

        idx_nearest = int(np.argmin(dists))
        nearest_dist = float(dists[idx_nearest])
        threshold = self._threshold(xyxy[idx_nearest])

        self._log(
            f"frame {frame_num}: jogador mais próximo "
            f"id={players_only.tracker_id[idx_nearest]} "
            f"class={players_only.class_id[idx_nearest]} "
            f"dist={nearest_dist:.1f}px limiar={threshold:.1f}px"
        )

        # 4. Bola solta
        if nearest_dist >= threshold:
            self._log(f"frame {frame_num}: bola solta — mantendo posse")
            current_team = (
                int(self.posse_atual[1].split()[-1])
                if self.posse_atual else None
            )
            self.possession_counter.update(current_team)
            return

        # 5. Atualiza dono
        new_owner_id     = int(players_only.tracker_id[idx_nearest])
        new_owner_team   = int(players_only.class_id[idx_nearest])
        new_owner_str    = f"Time {new_owner_team}"
        new_owner_coords = player_anchors[idx_nearest]

        self.possession_counter.update(new_owner_team)

        if self.posse_atual is None:
            self.posse_atual = (new_owner_id, new_owner_str, new_owner_coords)
            self.owner_frames = 1
            self._log(f"frame {frame_num}: primeira posse → jogador {new_owner_id} ({new_owner_str})")
            return

        prev_id, prev_team_str, prev_coords = self.posse_atual

        if new_owner_id == prev_id:
            self.owner_frames += 1
            self.posse_atual = (new_owner_id, new_owner_str, new_owner_coords)
            self._log(f"frame {frame_num}: posse mantida jogador {new_owner_id} ({self.owner_frames} frames)")
            return

        # 6. Mudança de dono
        self._log(
            f"frame {frame_num}: mudança de dono {prev_id}→{new_owner_id} "
            f"({prev_team_str}→{new_owner_str}) "
            f"owner_frames={self.owner_frames} min={self.min_frames}"
        )

        if self.owner_frames >= self.min_frames and prev_team_str == new_owner_str:
            dist_px = float(np.linalg.norm(prev_coords - new_owner_coords))
            tempo_s = round(frame_num / float(fps), 2)
            tipo = self._classify_event(dist_px)

            evento = {
                "frame":        frame_num,
                "tempo_s":      tempo_s,
                "de":           prev_id,
                "para":         new_owner_id,
                "time":         prev_team_str,
                "distancia_px": round(dist_px, 1),
                "tipo":         tipo,
            }

            if tipo == "CHUTE":
                self.shots_list.append(evento)
                num = len(self.shots_list)
                self._log(f"CHUTE #{num} detectado: {evento}")
                texto = (
                    f"CHUTE #{num}: {tempo_s:.1f}s | "
                    f"{prev_team_str} (jogador {prev_id}) "
                    f"dist={dist_px:.0f}px"
                )
            else:
                self.passes_list.append(evento)
                num = len(self.passes_list)
                self._log(f"{tipo} #{num} detectado: {evento}")
                texto = (
                    f"{tipo} #{num}: {tempo_s:.1f}s | "
                    f"{prev_team_str} ({prev_id}→{new_owner_id}) "
                    f"dist={dist_px:.0f}px"
                )

            self.eventos_display.append({
                "texto":        texto,
                "frame_inicio": frame_num,
                "duracao":      self.display_duration,
                "tipo":         tipo,
            })

        else:
            reason = (
                "posse instável (owner_frames < min_frames)"
                if self.owner_frames < self.min_frames
                else "times diferentes (interceptação)"
            )
            self._log(f"frame {frame_num}: mudança de dono sem evento — {reason}")

        self.posse_atual = (new_owner_id, new_owner_str, new_owner_coords)
        self.owner_frames = 1

    def possession_summary(self) -> Dict:
        return self.possession_counter.summary()

    def events_summary(self) -> Dict:
        """Resumo de passes e chutes por time."""
        summary = {
            "total_passes": len(self.passes_list),
            "total_chutes": len(self.shots_list),
        }
        for team in [0, 1]:
            t = f"Time {team}"
            summary[f"passes_time_{team}"] = sum(1 for p in self.passes_list if p["time"] == t)
            summary[f"chutes_time_{team}"] = sum(1 for s in self.shots_list if s["time"] == t)
            passes_t = [p["distancia_px"] for p in self.passes_list if p["time"] == t]
            summary[f"dist_media_passes_time_{team}"] = (
                round(sum(passes_t) / len(passes_t), 1) if passes_t else 0.0
            )
        return summary

    def annotate_frame(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        annotated = frame.copy()

        # Eventos de passe/chute
        for i in range(len(self.eventos_display) - 1, -1, -1):
            ev = self.eventos_display[i]
            if frame_num < ev["frame_inicio"] + ev["duracao"]:
                # Cor diferente para chute vs passe
                cor = (0, 220, 255) if ev["tipo"] == "CHUTE" else (255, 255, 255)
                overlay = annotated.copy()
                cv2.rectangle(overlay, (10, 10), (960, 60), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
                cv2.putText(
                    annotated, ev["texto"], (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2,
                )
            else:
                self.eventos_display.pop(i)

        # Contador de posse
        summary = self.possession_counter.summary()
        pct0 = summary.get("pct_time_0", 0.0)
        pct1 = summary.get("pct_time_1", 0.0)

        h, w = annotated.shape[:2]
        bar_x = w - 220
        bar_y = 15

        overlay = annotated.copy()
        cv2.rectangle(overlay, (bar_x - 10, bar_y - 5), (w - 10, bar_y + 115), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

        ev_s = self.events_summary()

        cv2.putText(annotated, f"Posse T0: {pct0:.1f}%",
                    (bar_x, bar_y + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (60, 60, 255), 2)
        cv2.putText(annotated, f"Posse T1: {pct1:.1f}%",
                    (bar_x, bar_y + 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 60, 60), 2)
        cv2.putText(annotated,
                    f"Passes T0:{ev_s['passes_time_0']} T1:{ev_s['passes_time_1']}",
                    (bar_x, bar_y + 66), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)
        cv2.putText(annotated,
                    f"Chutes T0:{ev_s['chutes_time_0']} T1:{ev_s['chutes_time_1']}",
                    (bar_x, bar_y + 88), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 220, 255), 1)

        return annotated
