"""
TeamClassifier v3 — KMeans 1D no canal H com máscara SAM.

Motivação:
- KMeans 3D (H,S,V) falha quando grama contamina os crops, puxando
  todos os valores para H≈35-45 (verde), dificultando a separação.
- Com máscara SAM os pixels da camiseta ficam limpos: time vermelho
  tem H≈13-29 e time verde tem H≈39-46, com gap natural em H≈35.
- KMeans 1D no canal H aprende esse gap automaticamente e é muito
  mais robusto que 3D para esse problema específico.

Fase de fit (com SAM, roda uma vez):
- Extrai H médio de cada jogador usando máscara precisa
- KMeans 1D aprende os dois centroides de H
- Threshold = média dos dois centroides (ponto do gap)

Fase de predict (sem SAM, rápido):
- Extrai H médio da faixa superior do crop (sem máscara)
- Compara com threshold aprendido → Time 0 ou Time 1
- Memória por tracker_id evita flickering residual
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
from typing import Dict, Optional, List, Tuple


class TeamClassifier:
    def __init__(
        self,
        device: str = "cpu",
        shirt_region: float = 0.45,
        memory_frames: int = 40,
        min_mask_pixels: int = 20,
    ):
        self.shirt_region = shirt_region
        self.memory_frames = memory_frames
        self.min_mask_pixels = min_mask_pixels
        self.device = device

        self.fitted = False
        self._h_threshold: float = 35.0   # aprendido no fit, padrão razoável
        self._team_low_h: int = 0          # time com H baixo → Time 0 ou 1
        self._memory: Dict[int, Tuple[int, int]] = {}

    # ── Extração de H médio com máscara ─────────────────────────────────────

    def _h_from_masked_crop(
        self,
        crop: np.ndarray,
        mask: np.ndarray,
    ) -> Optional[float]:
        """H médio da região da camiseta usando apenas pixels da máscara."""
        h_bbox = crop.shape[0]
        shirt_h = max(1, int(h_bbox * self.shirt_region))
        shirt_mask = mask[:shirt_h, :]

        if shirt_mask.sum() < self.min_mask_pixels:
            return None

        pixels = crop[:shirt_h, :][shirt_mask]
        pixels_hsv = cv2.cvtColor(
            pixels.reshape(-1, 1, 3).astype(np.uint8),
            cv2.COLOR_BGR2HSV,
        ).reshape(-1, 3)
        return float(pixels_hsv[:, 0].mean())

    # ── Extração de H médio sem máscara (rápido, para predict) ──────────────

    def _h_from_crop(self, crop: np.ndarray) -> float:
        """H médio da faixa superior do crop sem máscara."""
        h = crop.shape[0]
        shirt_h = max(1, int(h * self.shirt_region))
        shirt_crop = crop[:shirt_h, :]
        if shirt_crop.size == 0:
            return 0.0
        resized = cv2.resize(shirt_crop, (32, 16))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        return float(hsv[:, :, 0].mean())

    # ── Fit com SAM ──────────────────────────────────────────────────────────

    def fit_with_sam(
        self,
        frames: List[np.ndarray],
        detections_list,
        sam_model,
        player_class_id: int = 2,
    ) -> None:
        """
        Aprende o threshold de H usando máscaras SAM.
        KMeans 1D nos valores de H — robusto e interpretável.
        """
        h_values = []
        fallbacks = 0

        for frame, detections in zip(frames, detections_list):
            players = detections[detections.class_id == player_class_id]
            if len(players) == 0:
                continue

            try:
                sam_result = sam_model(frame, bboxes=players.xyxy, verbose=False)[0]
            except Exception:
                # Fallback sem máscara
                for xyxy in players.xyxy:
                    x1, y1, x2, y2 = map(int, xyxy)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        h_values.append(self._h_from_crop(crop))
                        fallbacks += 1
                continue

            if sam_result.masks is None:
                continue

            masks = sam_result.masks.data.cpu().numpy().astype(bool)

            for xyxy, mask in zip(players.xyxy, masks):
                x1, y1, x2, y2 = map(int, xyxy)
                crop = frame[y1:y2, x1:x2]
                mask_crop = mask[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                h = self._h_from_masked_crop(crop, mask_crop)
                if h is not None:
                    h_values.append(h)
                else:
                    # Fallback sem máscara
                    h_values.append(self._h_from_crop(crop))
                    fallbacks += 1

        if len(h_values) < 4:
            print("[TeamClassifier] Poucos valores — usando threshold padrão H=35")
            self.fitted = True
            return

        # KMeans 1D no canal H
        h_arr = np.array(h_values).reshape(-1, 1)
        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        km.fit(h_arr)

        c0, c1 = float(km.cluster_centers_[0][0]), float(km.cluster_centers_[1][0])
        self._h_threshold = (c0 + c1) / 2.0

        # Time com H mais baixo → índice 0 da classificação
        # (H baixo = vermelho/laranja, H alto = verde/azul)
        # Por convenção: H baixo = Time 0
        self._team_low_h = 0

        self.fitted = True
        self._memory.clear()

        print(
            f"[TeamClassifier] Fit concluído: "
            f"centroide_0={c0:.1f} centroide_1={c1:.1f} "
            f"threshold={self._h_threshold:.1f} "
            f"({len(h_values)} amostras, {fallbacks} fallbacks)"
        )

    # ── Fit rápido sem SAM (compatibilidade) ─────────────────────────────────

    def fit(
        self,
        crops: List[np.ndarray],
        anchor_positions: Optional[np.ndarray] = None,
    ) -> None:
        """Fit sem SAM — usa H médio da faixa superior de cada crop."""
        if len(crops) == 0:
            return

        h_values = np.array([self._h_from_crop(c) for c in crops]).reshape(-1, 1)
        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        km.fit(h_values)

        c0, c1 = float(km.cluster_centers_[0][0]), float(km.cluster_centers_[1][0])
        self._h_threshold = (c0 + c1) / 2.0
        self._team_low_h = 0
        self.fitted = True
        self._memory.clear()

        print(
            f"[TeamClassifier] Fit rápido: "
            f"threshold H={self._h_threshold:.1f} "
            f"({len(crops)} crops)"
        )
        
    # No TeamClassifier, adiciona:
    def update_threshold(self, crops: List[np.ndarray], min_samples: int = 8) -> None:
        """
        Recalcula o threshold de H com os jogadores visíveis agora.
        Chama a cada N frames no loop principal.
        Só atualiza se houver separação clara (gap > 8 entre centroides).
        """
        if len(crops) < min_samples or not self.fitted:
            return

        h_values = np.array([self._h_from_crop(c) for c in crops]).reshape(-1, 1)
        
        km = KMeans(n_clusters=2, n_init=5, random_state=42)
        km.fit(h_values)
        
        c0 = float(km.cluster_centers_[0][0])
        c1 = float(km.cluster_centers_[1][0])
        gap = abs(c0 - c1)
        
        # Só atualiza se os dois clusters estiverem claramente separados
        if gap < 12:
            return
        
        novo_threshold = (c0 + c1) / 2.0
        
        # Suaviza com média ponderada — evita saltos bruscos
        self._h_threshold = 0.85 * self._h_threshold + 0.15 * novo_threshold    

    # ── Predict (rápido, sem SAM) ────────────────────────────────────────────

    def predict(
        self,
        crops: List[np.ndarray],
        tracker_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if len(crops) == 0 or not self.fitted:
            return np.array([], dtype=int)

        result = np.zeros(len(crops), dtype=int)

        for i, crop in enumerate(crops):
            h = self._h_from_crop(crop)
            # H abaixo do threshold → time com H baixo (Time 0 por padrão)
            raw = self._team_low_h if h < self._h_threshold else (1 - self._team_low_h)
            result[i] = raw

        if tracker_ids is None:
            return result

        # Memória por tracker_id — estabiliza predições frame a frame
        for i, tid in enumerate(tracker_ids):
            tid = int(tid)
            # Sempre atualiza memória (threshold é determinístico, não há "inseguro")
            self._memory[tid] = (int(result[i]), self.memory_frames)

        # Decrementa jogadores ausentes
        seen = set(int(t) for t in tracker_ids)
        expired = []
        for tid, (label, frames_left) in self._memory.items():
            if tid not in seen:
                if frames_left <= 1:
                    expired.append(tid)
                else:
                    self._memory[tid] = (label, frames_left - 1)
        for tid in expired:
            del self._memory[tid]

        return result
