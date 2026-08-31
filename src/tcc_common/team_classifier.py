"""Classifica os times pela cor circular da camiseta e estabiliza por tracker."""

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
        # Matiz é circular: 0 e 179 representam praticamente a mesma cor.
        # Cosseno/seno evita separar artificialmente os dois extremos.
        self._centers = np.array([[1.0, 0.0], [-0.5, -0.866]], dtype=float)
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

    def _circular_feature_from_pixels(
        self, pixels: np.ndarray
    ) -> Optional[np.ndarray]:
        """Representa H como um ponto no círculo, filtrando pixels sem cor."""
        if pixels.size == 0:
            return None
        hsv = cv2.cvtColor(
            pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV
        ).reshape(-1, 3)
        colorful = hsv[(hsv[:, 1] >= 40) & (hsv[:, 2] >= 30)]
        if len(colorful) < self.min_mask_pixels:
            return None
        angles = colorful[:, 0].astype(float) * (2.0 * np.pi / 180.0)
        weights = colorful[:, 1].astype(float) / 255.0
        feature = np.array(
            [np.average(np.cos(angles), weights=weights),
             np.average(np.sin(angles), weights=weights)],
            dtype=float,
        )
        norm = np.linalg.norm(feature)
        return feature / norm if norm > 1e-8 else None

    def _circular_feature_from_crop(self, crop: np.ndarray) -> np.ndarray:
        """Extrai a cor da região central da camiseta, reduzindo fundo/grama."""
        height, width = crop.shape[:2]
        shirt_h = max(1, int(height * self.shirt_region))
        margin = int(width * 0.15)
        shirt = crop[:shirt_h, margin : width - margin or width]
        if shirt.size:
            shirt = cv2.resize(shirt, (32, 16))
        feature = self._circular_feature_from_pixels(shirt)
        return feature if feature is not None else np.array([1.0, 0.0])

    @staticmethod
    def _feature_hue(feature: np.ndarray) -> float:
        return float((np.arctan2(feature[1], feature[0]) % (2 * np.pi)) * 180 / (2 * np.pi))

    def _extract_feature(self, crop: np.ndarray) -> np.ndarray:
        """Retorna a cor BGR mediana da região da camiseta."""
        h = crop.shape[0]
        shirt_h = max(1, int(h * self.shirt_region))
        shirt_crop = crop[:shirt_h, :]
        if shirt_crop.size == 0:
            return np.zeros(3, dtype=float)
        resized = cv2.resize(shirt_crop, (32, 16))
        return np.median(resized.reshape(-1, 3), axis=0).astype(float)

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
        features = []
        fallbacks = 0
        resized_masks = 0

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
                        features.append(self._circular_feature_from_crop(crop))
                        fallbacks += 1
                continue

            if sam_result.masks is None:
                continue

            masks = sam_result.masks.data.cpu().numpy().astype(bool)

            for xyxy, mask in zip(players.xyxy, masks):
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    resized_masks += 1
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                mask_crop = mask[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                shirt_h = max(1, int(crop.shape[0] * self.shirt_region))
                masked_pixels = crop[:shirt_h, :][mask_crop[:shirt_h, :]]
                feature = self._circular_feature_from_pixels(masked_pixels)
                if feature is not None:
                    features.append(feature)
                else:
                    # Fallback sem máscara
                    features.append(self._circular_feature_from_crop(crop))
                    fallbacks += 1

        if len(features) < 4:
            print("[TeamClassifier] Poucas cores — usando centros padrão")
            self.fitted = True
            return

        feature_array = np.asarray(features)
        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        km.fit(feature_array)
        centers = km.cluster_centers_
        centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-8)
        order = np.argsort([self._feature_hue(center) for center in centers])
        self._centers = centers[order]
        hues = [self._feature_hue(center) for center in self._centers]

        # Time com H mais baixo → índice 0 da classificação
        # (H baixo = vermelho/laranja, H alto = verde/azul)
        # Por convenção: H baixo = Time 0
        self._team_low_h = 0

        self.fitted = True
        self._memory.clear()

        print(
            f"[TeamClassifier] Fit concluído: "
            f"matiz_time_0={hues[0]:.1f} matiz_time_1={hues[1]:.1f} "
            f"({len(features)} amostras, {fallbacks} fallbacks, "
            f"{resized_masks} máscaras redimensionadas)"
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

        features = np.array([self._circular_feature_from_crop(c) for c in crops])
        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        km.fit(features)
        centers = km.cluster_centers_
        centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-8)
        order = np.argsort([self._feature_hue(center) for center in centers])
        self._centers = centers[order]
        self._team_low_h = 0
        self.fitted = True
        self._memory.clear()

        print(
            f"[TeamClassifier] Fit rápido: "
            f"matizes={[round(self._feature_hue(c), 1) for c in self._centers]} "
            f"({len(crops)} crops)"
        )
        
    # No TeamClassifier, adiciona:
    def update_threshold(self, crops: List[np.ndarray], min_samples: int = 8) -> None:
        """
        Atualiza suavemente os centros de cor com os jogadores visíveis.
        """
        if len(crops) < min_samples or not self.fitted:
            return

        features = np.array([self._circular_feature_from_crop(c) for c in crops])
        km = KMeans(n_clusters=2, n_init=5, random_state=42)
        km.fit(features)
        new_centers = km.cluster_centers_
        new_centers /= np.linalg.norm(new_centers, axis=1, keepdims=True).clip(min=1e-8)
        if np.linalg.norm(new_centers[0] - new_centers[1]) < 0.4:
            return

        direct = sum(np.linalg.norm(self._centers[i] - new_centers[i]) for i in (0, 1))
        swapped = sum(np.linalg.norm(self._centers[i] - new_centers[1 - i]) for i in (0, 1))
        if swapped < direct:
            new_centers = new_centers[::-1]
        self._centers = 0.85 * self._centers + 0.15 * new_centers
        self._centers /= np.linalg.norm(self._centers, axis=1, keepdims=True).clip(min=1e-8)

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
            feature = self._circular_feature_from_crop(crop)
            result[i] = int(np.argmin(np.linalg.norm(self._centers - feature, axis=1)))

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
