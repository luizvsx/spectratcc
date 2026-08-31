from collections import deque

import cv2
import numpy as np
import supervision as sv


def filter_ball_detections_on_pitch(
    frame: np.ndarray,
    detections: sv.Detections,
    *,
    patch_radius: int = 18,
    min_green_ratio: float = 0.2,
) -> sv.Detections:
    """Remove candidatos cercados por arquibancada, placas ou outros fundos.

    A bola ocupa poucos pixels; por isso analisamos uma janela ao redor dela e
    exigimos uma proporção mínima de pixels com aparência de gramado.
    """
    if len(detections) == 0:
        return detections

    height, width = frame.shape[:2]
    centers = detections.get_anchors_coordinates(sv.Position.CENTER).astype(int)
    keep = []
    for center_x, center_y in centers:
        x1, x2 = max(0, center_x - patch_radius), min(width, center_x + patch_radius + 1)
        y1, y2 = max(0, center_y - patch_radius), min(height, center_y + patch_radius + 1)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            keep.append(False)
            continue
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        green = (
            (hsv[:, :, 0] >= 25)
            & (hsv[:, :, 0] <= 95)
            & (hsv[:, :, 1] >= 35)
            & (hsv[:, :, 2] >= 30)
        )
        keep.append(float(green.mean()) >= min_green_ratio)
    return detections[np.asarray(keep, dtype=bool)]


class BallAnnotator:
    """
    A class to annotate frames with circles of varying radii and colors.

    Attributes:
        radius (int): The maximum radius of the circles to be drawn.
        buffer (deque): A deque buffer to store recent coordinates for annotation.
        color_palette (sv.ColorPalette): A color palette for the circles.
        thickness (int): The thickness of the circle borders.
    """

    def __init__(self, radius: int, buffer_size: int = 5, thickness: int = 2):

        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        """
        Interpolates the radius between 1 and the maximum radius based on the index.

        Args:
            i (int): The current index in the buffer.
            max_i (int): The maximum index in the buffer.

        Returns:
            int: The interpolated radius.
        """
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Annotates the frame with circles based on detections.

        Args:
            frame (np.ndarray): The frame to annotate.
            detections (sv.Detections): The detections containing coordinates.

        Returns:
            np.ndarray: The annotated frame.
        """
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)
        for i, xy in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            for center in xy:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
        return frame


class BallTracker:
    """
    A class used to track a soccer ball's position across video frames.

    The BallTracker class maintains a buffer of recent ball positions and uses this
    buffer to predict the ball's position in the current frame by selecting the
    detection closest to the average position (centroid) of the recent positions.

    Attributes:
        buffer (collections.deque): A deque buffer to store recent ball positions.
    """
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Updates the buffer with new detections and returns the detection closest to the
        centroid of recent positions.

        Args:
            detections (sv.Detections): The current frame's ball detections.

        Returns:
            sv.Detections: The detection closest to the centroid of recent positions.
            If there are no detections, returns the input detections.
        """
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.append(xy)

        if len(detections) == 0:
            return detections

        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        return detections[[index]]
