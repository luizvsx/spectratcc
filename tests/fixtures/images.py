"""Imagens sintéticas para testar classificação sem arquivos externos."""

import cv2
import numpy as np


def solid_hsv_crop(hue: int, *, saturation: int = 220, value: int = 220) -> np.ndarray:
    hsv = np.full((40, 20, 3), (hue, saturation, value), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
