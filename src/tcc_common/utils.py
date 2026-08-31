import numpy as np
from typing import Tuple

def dist_euclidiana(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calcula a distância euclidiana entre dois pontos."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
