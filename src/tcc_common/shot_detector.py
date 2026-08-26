# tcc_common/shot_detector.py

import numpy as np


class ShotDetector:

    def __init__(
        self,
        short_pass_threshold=120,
        long_pass_threshold=280,
        shot_threshold=400,
    ):
        self.short_pass_threshold = short_pass_threshold
        self.long_pass_threshold = long_pass_threshold
        self.shot_threshold = shot_threshold

    def classify(
        self,
        start_ball,
        end_ball,
        start_player=None,
        end_player=None,
    ):
        """
        Classifica um evento usando apenas a distância percorrida pela bola.
        """

        distance = np.linalg.norm(
            np.array(end_ball) - np.array(start_ball)
        )

        if distance < self.short_pass_threshold:
            event = "SHORT_PASS"

        elif distance < self.long_pass_threshold:
            event = "LONG_PASS"

        elif distance < self.shot_threshold:
            event = "LONG_PASS"

        else:
            event = "SHOT"

        return {
            "event": event,
            "distance": distance,
        }
