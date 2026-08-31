import numpy as np
import supervision as sv


def create_player(tracker_id, x, y, team_id=2):
    return sv.Detections(
        xyxy=np.array([[x, y, x + 10, y + 10]]),
        class_id=np.array([team_id]),  # IMPORTANTE: numpy array
        tracker_id=np.array([tracker_id])
    )


def create_ball(x, y):
    return sv.Detections(
        xyxy=np.array([[x, y, x + 5, y + 5]]),
        class_id=np.array([0]),
        tracker_id=np.array([-1])
    )


def merge_players(*players):
    return sv.Detections.merge(players)
