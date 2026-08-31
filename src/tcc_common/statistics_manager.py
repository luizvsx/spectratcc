import json
import csv
from collections import defaultdict


class StatisticsManager:

    def __init__(self):

        self.events = []

        self.player_stats = defaultdict(lambda: {
            "team": None,
            "passes": 0,
            "short_passes": 0,
            "long_passes": 0,
            "shots": 0,
            "distance": 0.0,
            "touches": 0,
        })

        self.team_possession = defaultdict(float)

        self.tracking = defaultdict(list)
        
    def add_event(self, event: dict):

        self.events.append(event)

        player = event.get("player")

        if player is None:
            return

        stats = self.player_stats[player]

        stats["team"] = event.get("team")

        event_type = event["type"]

        if event_type == "short_pass":
            stats["passes"] += 1
            stats["short_passes"] += 1

        elif event_type == "long_pass":
            stats["passes"] += 1
            stats["long_passes"] += 1

        elif event_type == "shot":
            stats["shots"] += 1

        elif event_type == "touch":
            stats["touches"] += 1
        
        
        def add_tracking(
            self,
            frame,
            tracker_id,
            x,
            y
        ):

            self.tracking[tracker_id].append({

                "frame": frame,
                "x": x,
                "y": y

            })
        
        
        def add_possession(self, team, seconds):

            self.team_possession[team] += seconds
            
            
        def summary(self):
            return {

                "players": dict(self.player_stats),

                "possession": dict(self.team_possession),

                "events": len(self.events)

            }
            
        def export_json(self, path):

            data = {

                "events": self.events,

                "players": dict(self.player_stats),

                "possession": dict(self.team_possession)

            }

            with open(path, "w", encoding="utf8") as f:

                json.dump(data, f, indent=4, ensure_ascii=False)
                
        
        def export_csv(self, path):

            with open(path, "w", newline="", encoding="utf8") as csvfile:

                writer = csv.writer(csvfile)

                writer.writerow([
                    "player",
                    "team",
                    "passes",
                    "short_passes",
                    "long_passes",
                    "shots",
                    "touches",
                    "distance"
                ])

                for player, stats in self.player_stats.items():

                    writer.writerow([

                        player,

                        stats["team"],

                        stats["passes"],

                        stats["short_passes"],

                        stats["long_passes"],

                        stats["shots"],

                        stats["touches"],

                        stats["distance"]

                    ])
