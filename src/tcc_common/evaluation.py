"""Comparação dos eventos detectados com anotações manuais."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from tcc_common.metrics import EventMetrics, metrics_from_counts


@dataclass(frozen=True)
class TimedEvent:
    tipo: str
    tempo_s: float
    time: str


def _normalize_event(row: dict) -> TimedEvent:
    return TimedEvent(
        tipo=row["tipo"].strip().upper(),
        tempo_s=float(row["tempo_s"]),
        time=row.get("time", "").strip().lower(),
    )


def read_ground_truth(path: str) -> List[TimedEvent]:
    """Lê um CSV manual com colunas tipo, tempo_s e time."""
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"tipo", "tempo_s", "time"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("ground truth deve conter: tipo, tempo_s, time")
        return [_normalize_event(row) for row in reader]


def read_spectra_events(path: str) -> List[TimedEvent]:
    """Extrai eventos da seção EVENTOS do CSV gerado pelo SPECTRA."""
    events = []
    in_events = False
    headers = None
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if not row:
                continue
            if row[0].strip() == "## EVENTOS":
                in_events = True
                headers = None
                continue
            if row[0].strip().startswith("##") and in_events:
                break
            if not in_events:
                continue
            if headers is None:
                headers = [value.strip() for value in row]
                continue
            values = dict(zip(headers, row))
            events.append(_normalize_event(values))
    return events


def evaluate_timed_events(
    detected: Iterable[TimedEvent],
    ground_truth: Iterable[TimedEvent],
    tolerance_s: float = 1.0,
) -> EventMetrics:
    """Casa eventos do mesmo tipo/time dentro da tolerância temporal."""
    if tolerance_s < 0:
        raise ValueError("tolerance_s não pode ser negativa")

    detected = list(detected)
    truth = list(ground_truth)
    unmatched_truth = set(range(len(truth)))
    true_positives = 0

    for event in detected:
        candidates = [
            index for index in unmatched_truth
            if truth[index].tipo == event.tipo
            and truth[index].time == event.time
            and abs(truth[index].tempo_s - event.tempo_s) <= tolerance_s
        ]
        if candidates:
            nearest = min(
                candidates,
                key=lambda index: abs(truth[index].tempo_s - event.tempo_s),
            )
            unmatched_truth.remove(nearest)
            true_positives += 1

    return metrics_from_counts(
        true_positives=true_positives,
        false_positives=len(detected) - true_positives,
        false_negatives=len(unmatched_truth),
    )
