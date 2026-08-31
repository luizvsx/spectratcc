"""Métricas de avaliação dos eventos detectados pelo SPECTRA."""

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable


@dataclass(frozen=True)
class EventMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def metrics_from_counts(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> EventMetrics:
    """Monta as métricas a partir da matriz de contagem dos eventos."""
    predicted = true_positives + false_positives
    expected = true_positives + false_negatives
    precision = true_positives / predicted if predicted else 0.0
    recall = true_positives / expected if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return EventMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def compute_event_metrics(
    detected: Iterable[Hashable],
    ground_truth: Iterable[Hashable],
) -> EventMetrics:
    """Calcula precisão, revocação e F1 preservando eventos repetidos."""
    detected_counts = Counter(detected)
    truth_counts = Counter(ground_truth)

    true_positives = sum((detected_counts & truth_counts).values())
    false_positives = sum((detected_counts - truth_counts).values())
    false_negatives = sum((truth_counts - detected_counts).values())

    return metrics_from_counts(
        true_positives, false_positives, false_negatives
    )
