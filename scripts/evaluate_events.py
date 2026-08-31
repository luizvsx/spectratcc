#!/usr/bin/env python3
"""Avalia o CSV do SPECTRA contra eventos anotados manualmente."""

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tcc_common.evaluation import (
    evaluate_timed_events,
    read_ground_truth,
    read_spectra_events,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detected", required=True, help="CSV gerado pelo SPECTRA")
    parser.add_argument("--ground-truth", required=True, help="CSV anotado manualmente")
    parser.add_argument("--tolerance", type=float, default=1.0, help="tolerância em segundos")
    args = parser.parse_args()

    detected = read_spectra_events(args.detected)
    truth = read_ground_truth(args.ground_truth)
    metrics = evaluate_timed_events(detected, truth, args.tolerance)

    print(f"Eventos detectados: {len(detected)}")
    print(f"Eventos reais: {len(truth)}")
    print(f"Verdadeiros positivos: {metrics.true_positives}")
    print(f"Falsos positivos: {metrics.false_positives}")
    print(f"Falsos negativos: {metrics.false_negatives}")
    print(f"Precisão: {metrics.precision:.3f}")
    print(f"Recall: {metrics.recall:.3f}")
    print(f"F1: {metrics.f1:.3f}")


if __name__ == "__main__":
    main()
