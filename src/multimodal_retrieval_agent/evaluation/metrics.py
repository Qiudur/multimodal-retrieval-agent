from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BinaryMetrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def precision_recall_f1(tp: int, fp: int, fn: int) -> BinaryMetrics:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(precision, recall, f1, tp, fp, fn)


def threshold_metrics(pos_scores: Iterable[float], neg_scores: Iterable[float], threshold: float) -> BinaryMetrics:
    pos = list(pos_scores)
    neg = list(neg_scores)
    tp = sum(score >= threshold for score in pos)
    fp = sum(score >= threshold for score in neg)
    fn = sum(score < threshold for score in pos)
    return precision_recall_f1(tp, fp, fn)


def top_k_precision(scores: Iterable[tuple[float, int]], k: int) -> float:
    ranked = sorted(scores, key=lambda item: item[0], reverse=True)[:k]
    return sum(label for _, label in ranked) / k if k else 0.0
