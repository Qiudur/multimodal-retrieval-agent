from __future__ import annotations

from collections import defaultdict
from typing import Callable

from multimodal_retrieval_agent.intent.schema import IMAGE_TO_IMAGE, NO_RETRIEVAL, TEXT_TO_IMAGE

LABELS = [TEXT_TO_IMAGE, IMAGE_TO_IMAGE, NO_RETRIEVAL]


def _first(value, default=""):
    if isinstance(value, list):
        return value[0] if value else default
    return value if value is not None else default


def normalize_record(record: dict) -> dict:
    question = _first(record.get("question"))
    query_type = _first(record.get("query_type"))
    needs = record.get("needs_retrieval")
    if isinstance(needs, list):
        needs = _first(needs, 0)
    if needs in ["", None]:
        needs = 0

    if not int(needs):
        label = NO_RETRIEVAL
    elif query_type == TEXT_TO_IMAGE:
        label = TEXT_TO_IMAGE
    elif query_type == IMAGE_TO_IMAGE:
        label = IMAGE_TO_IMAGE
    else:
        label = query_type or TEXT_TO_IMAGE

    return {
        "question": question,
        "label": label,
        "category": _first(record.get("category")),
        "mode": record.get("search_mode", [""]),
    }


def evaluate_intent_dataset(dataset: dict, predictor: Callable[[str], dict]) -> dict:
    counters = {label: defaultdict(int) for label in LABELS}

    for label, records in dataset.items():
        for record in records:
            gold = normalize_record(record)
            pred = normalize_record(predictor(gold["question"]))
            for candidate in LABELS:
                if pred["label"] == candidate and gold["label"] == candidate:
                    counters[candidate]["tp"] += 1
                elif pred["label"] == candidate and gold["label"] != candidate:
                    counters[candidate]["fp"] += 1
                elif pred["label"] != candidate and gold["label"] == candidate:
                    counters[candidate]["fn"] += 1

    result = {}
    for label, values in counters.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return result
