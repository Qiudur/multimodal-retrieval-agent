from __future__ import annotations

import re

from .schema import IMAGE_TO_IMAGE, NO_RETRIEVAL, TEXT_TO_IMAGE, IntentResult


CATEGORY_ALIASES = {
    "chair": ["chair", "seat"],
    "cup": ["cup", "mug"],
    "bicycle": ["bicycle", "bike"],
    "skateboard": ["skateboard"],
    "oven": ["oven"],
}

TEXT_RETRIEVAL_WORDS = ("image", "images", "photo", "photos", "picture", "pictures", "find", "search", "show")
IMAGE_REFERENCE_WORDS = ("this image", "uploaded", "similar", "same style", "look like", "like this")
COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def detect_category(question: str) -> str:
    q = question.lower()
    for category, aliases in CATEGORY_ALIASES.items():
        if any(alias.lower() in q for alias in aliases):
            return category
    return ""


def detect_top_k(question: str) -> int | None:
    q = question.lower()
    digit_match = re.search(r"(\d+)\s*(image|images|photo|photos|picture|pictures)", q)
    if digit_match:
        return int(digit_match.group(1))
    for word, value in COUNT_WORDS.items():
        if re.search(rf"\b{word}\b", q) and any(term in q for term in TEXT_RETRIEVAL_WORDS):
            return value
    if "most similar" in q or "closest" in q or "best match" in q:
        return 1
    return None


def classify_intent(question: str, has_image: bool = False) -> IntentResult:
    q = question.lower()
    category = detect_category(question)
    top_k = detect_top_k(question)
    has_retrieval_word = any(word in q for word in TEXT_RETRIEVAL_WORDS)
    image_reference = has_image or any(word in q for word in IMAGE_REFERENCE_WORDS)

    if not has_retrieval_word and not image_reference:
        return IntentResult(question, 0, NO_RETRIEVAL, "", "")

    query_type = IMAGE_TO_IMAGE if image_reference else TEXT_TO_IMAGE
    search_mode = "top_k" if top_k else "threshold"
    return IntentResult(question, 1, query_type, category, search_mode, top_k)
