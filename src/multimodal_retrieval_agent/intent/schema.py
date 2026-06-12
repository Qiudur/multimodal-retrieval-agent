from __future__ import annotations

from dataclasses import dataclass


TEXT_TO_IMAGE = "text_to_image"
IMAGE_TO_IMAGE = "image_to_image"
NO_RETRIEVAL = "no_retrieval"


@dataclass(frozen=True)
class IntentResult:
    question: str
    needs_retrieval: int
    query_type: str
    category: str
    search_mode: str
    top_k: int | None = None

    def to_dict(self) -> dict:
        mode: list[str | int]
        if self.search_mode == "top_k":
            mode = ["TopK", self.top_k or 1]
        elif self.search_mode == "TopK":
            mode = ["TopK", self.top_k or 1]
        elif self.search_mode:
            mode = [self.search_mode]
        else:
            mode = [""]

        return {
            "question": self.question,
            "needs_retrieval": self.needs_retrieval,
            "query_type": self.query_type,
            "category": self.category,
            "search_mode": mode,
        }
