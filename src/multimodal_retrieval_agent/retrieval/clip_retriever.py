from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image


@dataclass(frozen=True)
class RetrievalItem:
    id: int
    file_path: str
    label: str
    label_cn: str
    score: float


class TaiyiClipRetriever:
    """Minimal Taiyi-CLIP text-to-image retriever.

    This class keeps heavy model imports lazy so the rest of the project can be
    used without installing GPU/model dependencies.
    """

    def __init__(self, device: str | None = None):
        from transformers import BertForSequenceClassification, BertTokenizer, CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese")
        self.text_encoder = BertForSequenceClassification.from_pretrained(
            "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese"
        ).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.image_encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(self.device)

    def encode_text(self, text: str) -> torch.Tensor:
        tokens = self.text_tokenizer(text, return_tensors="pt", padding=True)["input_ids"].to(self.device)
        with torch.no_grad():
            features = self.text_encoder(tokens).logits
        return features / torch.linalg.norm(features, dim=1, keepdim=True)

    def encode_image(self, image_path: str | Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.image_encoder.get_image_features(**inputs)
        if isinstance(features, dict):
            features = features["pooler_output"]
        return features / torch.linalg.norm(features, dim=1, keepdim=True)

    def search_text(self, text: str, metadata: list[dict], top_k: int = 5) -> list[RetrievalItem]:
        text_features = self.encode_text(text)
        results = []
        for item in metadata:
            path = Path(item["file_path"])
            if not path.exists():
                continue
            image_features = self.encode_image(path)
            score = float((self.image_encoder.logit_scale.exp() * image_features @ text_features.T).item())
            results.append(
                RetrievalItem(
                    id=int(item["id"]),
                    file_path=str(path),
                    label=item.get("label", ""),
                    label_cn=item.get("label_cn", ""),
                    score=score,
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]
