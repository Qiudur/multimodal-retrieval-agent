from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_retrieval_agent.evaluation.intent_eval import evaluate_intent_dataset
from multimodal_retrieval_agent.intent.rule_based import classify_intent
from multimodal_retrieval_agent.utils.io import read_json


def cmd_intent(args: argparse.Namespace) -> None:
    result = classify_intent(args.question, has_image=args.has_image)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def cmd_eval_intent(args: argparse.Namespace) -> None:
    dataset = read_json(args.dataset)
    metrics = evaluate_intent_dataset(dataset, lambda question: classify_intent(question).to_dict())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def cmd_check_data(args: argparse.Namespace) -> None:
    data = read_json(args.metadata)
    images = data.get("images", [])
    missing = [item["file_path"] for item in images if not Path(item["file_path"]).exists()]
    summary = {
        "metadata": args.metadata,
        "num_images": len(images),
        "num_categories": len(data.get("info", {})),
        "missing_image_files": len(missing),
        "first_missing_examples": missing[:10],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmra", description="Multimodal retrieval agent utilities")
    sub = parser.add_subparsers(required=True)

    intent = sub.add_parser("intent", help="Classify a user query with the lightweight rule-based agent")
    intent.add_argument("question")
    intent.add_argument("--has-image", action="store_true")
    intent.set_defaults(func=cmd_intent)

    eval_intent = sub.add_parser("eval-intent", help="Evaluate the rule-based intent baseline")
    eval_intent.add_argument("--dataset", default="data/metadata/gpt_intent.json")
    eval_intent.set_defaults(func=cmd_eval_intent)

    check_data = sub.add_parser("check-data", help="Inspect metadata and image availability")
    check_data.add_argument("--metadata", default="data/metadata/image_dataset.json")
    check_data.set_defaults(func=cmd_check_data)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
