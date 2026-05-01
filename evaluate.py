"""
Evaluation harness.

Runs the pipeline over a dataset split and computes accuracy, selective
accuracy, coverage, verification rate, average iterations, and failure modes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from env_loader import get_openai_api_key, load_project_env

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("eval")


def load_dataset(split_path: str, image_root: str) -> list[dict]:
    with open(split_path, encoding="utf-8") as handle:
        items = json.load(handle)
    root = Path(image_root)
    for item in items:
        item["image_path"] = str(root / item["image_path"])
    return items


def run_evaluation(args) -> dict:
    from langchain_openai import ChatOpenAI
    from pipeline import run_pipeline

    api_key = get_openai_api_key(args.openai_key)
    if not api_key:
        logger.error("OpenAI API key required. Add OPENAI_API_KEY to D:\\files\\.env or your shell env.")
        sys.exit(1)
    llm = ChatOpenAI(
        model=args.planner_model,
        api_key=api_key,
        temperature=0,
        max_retries=0,
        timeout=60,
    )
    exec_cfg = {
        "backend": args.backend,
        "openai_key": api_key,
        "model": args.vision_model,
    }

    items = load_dataset(args.split, args.image_root)
    logger.info(f"Evaluating {len(items)} items from {args.dataset}")

    results = []
    correct = 0
    abstained = 0
    total_iters = 0
    failure_modes: dict[str, int] = {}

    for idx, item in enumerate(items):
        try:
            graph = run_pipeline(
                image_path=item["image_path"],
                question=item["question"],
                llm=llm,
                executor_config=exec_cfg,
                critic_config={"allow_mock_models": False},
                max_iterations=args.max_iterations,
                strict_models=True,
            )
        except Exception as exc:
            logger.error(f"[{idx}] Pipeline error: {exc}")
            results.append({"idx": idx, "error": str(exc)})
            continue

        gt = str(item["answer"]).strip().lower()
        pred = graph.answer_str.strip().lower()

        is_abstain = pred == "abstain"
        is_correct = (not is_abstain) and (pred == gt)

        if is_abstain:
            abstained += 1
            failure_modes[graph.failure_mode] = failure_modes.get(graph.failure_mode, 0) + 1
        elif is_correct:
            correct += 1

        total_iters += graph.iterations

        results.append(
            {
                "idx": idx,
                "question": item["question"],
                "gt": gt,
                "pred": pred,
                "correct": is_correct,
                "abstain": is_abstain,
                "verified": graph.verified,
                "iterations": graph.iterations,
                "failure_mode": graph.failure_mode,
            }
        )

        if (idx + 1) % 10 == 0:
            answered = len(results) - abstained
            selective_acc = correct / max(answered, 1)
            logger.info(
                f"[{idx + 1}/{len(items)}] selective_acc={selective_acc:.3f} "
                f"abstain={abstained}"
            )

    n = len(results)
    answered = n - abstained
    accuracy = correct / max(n, 1)
    selective_acc = correct / max(answered, 1)
    coverage = answered / max(n, 1)
    avg_iters = total_iters / max(n, 1)
    verification_rate = sum(1 for result in results if result.get("verified", False)) / max(n, 1)

    summary = {
        "dataset": args.dataset,
        "n": n,
        "accuracy": round(accuracy, 4),
        "selective_accuracy": round(selective_acc, 4),
        "coverage": round(coverage, 4),
        "abstain_rate": round(abstained / max(n, 1), 4),
        "verification_rate": round(verification_rate, 4),
        "avg_iterations": round(avg_iters, 4),
        "failure_modes": failure_modes,
    }

    logger.info("=== SUMMARY ===")
    for key, value in summary.items():
        logger.info(f"  {key}: {value}")

    output = {"summary": summary, "results": results}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
        logger.info(f"Results saved -> {args.output}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Spatial Agent Evaluation Harness")
    parser.add_argument("--dataset", default="whatsup", choices=["whatsup", "gqa", "3dsrbench"])
    parser.add_argument("--split", required=True, help="Path to the JSON annotation file")
    parser.add_argument("--image_root", required=True, help="Root directory for images")
    parser.add_argument("--backend", default="local", choices=["local", "openai"])
    parser.add_argument("--openai_key", default="", help="OpenAI API key")
    parser.add_argument("--planner_model", default="gpt-4o-mini")
    parser.add_argument("--vision_model", default="gpt-4o")
    parser.add_argument("--max_iterations", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--output", default="", help="Path to save JSON results")
    args = parser.parse_args()
    load_project_env()
    run_evaluation(args)


if __name__ == "__main__":
    main()
