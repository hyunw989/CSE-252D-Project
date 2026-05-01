"""
Spatial Evidence Agent CLI entry point.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from env_loader import get_openai_api_key, load_project_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def get_llm(args):
    """Build the LangChain LLM for the Planner."""
    from langchain_openai import ChatOpenAI

    api_key = get_openai_api_key(args.openai_key)
    if not api_key:
        logger.error("OpenAI API key required. Add OPENAI_API_KEY to D:\\files\\.env or your shell env.")
        sys.exit(1)

    return ChatOpenAI(
        model=args.planner_model,
        api_key=api_key,
        temperature=0,
        max_retries=0,
        timeout=60,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Spatial Evidence Agent - verified spatial VQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--image", required=True, help="Path to input RGB image")
    parser.add_argument("--question", required=True, help="Binary spatial question")
    parser.add_argument(
        "--backend",
        default="local",
        choices=["local", "openai"],
        help="Executor backend: 'local' (LLaVA on GPU) or 'openai' (vision API)",
    )
    parser.add_argument("--openai_key", default="", help="OpenAI API key")
    parser.add_argument(
        "--planner_model",
        default="gpt-4o-mini",
        help="LLM model for the Planner",
    )
    parser.add_argument(
        "--vision_model",
        default="gpt-4o",
        help="Vision model for the OpenAI-compatible Executor",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Max executor/critic attempts (ablation: 1, 2, or 3)",
    )
    parser.add_argument("--save_annotation", default="", help="Save annotated image to this path")
    parser.add_argument("--save_graph", default="", help="Save evidence graph JSON to this path")
    parser.add_argument("--verbose", action="store_true", help="Verbose evidence output")

    args = parser.parse_args()
    loaded_env_files = load_project_env()

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)

    logger.info(f"Image   : {image_path}")
    logger.info(f"Question: {args.question}")
    logger.info(f"Backend : {args.backend} | k={args.k}")
    if loaded_env_files:
        logger.info("Loaded env file(s): " + ", ".join(str(path) for path in loaded_env_files))

    llm = get_llm(args)
    openai_key = get_openai_api_key(args.openai_key)

    exec_cfg = {
        "backend": args.backend,
        "openai_key": openai_key,
        "model": args.vision_model,
    }

    from pipeline import run_pipeline

    try:
        graph = run_pipeline(
            image_path=str(image_path),
            question=args.question,
            llm=llm,
            executor_config=exec_cfg,
            critic_config={"allow_mock_models": False},
            max_iterations=args.k,
            strict_models=True,
        )
    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}")
        sys.exit(1)

    from visualize import annotate_image, export_graph_json, print_graph

    print_graph(graph, verbose=args.verbose)

    if args.save_annotation:
        annotate_image(str(image_path), graph, out_path=args.save_annotation)
    if args.save_graph:
        export_graph_json(graph, args.save_graph)

    sys.exit(0)


if __name__ == "__main__":
    main()
