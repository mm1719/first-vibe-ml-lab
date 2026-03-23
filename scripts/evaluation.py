"""Command-line entrypoint for evaluation."""

import argparse

from configs.config import MODEL_PATH
from src.utils.evaluation import evaluate


def main() -> None:
    """Parse CLI arguments and run evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Path to the model checkpoint used for evaluation.",
    )
    parser.add_argument(
        "--resume-run-id",
        type=str,
        default=None,
        help="When provided, evaluation logs will be appended to the existing W&B run id (resume).",
    )
    args = parser.parse_args()
    evaluate(model_path=args.model_path, resume_run_id=args.resume_run_id)


if __name__ == "__main__":
    main()
