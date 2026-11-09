"""CLI entry point; manual scores support an offline demo."""

import argparse
import json

from .effort_selector import select_effort
from .scoring import CRITERIA, calculate_difficulty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--checkpoint", default="models/difficulty_model.pt")
    parser.add_argument("--scores", nargs=5, type=float, metavar="SCORE")
    parser.add_argument("--weights", nargs=5, type=float, metavar="WEIGHT")
    parser.add_argument("--low-threshold", type=float, default=10)
    parser.add_argument("--medium-threshold", type=float, default=17)
    args = parser.parse_args()
    try:
        if not args.prompt.strip():
            raise ValueError("Prompt must not be empty.")
        if args.scores is None:
            from .predict import DifficultyPredictor
            scores = DifficultyPredictor(args.checkpoint).predict(args.prompt)
        else:
            scores = args.scores
        difficulty = calculate_difficulty(scores, args.weights)
        effort = select_effort(difficulty, args.low_threshold, args.medium_threshold)
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps({"prompt": args.prompt, "source": "manual" if args.scores is not None else "model",
                      "scores": dict(zip(CRITERIA, scores)), "difficulty": difficulty,
                      "effort": effort}, indent=2))


if __name__ == "__main__":
    main()
