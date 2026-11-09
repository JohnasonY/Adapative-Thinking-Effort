"""Report held-out overall and per-criterion MAE/MSE."""

import argparse
import json

from .dataset import load_rows
from .scoring import CRITERIA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/test.csv")
    parser.add_argument("--checkpoint", default="models/difficulty_model.pt")
    args = parser.parse_args()
    try:
        rows = load_rows(args.data)
        from .predict import DifficultyPredictor
        predictor = DifficultyPredictor(args.checkpoint)
        errors = [[predicted - target for predicted, target in zip(predictor.predict(prompt), labels)]
                  for prompt, labels in rows]
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.error(str(error))
    per_criterion = {
        name: {"mae": sum(abs(row[index]) for row in errors) / len(rows),
               "mse": sum(row[index] ** 2 for row in errors) / len(rows)}
        for index, name in enumerate(CRITERIA)
    }
    print(json.dumps({"samples": len(rows),
                      "mae": sum(item["mae"] for item in per_criterion.values()) / 5,
                      "mse": sum(item["mse"] for item in per_criterion.values()) / 5,
                      "per_criterion": per_criterion}, indent=2))


if __name__ == "__main__":
    main()
