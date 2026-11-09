"""Independent, validated difficulty scoring."""

import math

CRITERIA = ("reasoning", "knowledge", "steps", "ambiguity", "math")


def calculate_difficulty(scores, weights=None):
    """Weighted sum of five 1–5 scores; default range is 5–25."""
    scores = list(scores)
    weights = [1.0] * len(CRITERIA) if weights is None else list(weights)
    if len(scores) != 5 or len(weights) != 5:
        raise ValueError("Expected five scores and five weights in criteria order.")
    if any(not math.isfinite(x) or not 1 <= x <= 5 for x in scores):
        raise ValueError("Each score must be finite and between 1 and 5.")
    if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) == 0:
        raise ValueError("Weights must be finite, nonnegative, and not all zero.")
    return sum(score * weight for score, weight in zip(scores, weights))
