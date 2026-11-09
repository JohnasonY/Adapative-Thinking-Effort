"""Map difficulty to configurable effort thresholds."""

import math


def select_effort(difficulty, low_threshold=10.0, medium_threshold=17.0):
    if not all(math.isfinite(x) for x in (difficulty, low_threshold, medium_threshold)):
        raise ValueError("Difficulty and thresholds must be finite.")
    if difficulty < 0 or not 0 <= low_threshold < medium_threshold:
        raise ValueError("Use nonnegative difficulty and ordered nonnegative thresholds.")
    if difficulty <= low_threshold:
        return "low"
    if difficulty <= medium_threshold:
        return "medium"
    return "high"
