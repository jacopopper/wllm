from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def approximate_entropy_from_top_logprobs(logprobs: Sequence[float] | np.ndarray) -> float:
    values = np.asarray(logprobs, dtype=np.float64)
    if values.size == 0:
        return 0.0
    probabilities = np.exp(values)
    mass = float(np.sum(probabilities))
    if mass <= 0:
        return 0.0
    normalized = probabilities / mass
    positive = normalized[normalized > 0]
    return float(-np.sum(positive * np.log(positive)))
