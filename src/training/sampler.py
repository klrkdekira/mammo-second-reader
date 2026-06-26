"""Optional balanced sampler for the training loader.

pos_weight and WeightedRandomSampler are complementary, not redundant.
Measure val AUC with each before combining them.
"""

from collections.abc import Sequence

import pandas as pd
from torch.utils.data import WeightedRandomSampler


def balanced_sampler(labels: Sequence[int]) -> WeightedRandomSampler:
    counts = pd.Series(labels).value_counts().to_dict()
    weights = [1.0 / counts[y] for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
