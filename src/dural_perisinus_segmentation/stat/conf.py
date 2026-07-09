import math
from typing import Sequence

import numpy as np
from scipy import stats


def compute_t_conf(samples: Sequence[float], conf: float = 0.95) -> tuple[float, float]:
    """
    Computes confidence interval with the Student's t distribution.

    Parameters
    ----------
    samples : Sequence[float]
        The samples.
    conf : float, default=0.95
        The confidence level.

    Returns
    -------
    tuple[float, float]
        The boundaries of the confidence interval.
    """
    mean = np.mean(samples)
    std = np.std(samples)
    n = len(samples)
    return stats.t.interval(conf, n - 1, loc=mean, scale=std / math.sqrt(n))
