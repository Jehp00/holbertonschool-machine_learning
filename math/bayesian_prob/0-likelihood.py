#!/usr/bin/env python3
"""This module contains the function likelihood"""
import numpy as np
from scipy.special import comb


def likelihood(x, n, P):
    """calculates the likelihood of obtaining this data given
       various hypothetical probabilities of developing severe side effects."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')

    if not isinstance(x, int) or x <= 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')

    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError('P must be a 1D numpy.ndarray')

    if not np.all(np.logical_and(P >= 0, P <= 1)):
        raise ValueError('All values in P must be in the range [0, 1]')

    C = comb(n, x)

    return C * (P ** x) * ((1 - P) ** (n - x))
