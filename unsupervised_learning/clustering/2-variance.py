#!/usr/bin/env python3
"""Variance"""

import numpy as np


def variance(X, C):
    """
    Calculate the total intra-cluste variance for a data set
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    N, D = X.shape

    ctroid_extended = C[:, np.newaxis]
    distance = np.sqrt(((X - ctroid_extended) ** 2).sum(axis=2))
    min_distance = np.min(distance, axis=0)

    var = np.sum(min_distance ** 2)

    return var
