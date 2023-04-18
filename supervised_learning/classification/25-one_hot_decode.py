#!/usr/bin/env python3
"""module one hot decode"""

import numpy as np


def one_hot_decode(one_hot):
    """converts a one-hote matrix into a vector of labels"""
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    r = np.argmax(one_hot, axis=0)

    if r is int:
        return None

    return r
