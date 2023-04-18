#!/usr/bin/env python3
"""module one hot encode"""

import numpy as np


def one_hot_encode(Y, classes):
    """Become a numerical label vector into a one-hot matrix"""
    if not isinstance(Y, np.ndarray) or not \
            isinstance(classes, int) or max(Y) > classes:
        return None

    b = np.zeros((Y.size, classes))
    b[np.arange(Y.size), Y] = 1
    return b.T
