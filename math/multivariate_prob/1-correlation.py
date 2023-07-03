#!/usr/bin/env python3
"""This module contains the function mean_cov"""
import numpy as np


def correlation(C):
    """Calculates a correlationed matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if len(C.shape) < 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    cor = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    return cor
