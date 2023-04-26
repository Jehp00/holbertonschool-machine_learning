#!/usr/bin/env python3
"""Module optimization"""
import numpy as np


def normalize(X, m, s):
    """
    Normalize a matrix

    :param X: matrix [np.ndarray with shape (d, nx]
    :param m: mean of all features of X
    :param s: standard deviation of all features of X
    :return: X normalized
    """
    return ((X - m) / s)
