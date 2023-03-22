#!/usr/bin/env python3
"""Module cats got your tongue"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    :param mat1: matrix
    :param mat2: matrix
    :param axis: axis of the concatenation
    :return: the matrix concatenated
    """
    return np.concatenate((mat1, mat2), axis=axis)
