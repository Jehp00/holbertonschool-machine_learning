#!/usr/bin/env python3
"""Module gettin cozy"""
import numpy as np


def cat_matrices2D(mat1, mat2, axis=0):
    """
    :param mat1: matrix of numbers
    :param mat2: matrix of numbers
    :param axis: axis for the matrices
    :return: new matrix with the previous ones
    """
    con = np.concatenate((mat1, mat2), axis)
    return con
