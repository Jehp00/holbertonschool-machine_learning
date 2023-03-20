#!/usr/bin/env python3
"""Module across the planes task 5"""


def add_matrices2D(mat1, mat2):
    """
    :param mat1: matrix to add
    :param mat2: matrix to recive
    :return: the addition of the matrices
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    return [[a1 + a2 for a1, a2 in zip(arr1, arr2)]
            for arr1, arr2 in zip(mat1, mat2)]
