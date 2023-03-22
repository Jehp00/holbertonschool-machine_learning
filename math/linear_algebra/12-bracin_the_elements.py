#!/usr/bin/env python3
"""Module bracin the elements"""


def np_elementwise(mat1, mat2):
    """
    :param mat1: matrix
    :param mat2: matrix
    :return: element-wise addition, subtraction, multiplication, and division
    """
    res = []
    res.append(mat1 + mat2)
    res.append(mat1 - mat2)
    res.append(mat1 * mat2)
    res.append(mat1 / mat2)
    return tuple(res)
