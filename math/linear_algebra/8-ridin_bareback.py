#!/usr/bin/env python3
"""Module ridin bareback"""


def mat_mul(mat1, mat2):
    """
    :param mat1: matrix 3.2
    :param mat2: matrix 2.4
    :return: the miltiplaction btween mat1 and mat2
    """
    A = len(mat1[0])
    B = len(mat2)

    if A != B:
        return None

    new_mat = []
    for row_idx, row in enumerate(mat1):
        new_mat.append([])
        for column_idx in range(len(mat2[0])):
            dot = 0
            for index in range(A):
                dot += (mat1[row_idx][index] * mat2[index][column_idx])
            new_mat[row_idx].append(dot)
    return new_mat
