#!/usr/bin/env python3
"""
    module flip the list over
"""


def matrix_transpose(matrix):
    """
    :param matrix: the matrix to filp the elements from its lists
    :return: the new matrix fliped over
    """
    new_row = []
    for i in range(len(matrix[0])):
        row = []
        for item in matrix:
            row.append(item[i])
        new_row.append(row)
    return new_row
