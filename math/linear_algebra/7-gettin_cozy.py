#!/usr/bin/env python3
"""Module gettin cozy"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    :param mat1: matrix of numbers
    :param mat2: matrix of numbers
    :param axis: axis for the matrices
    :return: new matrix with the previous ones
    """
    if axis == 0:
        for row in mat1:
            mat1_columns = len(row)
        for row in mat2:
            mat2_columns = len(row)
        if mat1_columns != mat2_columns:
            return None

        cat_matrix = []
        for index1, row in enumerate(mat1):
            cat_matrix.append([])
            for i in row:
                cat_matrix[index1].append(i)
        index1 += 1
        for index2, row in enumerate(mat2):
            cat_matrix.append([])
            for i in row:
                cat_matrix[index1 + index2].append(i)
        return cat_matrix

    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        cat_matrix = []
        for index, row in enumerate(mat1):
            cat_matrix.append([])
            for i in mat1[index]:
                cat_matrix[index].append(i)
            for i in mat2[index]:
                cat_matrix[index].append(i)
        return cat_matrix
    return None
