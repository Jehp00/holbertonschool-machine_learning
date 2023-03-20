#!/usr/bin/env python3
"""
Module size me please
"""
def matrix_shape(matrix):
    """
    :param matrix: calculate the shape of the matrix
    :return: the lenght of the lists that are lists
    """
    new_l = []
    block = matrix
    while True:
        new_l.append(len(block))
        block = block[0]
        if not isinstance(block, list):
            break;
    return new_l
