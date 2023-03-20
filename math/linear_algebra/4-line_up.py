#!/usr/bin/env python3
"""Module line-up-us"""


def add_arrays(arr1, arr2):
    """
    :param arr1: list of int
    :param arr2: list of ints
    :return: the addition between arr1 plus arr2 if they have the same shape
    """
    if len(arr1) != len(arr2):
        return None
    return [a1 + a2 for a1, a2 in zip(arr1, arr2)]
