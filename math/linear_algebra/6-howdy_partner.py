#!/usr/bin/env python3
"""Module howdy partner"""


def cat_arrays(arr1, arr2):
    """
    :param arr1: array of ints
    :param arr2: array of ints
    :return: the concatenation from arr1 and arr2
    """
    arr = [*arr1, *arr2]
    return arr
