#!/usr/bin/env python3
"""Module sigma sum"""


def summation_i_squared(n):
    """n: limit number of the sum
       return: The sum over each number of thee sigma sum"""
    if not isinstance(n, int) and not isinstance(n, float):
        return None
    return int((n * (n + 1) * (2 * n + 1)) / 6)

