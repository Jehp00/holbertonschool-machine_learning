#!/usr/bin/env python3
"""Module sigma sum"""


def summation_i_squared(n):
    """n: limit number of the sum
       return: The sum over each number of thee sigma sum"""
    num = 0
    for i in range(n + 1):
        num += i**2
    return num

