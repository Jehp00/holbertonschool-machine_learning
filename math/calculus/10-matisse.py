#!/usr/bin/env python3
"""Module poly derivative"""


def poly_derivative(poly):
    """
    poly: list of coefficients representing a polynomial
    return: new list of coefficients
    """
    if not isinstance(poly, list):
        return None;
    if len(poly) == 1:
        return [0]
    return [(index + 1) * elem for index, elem in enumerate(poly[1:])]
