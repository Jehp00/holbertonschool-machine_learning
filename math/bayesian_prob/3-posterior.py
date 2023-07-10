#!/usr/bin/env python3
"""This module contains the function posterior"""

intersection = __import__('1-intersection').intersection
marginal = __import__('2-marginal').marginal


def posterior(x, n, P, Pr):
    """calculates the posterior probability for the various hypothetical
       probabilities of developing severe side effects given the data."""
    interse = intersection(x, n, P, Pr)
    margi = marginal(x, n, P, Pr)

    return interse / margi
