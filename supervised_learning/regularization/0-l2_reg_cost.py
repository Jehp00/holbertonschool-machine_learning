#!/usr/bin/env python3
"""
Defines a function that calculates the cost of a neural network
using L2 Regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization
    """
    weight_squared = 0
    for i in range(1, L + 1):
        level_weights = weights[f"W{i}"]
        weight_squared += np.linalg.norm(level_weights)
    l2_reg_cost = cost + ((lambtha / (2 * m)) * weight_squared)
    return l2_reg_cost
