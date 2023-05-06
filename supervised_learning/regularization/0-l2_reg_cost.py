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
    weights_l2 = sum([np.linalg.norm(v) for k, v in weights.items()
                      if k[0] == 'W'])
    cost_l2 = cost + ((lambtha / (2 * m)) * weights_l2)

    return cost_l2
