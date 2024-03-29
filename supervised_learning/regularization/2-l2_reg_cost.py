#!/usr/bin/env python3
"""This module contains the function l2_reg_gradient_descent"""
import tensorflow as tf


def l2_reg_cost(cost):
    """calculates the cost of a neural network with L2 regularization"""
    return cost + tf.losses.get_regularization_losses()
