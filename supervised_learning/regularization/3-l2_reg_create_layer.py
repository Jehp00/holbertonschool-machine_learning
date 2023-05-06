#!/usr/bin/env python3
"""This module contains the function l2_reg_gradient_descent"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """creates a tensorflow layer that includes L2 regularization"""
    init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")

    regularizer = tf.contrib.layers.l2_regularizer(lambtha)

    layer = tf.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=regularizer)

    return layer(prev)
