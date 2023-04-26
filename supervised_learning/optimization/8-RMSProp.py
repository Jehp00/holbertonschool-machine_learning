#!/usr/bin/env python3
"""module  optimization"""
import tensorflow as tf

def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Creates the training operation for a neural network in TensorFlow
        using the RMSProp optimization algorithm"""
    op = tf.train.RMSPropOptimizer(
        alpha, decay=beta2, epsilon=epsilon).minimize(loss)
    return op
