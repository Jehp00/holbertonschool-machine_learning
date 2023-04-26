#!/usr/bin/env python3
"""module  optimization"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """creates the training operation for a neural network
    in tensorflow using the gradient descent with momentum
    optimization algorithm"""
    op = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
    return op
