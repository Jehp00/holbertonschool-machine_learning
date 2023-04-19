#!/usr/bin/env python3
"""module tensor flow"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """Generate a layer for nn"""
    with tf.name_scope("layer"):
        layer = tf.layer.dense(
            prev,
            activation,
            units=n,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                mode="FAN_AVG"))
    return layer
