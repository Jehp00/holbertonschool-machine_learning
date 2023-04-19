#!/usr/bin/env python3
"""module tensor flow"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """return 2 placeholders, x and y"""
    x = tf.placeholder("float", [None, nx], name='x')
    y = tf.placeholder("float", [None, classes], name='y')

    return x, y
