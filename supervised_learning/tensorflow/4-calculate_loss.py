#!/usr/bin/env python3
"""This module contains the function forward_prop"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """softmax cross-entropy loss"""
    return tf.losses.softmax_cross_entropy(y, y_pred)
