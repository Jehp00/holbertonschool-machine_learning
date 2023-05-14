#!/usr/bin/env python3
"""This module contains the function test_model"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """tests a neural network"""
    evaluation = network.evaluate(data, labels, verbose=verbose)
    return evaluation
