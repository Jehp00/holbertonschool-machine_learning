#!/usr/bin/env python3
"""This module contains the function predict"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """makes a prediction using a neural network"""
    prediction = network.predict(data, verbose=verbose)
    return prediction
