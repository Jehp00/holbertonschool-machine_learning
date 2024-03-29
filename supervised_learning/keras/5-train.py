#!/usr/bin/env python3
"""This module contains the function train_model"""
import tensorflow.keras as K


def train_model(
        network,
        data,
        labels,
        batch_size,
        epochs,
        validation_data=None,
        verbose=True,
        shuffle=False):
    """trains a model using mini-batch gradient descent"""
    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data)
    return history
