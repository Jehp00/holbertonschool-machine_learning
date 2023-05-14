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
        early_stopping=False,
        patience=0,
        verbose=True,
        shuffle=False):
    """trains a model using mini-batch gradient descent"""
    callbacks = []
    if early_stopping and validation_data:
        es = K.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', patience=patience)
        callbacks.append(es)

    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          callbacks=callbacks,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data)
    return history
