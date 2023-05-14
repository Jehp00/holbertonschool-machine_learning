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
        learning_rate_decay=False,
        alpha=0.1,
        decay_rate=1,
        save_best=False,
        filepath=None,
        verbose=True,
        shuffle=False):
    """trains a model using mini-batch gradient descent"""
    callbacks = []
    if early_stopping and validation_data:
        es = K.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', patience=patience)
        callbacks.append(es)

    def learning_rate_fn(epoch):
        """The function that sets the learning rate for each epoch"""
        return alpha / (1 + decay_rate * epoch)

    if learning_rate_decay and validation_data:
        lrd = K.callbacks.LearningRateScheduler(learning_rate_fn, verbose=1)
        callbacks.append(lrd)

    if save_best:
        best = K.callbacks.ModelCheckpoint(filepath=filepath,
                                           save_best_only=True,
                                           mode='min')
        callbacks.append(best)

    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          callbacks=callbacks,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data)
    return history
