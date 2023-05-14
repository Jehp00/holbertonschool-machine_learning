#!/usr/bin/env python3
"""This module contains the function build_model"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library"""
    L2 = K.regularizers.l2(lambtha)
    inputs = K.Input(shape=(nx,))
    outputs = K.layers.Dense(layers[0],
                             activation=activations[0],
                             kernel_regularizer=L2,
                             name='dense')(inputs)

    for i in range(1, len(layers)):
        dropout = K.layers.Dropout(1 - keep_prob)(outputs)
        outputs = K.layers.Dense(layers[i],
                                 activation=activations[i],
                                 kernel_regularizer=L2,
                                 name='dense_' + str(i))(dropout)

    return K.Model(inputs=inputs, outputs=outputs)
