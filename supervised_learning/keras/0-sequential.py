#!/usr/bin/env python3
"""This module contains the function build_model"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library"""
    L2 = K.regularizers.l2(lambtha)
    model = K.Sequential()
    model.add(K.layers..Dense(layers[0], input_shape=(nx,),
                              activation=activations[0],
                              kernel_regularizer=L2,
                              name='dense'))

    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layers[i],
                                 activation=activations[0],
                                 kernel_regularizer=L2,
                                 name='dense_' + str(i)))

    return model
