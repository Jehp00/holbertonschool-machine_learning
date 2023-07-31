#!/usr/bin/env python3
"""Convolutional autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Convolutional autoencoder
    Args:
        input_dims: tuple of integers containing the dimensions of
                    the model input
        filters: list containing the number of filters for each convolutional
                 layer in the encoder, respectively
        latent_dims: tuple of integers containing the dimensions of the latent
                      space representation
    Returns: encoder, decoder, auto
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """
    x_inpt = keras.Input(shape=(input_dims,))

    encoded_conv = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                                       padding='same',
                                       activation='relu')(x_inpt)

    pool_encoded = keras.layers.MaxPoolind2D(pool_size=(2, 2),
                                             padding="same")(encoded_conv)

    for i in range(1, len(filters)):
        encoded_conv = keras.layers.Conv2D(filters=filters[i],
                                           kernel_size=(3, 3), padding='same',
                                           activation='relu')(pool_encoded)
        pool_encoded = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                 padding="same")(encoded_conv)

    latent_ly = pool_encoded
    encoder = keras.Model(x_inpt, latent_ly)

    x_decode = keras.Input(shape=latent_dims)
    decoded_conv = keras.layers.Conv2D(filters=filters[-1],
                                       kernel_size=(3, 3),
                                       padding='same',
                                       activation='relu')(x_decode)

    pool_decoded = keras.layers.UpSampling2D((2, 2))(decoded_conv)

    for i in range(len(filters) - 2, 0, -1):
        decoded_conv = keras.layers.Conv2D(filters=filters[i],
                                           kernel_size=(3, 3),
                                           padding='same',
                                           activation='relu')(pool_decoded)
        pool_decoded = keras.layers.UpSampling2D((2, 2))(decoded_conv)

    decoded_conv = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                                       padding='valid',
                                       activation='relu')(pool_decoded)

    pool_decoded = keras.layers.UpSampling2D((2, 2))(decoded_conv)

    output = keras.layers.Conv2D(filters=input_dims[-1], kernel_size=(3, 3),
                                 padding='same',
                                 activation='sigmoid')(pool_decoded)

    decoder = keras.Model(x_decode, output)

    x_inpt = keras.Input(shape=input_dims)
    e_opt = encoder(x_inpt)
    d_opt = encoder(e_opt)
    auto = keras.Model(inputs=x_inpt, outputs=d_opt)
    auto.compile(loss='binary_crossentropy', optimizer='adam')
    return encoder, decoder, auto
