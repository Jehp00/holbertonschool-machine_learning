#!/usr/bin/env python3
"""Basic autoencoder vanilla encoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
       autoencoder for nanilla
       Args:
           input_dims: integer containing the dimensions of the model input
           hidden_layers: list containing the number of nodes for each hidden
                          layer in the encoder, respectively
                          hidden layers should be reversed for the decoder
           latent_dims: integer containing the dimensions of the latent space
                         representation
       Returns: encoder, decoder, auto
               encoder: the encoder model
               decoder: the decoder model
               auto: the full autoencoder model
       """
    X_inpt_encoded = keras.Input(shape=(input_dims, ))
    hidden_lay = keras.layers.Dense(units=hidden_layers[0], activation='relu')

    Y_prev = hidden_lay(X_inpt_encoded)
    for i in range(1, len(hidden_layers)):
        hidden_lay = keras.layers.Dense(units=hidden_layers[i],
                                        activation='relu')
        Y_prev = hidden_lay(Y_prev)

    latent_ly = keras.layers.Dense(units=latent_dims, activation='relu')
    bottleneck = latent_ly(Y_prev)
    encoder = keras.Model(X_inpt_encoded, bottleneck)

    X_input_decoded = keras.Input(shape=(latent_dims,))
    hidden_ly_decoded = keras.layers.Dense(units=hidden_layers[-1],
                                           activation='relu')

    Y_prev = hidden_ly_decoded(X_input_decoded)
    for i in range(len(hidden_layers) - 2, -1, -1):
        hidden_ly_decoded = keras.layers.Dense(units=hidden_layers[i],
                                               activation='relu')
        Y_prev = hidden_ly_decoded(Y_prev)

    last_layer = keras.layers.Dense(units=input_dims, activation='sigmoid')
    output = last_layer(Y_prev)
    decoder = keras.Model(X_input_decoded, output)

    X_input = keras.Input(shape=(input_dims,))
    encoder_o = encoder(X_input)
    decoder_o = decoder(encoder_o)
    auto = keras.Model(X_input, decoder_o)
    auto.compile(loss='binary_crossentropy', optimizer='adam')

    return encoder, decoder, auto
