#!/usr/bin/env python3
""" Variational Autoencoder

Requires:
    - tensorflow
"""
import tensorflow.keras as keras


def sampling(args):
    """ Generates a latent variable from the learned space distribution
        - args: tuple, contains the two arguments, z_mean and z_log_var
            - z_mean: mean of the latent space representation
            - z_log_var: log variance of the latent space distribution
    """
    z_mean, z_log_var = args
    epsilon = keras.backend.random_normal(shape=keras.backend.shape(z_mean))

    return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ Builds a Variational Autoencoder
        - input_dims: integer containing the dimensions of the model input
        - filters: list containing the number of filters for each
                   convolutional layer in the encoder
        - latent_dims: tuple of integers containing the dimensions of the
                       latent space representation
    """
    # Encoder input
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    # Subsequent layers
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation="relu")(x)

    # Latent space
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims,))([z_mean, z_log_var])

    # Encoder model
    encoder = keras.Model(inputs, [z, z_mean, z_log_var], name="encoder")

    # Decoder input
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    # Subsequent reversed layers
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation="relu")(x)

    outputs = keras.layers.Dense(input_dims, activation="sigmoid")(x)

    # Decoder model
    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    # Full autoencoder
    autoencoder_outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs, autoencoder_outputs, name="autoencoder")

    # Compile autoencoder
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
