#!/usr/bin/env python3
""" Function that creates a vanilla autoencoder

Requires:
    - tensorflow
"""
import tensorflow.keras as keras

layers = keras.layers
models = keras.models


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ Creates a vanilla autoencoder
        - input_dims: integer, dimensions of the model input
        - hidden_layers: list, nodes number for each hidden layer
        - latent_dims: integer, dimensions of the latent space representation
    """
    # Encoder input
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    # Subsequent layers
    for units in hidden_layers:
        x = layers.Dense(units, activation='relu')(x)

    # Latent space
    latent = layers.Dense(latent_dims, activation='relu')(x)

    # Encoder model
    encoder = models.Model(encoder_input, latent, name="encoder")

    # Decoder input
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    # Subsequent decoder layers
    for units in reversed(hidden_layers):
        x = layers.Dense(units, activation='relu')(x)
    decoder_output = layers.Dense(input_dims, activation='sigmoid')(x)

    # Decoder model
    decoder = models.Model(decoder_input, decoder_output, name="decoder")

    # Autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = models.Model(auto_input, decoded, name="autoencoder")

    # Autoencoder compilation
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
