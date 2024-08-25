#!/usr/bin/env python3
""" Convolutional autoencoder

Requires:
    - tensorflow
"""

import tensorflow.keras as keras

layers, models = keras.layers, keras.models


def autoencoder(input_dims, filters, latent_dims):
    """ Builds a convolutional autoencoder
        - input_dims: integer containing the dimensions of the model input
        - filters: list containing the number of filters for each
                   convolutional layer in the encoder
        - latent_dims: tuple of integers containing the dimensions of the
                       latent space representation
    """
    # Encoder input
    encoder_input = keras.Input(shape=(input_dims))
    x = encoder_input

    # Subsequent layers
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation="relu",
                                padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)

    # Encoder model
    encoder = keras.Model(encoder_input, x)

    # Decoder inputs and latent space
    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input

    # Subsequent reversed layers
    for f in range(len(list(reversed(filters)))):
        if (f == len(filters) - 1):
            x = keras.layers.Conv2D(filters[f], (3, 3), activation="relu",
                                    padding="valid")(x)
        else:
            # Second to last layer
            x = keras.layers.Conv2D(filters[f], (3, 3), activation="relu",
                                    padding="same")(x)

        x = keras.layers.UpSampling2D((2, 2))(x)

    # Last later
    x = keras.layers.Conv2D(input_dims[-1], (3, 3), activation="sigmoid",
                            padding="same")(x)

    # Decoder model
    decoder = keras.Model(decoder_input, x)

    # Autoencoder model
    autoencoder = keras.Model(encoder_input, decoder(encoder(encoder_input)))
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, autoencoder
