#!/usr/bin/env python3
""" Function to convert a gensim model to keras function"""

import tensorflow as tf


def gensim_to_keras(model):
    """ Convers a gensim model to keras function and returns it
        - model: trained gensim word2vec models
    """
    keyed_vectors = model.wv  # structure holding the result of training
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array

    layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )

    return layer
