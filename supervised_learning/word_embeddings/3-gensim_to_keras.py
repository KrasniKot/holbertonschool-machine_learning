#!/usr/bin/env python3
""" Function to convert a gensim model to keras embedding layer """

from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """ Convers a gensim model to keras embedding layer and returns it
        - model: trained gensim word2vec models
    """
    keyed_vectors = model.wv  # structure holding the result of training
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array

    # which row in `weights` corresponds to which word?
    index_to_key = keyed_vectors.index_to_key
    layer = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=False,
    )
    return layer