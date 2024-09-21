#!/usr/bin/env python3
""" Function to convert a gensim model to keras function"""

import tensorflow


def gensim_to_keras(model):
    """ Convers a gensim model to keras function and returns it
        - model: trained gensim word2vec models
    """
    Embedding = tensorflow.keras.layers.Embedding
    return model.wv.get_keras_embedding(train_embeddings=False)
