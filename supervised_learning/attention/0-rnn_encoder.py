#!/usr/bin/env python3
""" Encoder for machine translation """

import tensorflow as tf


Layer = tf.keras.layers.Layer

class RNNEncoder(Layer):
    """ Recursive Neural Network Encoder for machine translation """

    def __init__(self, vocab, embedding, units, batch):
        """ Initializes an RNN Encoder
            - vocab: integer representing the size of the input vocabulary
            - embedding: integer representing the dimensionality
                         of the embedding vector
            - units: integer representing the number of hidden units
                     in the RNN cell
            - batch: integer representing the batch size
        """
