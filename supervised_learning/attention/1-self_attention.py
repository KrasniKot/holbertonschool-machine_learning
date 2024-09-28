#!/usr/bin/env python3
""" Class to calculate the self attention based on the paper:
    - https://arxiv.org/pdf/1409.0473
"""

import tensorflow as tf

layers = tf.keras.layers


class SelfAttention(layers.Layer):
    """ Defines a Self Attention class """

    def __init__(self, units):
        """ Initializes a SelfAttention
            - units: integer, number hidden units in alignment model
        """
        super().__init__()

        # ########## This layers will be used to calculate the alignment scores

        # Determines how important is the previous
        # encoder hidden state to the decoder hidden states
        self.W = tf.keras.layers.Dense(units=units)

        # Determines how important are the encoder hidden states
        # in relation to the previous decoder hidden state
        self.U = tf.keras.layers.Dense(units=units)

        # Determine the importance (or "attention")
        # of the encoder hidden states relative to the current decoder state
        self.V = tf.keras.layers.Dense(units=1)
        # ##########

    def call(self, s_prev, hidden_states):
        """ Calculates the context vector and the attention weights
            - s_prev: tensor of shape (batch, units)
                      containing the previous decoder hidden state
            - hidden_states: tensor of shape (batch, input_seq_len, units)
                             containing the outputs of the encoder
        """
        prev_dec_hstate = tf.expand_dims(s_prev, axis=1)

        # Expand s_prev to have the same time dimension as hidden_states
        # sprev: (batch, units) -> (batch, 1, units)
        prev_dec_hstate = tf.expand_dims(s_prev, axis=1)

        # Learnt attention weight matrices
        W = self.W(prev_dec_hstate)
        U = self.U(hidden_states)

        # Compute alignment scores: e_ij = v^T(tanh(W + U))
        alignment_scores = self.V(tf.nn.tanh(W + U))

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(alignment_scores, axis=1)

        # Expand the dimensions of the attention_weights so that it has the
        # same shape as the encoder hidden states tensor
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=-1)

        # Compute the context vector as the weighted sum of hidden states
        context_vector = tf.reduce_sum(attention_weights_expanded, axis=1)

        return context_vector, attention_weights
