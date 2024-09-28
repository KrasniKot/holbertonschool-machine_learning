#!/usr/bin/env python3
"""
    Module to create Class SelfAttention
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
        class to calculate attention ofr machine translation
    """
    def __init__(self, units):
        """
            class constructor
        :param units: integer, number hidden units in alignment model
        """
        if not isinstance(units, int):
            raise TypeError("units should be an integer")

        super().__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """ Calculates the context vector and the attention weights
            - s_prev: tensor of shape (batch, units)
                      containing the previous decoder hidden state
            - hidden_states: tensor of shape (batch, input_seq_len, units)
                             containing the outputs of the encoder
        """
        # Expand s_prev to have the same time dimension as hidden_states
        # sprev: (batch, units) -> (batch, 1, units)
        s_prev = tf.expand_dims(s_prev, axis=1)

        # Learnt attention weight matrices
        W = self.W(s_prev)
        U = self.U(hidden_states)

        # Compute alignment scores: e_ij = v^T(tanh(W + U))
        alignment_scores = self.V(tf.nn.tanh(W + U))

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(alignment_scores, axis=1)

        # Compute the context vector as the weighted sum of hidden states
        c = attention_weights * hidden_states
        context = tf.reduce_sum(c, axis=1)

        return context, attention_weights
