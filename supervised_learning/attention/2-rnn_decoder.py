#!/usr/bin/env python3
""" Decoder for machine translation """

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ Defines an RNN Decoder for Machine Translation """

    def __init__(self, vocab, embedding, units, batch):
        """ Initializes an RNN Decoder
            - vocab: integer representing the size of the output vocabulary
            - embedding: integer representing the dimensionality
                         of the embedding vector
            - units: integer representing the number of hidden
                     units in the RNN cell
            - batch: integer representing the batch size
        """
        super().__init__()

        Embedding = tf.keras.layers.Embedding
        GRU = tf.keras.layers.GRU
        Dense = tf.keras.layers.Dense

        self.embedding = Embedding(input_dim=vocab, output_dim=embedding)
        self.gru = GRU(units=units, return_sequences=True, return_state=True,
                       recurrent_initializer="glorot_uniform")
        self.F = Dense(units=vocab)
        self.sattention = SelfAttention(vocab)

    def call(self, x, s_prev, hidden_states):
        """ Computes the output for the decoder
            - x: tensor of shape (batch, 1) containing the previous word
                 in the target sequence as an index of the target vocabulary
            - s_prev: tensor of shape (batch, units) containing the previous
                      decoder hidden state
        """
        context, _ = self.attention(s_prev, hidden_states)

        # Pass the previous word index through the embedding layer
        x = self.embedding(x)

        # Concatenate the context vector with x
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        # Pass the concatenated vector through the GRU layer
        output, s = self.gru(x)

        # Remove the extra axis
        output = tf.squeeze(output, axis=1)

        # Pass the GRU output through the Dense layer to predict the next word
        y = self.F(output)

        return y, s