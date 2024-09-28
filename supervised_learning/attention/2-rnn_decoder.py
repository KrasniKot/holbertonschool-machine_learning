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
        GRU       = tf.keras.layers.GRU
        Dense     = tf.keras.layers.Dense

        self.embedding = Embedding(input_dim=vocab, output_dim=embedding)
        self.gru = GRU(units=units, return_sequences=True, return_state=True,
                       recurrent_initializer="glorot_uniform")
        self.F = Dense(units=vocab)
        self.sattention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """ Computes the output for the decoder
            - x: tensor of shape (batch, 1) containing the previous word
                 in the target sequence as an index of the target vocabulary
            - s_prev: tensor of shape (batch, units) containing the previous
                      decoder hidden state
        """
        # Apply the attention mechanism
        context, _ = self.sattention(s_prev, hidden_states)

        # The previous target word x is passed through an embedding layer
        # transforming it into a dense vector representation
        x = self.embedding(x)

        # Since the context vector has a shape (batch, units)
        # we expand the dimension to make it (batch, 1, units)
        context = tf.expand_dims(context, axis=1)

        # The context vector (batch, 1, units)
        # and the embedding vector x (with shape (batch, 1, embedding_dim))
        # are concatenated along the last dimension (feature dimension).
        # The resulting tensor has the shape
        # (batch, 1, feature dimension + embedding_dim).
        context_concat = tf.concat([context, x], axis=-1)

        # Retrieve the output and hidden state
        outputs, hidden_state = self.gru(context_concat)

        # The outputs tensor from the GRU has shape (batch, 1, units).
        # The reshaping removes the time dimension (which is 1 here)
        # so that the tensor becomes (batch, units).
        resh = tf.reshape
        noutputs = resh(outputs, shape=(outputs.shape[0], outputs.shape[2]))

        # The reshaped outputs are passed through a dense layer (likely self.F)
        # to produce the final output y, which represents the predicted word.
        y = self.F(noutputs)

        return y, hidden_state
