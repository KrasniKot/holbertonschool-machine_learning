#!/usr/bin/env python3
""" Encoder for machine translation """

import tensorflow as tf

layers = tf.keras.layers
Embedding = layers.Embedding
GRU = layers.GRU


class RNNEncoder(layers.Layer):
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
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = Embedding(input_dim=vocab, output_dim=embedding)
        self.gru = GRU(units=self.units, return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """ Initializes the hidden states for the RNN cell to a tensor of zeros
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """ Passes the parameters to the gru function
            - x: tensor of shape (batch, input_seq_len) containing the input
                 to the encoder layer as word indices within the vocabulary
            - initial: tensor of shape (batch, units),
                       contains the initial hidden state
        """
        # Convert word indices into embeddings
        x = self.embedding(x)

        # Pass the embeddings and the initial hidden state to the GRU
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
