#!/usr/bin/env python3
""" Class that performs multihead attention """

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """ Defines a MultiHead Attention mechanism """

    def __init__(self, dm, h):
        """ Initializes a MultiHead Attention
            - dm ... integer representing the dimensionality of the model
            - h .... integer representing the number of heads
        """
        super().__init__()
        self.h      = h                                         # Number of heads
        self.dm     = dm                                        # Model Dimensionality
        self.depth  = dm // h                                   # The Depth of the attention head

        self.Wq     = tf.keras.layers.Dense(units=self.dm)      # Query weights
        self.Wk     = tf.keras.layers.Dense(units=self.dm)      # Key weights
        self.Wv     = tf.keras.layers.Dense(units=self.dm)      # Value weights
        self.linear = tf.keras.layers.Dense(units=self.dm)      # Linear layer to generate the attention output

    def call(self, Q, K, V, mask):
        """ Returns the attention mechanism output and the attention weights
            - Q ...... tensor of shape (batch, seq_len_q, dk) containing the input to generate the query matrix
            - K ...... tensor of shape (batch, seq_len_v, dk) containing the input to generate the key matrix
            - V ..-... tensor of shape (batch, seq_len_v, dv) containing the input to generate the value matrix
            - mask ... always None
        """
        batch_size = tf.shape(Q)[0]

        # Extract Queries, Keys and Values
        QKVs = [self.Wq(Q), self.Wk(K), self.Wv(V)]

        #### Q, K, and V are reshaped into 4D tensors
        # The result is a tensor of shape:
        # (batch size, number of heads, seq_len, depth),
        # where attention can be computed independently across each head.
        for i in range(len(QKVs)):
            QKV = tf.reshape(QKVs[i], (batch_size, -1, self.h, self.depth))
            # The sequence length and head dimensions are swapped to facilitate
            # parallel computation across attention heads
            # perm=[0, 2, 1, 3] swaps the second and third dimensions
            QKVs[i] = tf.transpose(QKV, perm=[0, 2, 1, 3])
        ####

        Q, K, V = QKVs

        # Calculate the attention outputs and weights
        scaled_att, weights_att = sdp_attention(Q, K, V, mask)

        #### After computing attention, the outputs are transposed back
        # to the original shape by swapping the sequence
        # length and head dimensions again.
        # Then, the results are reshaped into a 3D tensor
        # (batch_size, seq_len, dm)
        # where dm is the model dimension (dm = h * depth).
        # This means all the attention heads are concatenated back together.
        scaled_att = tf.transpose(scaled_att, perm=[0, 2, 1, 3])
        concat_att = tf.reshape(scaled_att, (batch_size, -1, self.dm))
        ####

        # The concatenated attention output is passed through a
        # final linear layer (self.linear) to transform it into
        # the desired output dimension.
        Y = self.linear(concat_att)

        return Y, weights_att
