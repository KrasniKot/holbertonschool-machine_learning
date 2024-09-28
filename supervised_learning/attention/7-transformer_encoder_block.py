#!/usr/bin/env python3
""" Encoder block for a transformer """

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """ Defines an EncoderBlock """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ Initializes an Encoder Block
            - dm:          dimensionality of the model
            - h:           the number of heads
            - hidden:      number of hidden units in the fully connected layer
            - drop_rate:   dropout rate
        """
        Dense = tf.keras.layers.Dense
        LN = tf.keras.layers.LayerNormalization
        Dout = tf.keras.layers.Dropout

        super().__init__()

        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = Dense(units=hidden, activation='relu')
        self.dense_output = Dense(units=dm)
        self.layernorm1 = LN(epsilon=1e-6)
        self.layernorm2 = LN(epsilon=1e-6)
        self.dropout1 = Dout(drop_rate)
        self.dropout2 = Dout(drop_rate)

    def call(self, x, training, mask=None):
        """ Returns the encoder's output
            - x:          tensor of shape (batch, input_seq_len, dm)
                          containing the input to the encoder block.
            - training:   boolean to determine if the model is training
            - mask:       the mask to be applied for multi head attention
        """
        # Apply multi-head attention and dropout to the input x
        mattentions, _ = self.mha(x, x, x, mask)
        mattentions_dropped = self.dropout1(mattentions, training=training)

        # The input x is added to the dropout-modified attention output
        # to apply a residual connection (also called skip connection).
        # This helps in preserving information from the original input x.
        xmattentions = self.layernorm1(x + mattentions_dropped)

        # The skip connection output is passed through a feed-forward network
        ffy = self.dense_hidden(xmattentions)

        # The output of the first dense layer is then passed through
        # another dense layer and a dropout layer, which transforms
        # it back into the same dimensionality as the model (dm).
        ffy = self.dense_output(ffy)
        ffy = self.dropout2(ffy, training=training)

        # Apply another residual connection
        return self.layernorm2(xmattentions + ffy)
