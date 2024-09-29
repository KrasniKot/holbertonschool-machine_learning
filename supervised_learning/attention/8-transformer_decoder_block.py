#!/usr/bin/env python3
""" Transformer decoder block """

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """ Defines a DecoderBlock """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ Initializes a DecoderBlock
            - dm ......... dimensionality of the model
            - h .......... number of heads
            - hidden ..... number of hidden units in the fully connected layer
            - drop_rate .. dropout rate
        """
        # Used layers alias
        Dense     = tf.keras.layers.Dense
        LN        = tf.keras.layers.LayerNormalization
        Dout      = tf.keras.layers.Dropout

        super().__init__()

        # Instance attributes
        self.mha1         = MultiHeadAttention(dm, h)  # Autoregressive attention mechanism
        self.mha2         = MultiHeadAttention(dm, h)  # Encoder's output attention mechanism

        self.dense_hidden = Dense(hidden, activation='relu')
        self.dense_output = Dense(dm)
        self.layernorm1   = LN(epsilon=1e-6)
        self.layernorm2   = LN(epsilon=1e-6)
        self.layernorm3   = LN(epsilon=1e-6)
        self.dropout1     = Dout(drop_rate)
        self.dropout2     = Dout(drop_rate)
        self.dropout3     = Dout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """ Returns the decoder's output
            - x ................. tensor of shape (batch, target_seq_len, dm) containing the input to the decoder block
            - encoder_output .... tensor of shape (batch, input_seq_len, dm) containing the output of the encoder
            - training .......... boolean to determine if the model is training
            - look_ahead_mask ... mask to be applied to the first multi head attention layer
            - padding_mask ...... mask to be applied to the second multi head attention layer
        """
        ###### Apply masked attention and dropout to the input x to prevent
        # the model from looking ahead in the target
        # sequence during training and ensuring that each token is generated
        # based only on the context of previously generated tokens,
        # maintaining the integrity of the autoregressive generation process.
        mmattentions, _ = self.mha1(x, x, x, look_ahead_mask)
        mmattentions    = self.dropout1(mmattentions, training=training)
        ######

        # The input x is added to the dropout-modified attention output to
        # apply a skip connection to preserve information from the original input
        xmmattentions = self.layernorm1(x + mmattentions)

        ###### Apply the second attention mechanism to enable the decoder to
        # integrate information from the encoder's output while generating the sequence
        encattentions, _ = self.mha2(xmmattentions, encoder_output, encoder_output, padding_mask)
        encattentions    = self.dropout2(encattentions)
        ######

        # The output mattentions2 is added to the dropout-modified attention
        # output to apply a skip connection to preserve information from the
        # first attention.
        encxmmattentions = self.layernorm2(encattentions + xmmattentions)

        ###### Feed-forward neural network:
        # First layer: includes a non-linear activation function (ReLU)
        # after it to capture non-linearities in the data.
        ffy = self.dense_hidden(encxmmattentions)
        ######

        ###### Project the higher-dimensional representation back to the original
        # dimensionality of the model (dm).
        ffy = self.dense_output(ffy)
        ffy = self.dropout3(ffy, training=training)
        ######

        # The output of the FFNN is added to the skip connection output
        # to apply another skip connection to preserve information from it.
        return self.layernorm3(ffy + encxmmattentions)
