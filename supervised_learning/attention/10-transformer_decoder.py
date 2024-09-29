#!/usr/bin/env python3
""" Creates an Decoder for a Transformer """

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """ Defines a Decoder """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
        """ Initializes a Decoder
            - dm ............... dimensionality of the model
            - h ................ number of heads
            - hidden ........... number of hidden units in the fully connected layer
            - target_vocab ..... size of the target vocabulary
            - max_seq_len ...... maximum sequence length possible
            - drop_rate ........ dropout rate
        """
        # Aliases
        Dout  = tf.keras.layers.Dropout
        Emddg = tf.keras.layers.Embedding
        hdn   = hidden

        super().__init__()

        self.N                   = N
        self.dm                  = dm
        self.embedding           = Emddg(input_dim=target_vocab, output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks              = [DecoderBlock(dm, h, hdn, drop_rate) for _ in range(N)]
        self.dropout             = Dout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """ Returns the decoder output
            - x ................. tensor of shape (batch, target_seq_len, dm) containing the input to the decoder
            - encoder_output .... tensor of shape (batch, input_seq_len, dm) containing the output of the encoder
            - training .......... boolean to determine if the model is training
            - look_ahead_mask ... mask to be applied to the first multi head attention layer
            - padding_mask ...... mask to be applied to the second multi head attention layer
        """
        # Extract the length of the input sequence from the shape of the tensor
        input_seq_len = x.shape[1]

        # Convert each token into a high-dimensional vector
        x = self.embedding(x)  # Shape -> (batch, input_seq_len, dm)

        ###### Positional encoding and scaling
        # Scaling: values are scaled by sqrt(dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))

        # This operation combines the learned content of each token
        # (from the embedding) with information about its position in the
        # sequence (from the positional encoding).
        x += self.positional_encoding[:input_seq_len, :]

        # Apply dropout to the positional encoding
        x = self.dropout(x, training=training)
        ######

        # The input is passed through N encoder blocks where each block processes x further
        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training, look_ahead_mask, padding_mask)

        return x
