#!/usr/bin/env python3
""" Creates a Transformer Model """

import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.layers.Layer):
    """ Defines a Transformer Model """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab, max_seq_input, max_seq_target,
                 drop_rate=0.1):
        """ Initializes a Transformer Model
            - N .............. number of blocks in the encoder and decoder
            - dm ............. dimensionality of the model
            - h .............. number of heads
            - hidden ......... number of hidden units in the fully connected layers
            - input_vocab .... size of the input vocabulary
            - target_vocab ... size of the target vocabulary
            - max_seq_input .. maximum sequence length possible for the input
            - max_seq_target . maximum sequence length possible for the target
            - drop_rate ...... dropout rate
        """
        super().__init__()

        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask, decoder_mask):
        """ Returns the transformer's output
            - inputs .......... a tensor of shape (batch, input_seq_len) containing the inputs
            - target .......... a tensor of shape (batch, target_seq_len) containing the target
            - training ........ a boolean to determine if the model is training
            - encoder_mask .... the padding mask to be applied to the encoder
            - look_ahead_mask . the look ahead mask to be applied to the decoder
            - decoder_mask .... the padding mask to be applied to the decoder
        """
        # Encodes the input
        x = self.encoder(inputs, training, encoder_mask)

        # Decodes the context vector
        x = self.decoder(target, x, training, look_ahead_mask, decoder_mask)

        return self.linear(x)
