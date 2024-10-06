#!/usr/bin/env python3
""" Create masks the encoder and decoder """

import tensorflow as tf

def create_masks(inputs, target):
    """ Creates masks for the encoder and decoder.

        - inputs .... tensor of shape (batch_size, seq_len_in)
                      representing the input sentences.
        - target .... tensor of shape (batch_size, seq_len_out)
                      representing the target sentences.

        > Padding mask for the encoder, shape (batch_size, 1, 1, seq_len_in).
        > Combined mask for the decoder, shape (batch_size, 1, seq_len_o
        > Padding mask for the decoder, shape (batch_size, 1, 1, seq_len_out).
    """
    # 1. Create the encoder padding mask
    encoder_mask = tf.cast(tf.equal(inputs, 0), tf.float32)  # Shape: (batch_size, seq_len_in)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]  # Shape: (batch_size, 1, 1, seq_len_in)

    # 2. Create the decoder padding mask
    decoder_padding_mask = tf.cast(tf.equal(target, 0), tf.float32)  # Shape: (batch_size, seq_len_out)
    decoder_padding_mask = decoder_padding_mask[:, tf.newaxis, tf.newaxis, :]  # Shape: (batch_size, 1, 1, seq_len_out)

    # 3. Create the look-ahead mask for the target
    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len_out, seq_len_out)), -1, 0)  # Shape: (seq_len_out, seq_len_out)

    # 4. Expand dimensions of the look-ahead mask for broadcasting
    look_ahead_mask_expanded = look_ahead_mask[tf.newaxis, tf.newaxis, :, :]  # Shape: (1, 1, seq_len_out, seq_len_out)

    # 5. Ensure combined_mask has compatible shapes
    combined_mask = tf.maximum(look_ahead_mask_expanded, decoder_padding_mask)  # Shape: (batch_size, 1, seq_len_out, seq_len_out)

    return encoder_mask, combined_mask, decoder_padding_mask
