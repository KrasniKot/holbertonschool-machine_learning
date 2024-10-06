#!/usr/bin/env python3
""" Create masks the encoder and decoder """

import tensorflow as tf


def _build_padded_masks(datas):
    """ Builds a padded mask for the given datas
        - datas ... list containing the different datas to pad

        > Padded mask
    """
    for data in datas:
        # Convert the padding tokens into tf.float32 type
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # Reshape the mask to have the shape (batch_size, 1, 1, seq_length)
        yield seq[:, tf.newaxis, tf.newaxis, :]

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
    # 1. Generate the padding masks and  retrieve them
    mask_generator = _build_padded_masks([inputs, inputs])
    encoder_mask = next(mask_generator)  # Mask for inputs (encoder)
    decoder_mask = next(mask_generator)  # Mask for targets (decoder)

    # 2. Create the look-ahead mask for the target
    seqlen = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seqlen, seqlen)), -1, 0)

    # 3. Expand dimensions of the look-ahead mask for broadcasting
    look_ahead_mask_expanded = _build_padded_masks(target)

    # 4. Ensure combined_mask has compatible shapes
    combined_mask = tf.maximum(look_ahead_mask_expanded, decoder_padding_mask)

    return encoder_mask, combined_mask, decoder_padding_mask
