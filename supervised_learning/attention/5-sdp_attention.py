#!/usr/bin/env python3
""" Function that calculates the SDP Attention """

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """ Calculates the Scaled Dot-Product Attention
        - Q: tensor with its last two dimensions as (..., seq_len_q, dk)
             containing the query matrix
        - K: tensor with its last two dimensions as (..., seq_len_v, dk)
             containing the key matrix
        - V: tensor with its last two dimensions as (..., seq_len_v, dv)
             containing the value matrix
        - mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
                containing the optional mask
    """
    # Matrix multiplication of Q by transposed K
    dk = tf.cast(Q.shape[-1], dtype=tf.float32)
    scores = tf.matmul(Q, K, transpose_b=True)

    # Scale scores by sqrt(dk)
    scaleds = scores / tf.sqrt(dk)

    # Add mask
    if mask is not None:
        scaleds += mask * -1e9

    # Apply softmax to get attention weights
    attention_weights = tf.nn.softmax(scaleds, axis=-1)

    return tf.matmul(attention_weights, V), attention_weights
