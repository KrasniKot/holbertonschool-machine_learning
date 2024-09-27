#!/usr/bin/env python3
""" Function that calculates the positional encoding """

import numpy as np


def positional_encoding(max_seq_len, dm):
    """ Calculates the positional encoding for a transformer
        - max_seq_len: integer representing the maximum sequence length
        - dm: model depth
    """
    # Initialize a numpy array of zeros to hold the positional encodings
    positional_encoding = np.zeros((max_seq_len, dm))

    # Get the position indices for the sequence (0 to max_seq_len - 1)
    position = np.arange(max_seq_len)[:, np.newaxis]

    # Get the dimension indices (0 to dm - 1)
    dimension = np.arange(dm)[np.newaxis, :]

    # Calculate the angle rates using the formula (10000^(2i/dm))
    angle_rates = 1 / np.power(10000, (2 * (dimension // 2)) / dm)

    # Apply the sin to even indices and cos to odd indices
    positional_encoding[:, 0::2] = np.sin(position * angle_rates[:, 0::2])
    positional_encoding[:, 1::2] = np.cos(position * angle_rates[:, 0::2])

    return positional_encoding
