#!/usr/bin/env python3
""" This module contains one_hot(),
    that converts a label vector into a one-hot matrix.

    requires:
        - numpy,
        - tensorflow.
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ Converts a label vector into a one-hot matrix.
    """
    return K.utils.to_categorical(labels, classes)
