#!/usr/bin/env python3
""" This module contins calculate_loss()
    which returns the loss of a prediction
"""

import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """ Calculates the loss of a prediction using softmax cross-entropy
        - y: placeholder for the labels of the input data (expected result).
        - y_pred: tensor containing the network’s predictions.
    """
    return ts.loss.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
