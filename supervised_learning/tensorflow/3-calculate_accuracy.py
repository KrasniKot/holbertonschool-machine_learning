#!/usr/bin/env python3
""" This module contains calculate_accuracy()
    which returns the accuarcy of the model
"""

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """ Calculates the accuarcy of the model
        - y: placeholder for the labels of the input data.
        - y_pred: tensor containing the network’s predictions
    """
    correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))

    return tf.reduce_mean(tf.cast(correct, tf.float32))
