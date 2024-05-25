#!/usr/bin/env python3
""" This module contains l2_reg_cost(),
    that calculates the cost of a neural network with L2 regularization.

    requires:
        - numpy.
        - tensorflow.
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    - cost: a tensor containing the cost of the network without L2 regularization
    - model: a Keras model that includes layers with L2 regularization
    """
    return cost + tf.add_n(model.losses)
