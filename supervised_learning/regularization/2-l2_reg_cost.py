#!/usr/bin/env python3
""" This module contains l2_reg_cost(),
    that calculates the cost of a neural network with L2 regularization.

    requires:
        - numpy.
        - tensorflow.
"""

import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """ Calculates the cost of a nn with L2 regularization.
        - cost: tensor containing the cost of the network
            without L2 regularization.
    """
    return cost + tf.losses.get_regularization_losses()
