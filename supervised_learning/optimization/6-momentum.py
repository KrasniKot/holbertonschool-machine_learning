#!/usr/bin/env python3
""" This module contains create_momentum_op(),
    which performs the training operation for a nn using gc with momentum.
    requires:
        - matplotlib,
        - numpy,
        - tensorflow.
"""

import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """ Trains a nn by gd with momentum.
        - loss: loss of the nn,
        - alpha: learning rate,
        - beta1: momentum weight
    """
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
