#!/usr/bin/env python3
""" This module contains create_train_op()
    which trains the nn
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """ Creates the training operation for the nn.
     - loss: loss of the network’s prediction.
     - alpha: learning rate.
    """

    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
