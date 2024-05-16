#!/usr/bin/env python3
""" This module contains learning_rate_decay(),
    which creates a learning decay operation using inverse time decay.

    requires:
        - numpy,
        - tensorflow.
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ Creates a learning rate decay operation in tensorflow
        using inverse time decay.
        - alpha: learning rate,
        - decay_rate: weight, determines the rate at which alpha will decay,
        - global_step: number of passes of gradient descent that have elapsed,
        - decay_step: number of passes of gradient descent
            that should occur before alpha is decayed further.
    """
    return tf.train.inverse_time_decay(
            alpha, global_step, decay_step, decay_rate, staircase=True)
