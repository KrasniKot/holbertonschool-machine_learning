#!/usr/bin/env python3
""" This module contains learning_rate_decay(),
    which updates the learning rate using inverse time decay in numpy.

    requires:
        - numpy,
        - matplotlib.
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ Updates the learning rate using inverse time decay.
        - alpha: learning rate,
        - decay_rate: weight, determines the rate at which alpha will decay.
        - global_step: number of passes of gradient descent that have elapsed
        - decay_step: number of passes of gradient descent
            that should occur before alpha is decayed further
    """
    return alpha / (1 + decay_rate * np.floor(global_step / decay_step))
