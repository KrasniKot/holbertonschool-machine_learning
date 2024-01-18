#!/usr/bin/env python3
""" This module contains update_variables_momentum(),
    which updates a variable using the gd with momentum.
    requires:
        - matplotlib.
        - numpy
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ Updates a variable using the gradient descent with momentum.
        - alpha: learning rate,
        - beta: momentum weight,
        - var: numpy.ndarray containing the variable to update,
        - grad: numpy.ndarray containing the gradient of var,
        - v: previous first moment of var.
    """
    mom = beta1 * v + ((1 - beta1) * var)
    var -= alpha * mom

    return var, mom
