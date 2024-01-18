#!/usr/bin/env python3
""" This module contains update_variables_RMSProp(),
    which updates a variable using the RMSProp algorithm.

    requires:
        - matplotlib,
        - numpy.
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ Updates a variable using the RMSProp algorithm,
        - alpha: learning rate,
        - beta2: RMSProp weight,
        - epsilon: small number to avoid division by zero,
        - var: numpy.ndarray containing the variable to be updated,
        - grad: numpy.ndarray containing the gradient of var,
        - s: previous second moment of var.
    """
    mom = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    var -= alpha * (grad / (epsilon + (mom ** (1 / 2))))
    return var, mom
