#!/usr/bin/env python3
""" This module contains update_variables_Adam(),
    which updates a variable in place using Adam.

    requires:
        - numpy,
        - matplotlib.
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ Updates a variable in place using Adam.
        - alpha: learnign rate,
        - beta1: weight used for the first moment,
        - beta2: weight used for the second moment,
        - epsilon: small number to prevent division by zero,
        - var: numpy.ndarray containing the variable to be updated,
        - grad: numpy.ndarray containing the gradient of var,
        - v: previous first moment of var,
        - s: previous second moment of var,
        - t: time step used for bias correction.
    """
    mom = (beta1 * v) + ((1 - beta1) * grad)
    mom2 = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    cmom = mom / (1 - (beta1 ** t))
    cmom2 = mom2 / (1 - (beta2 ** t))
    var -= alpha * (cmom / (epsilon + (cmom2 ** (1 / 2))))
    return var, mom, mom2
