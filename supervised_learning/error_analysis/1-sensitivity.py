#!/usr/bin/env python3
""" This module contains sensitivity(),
    that calculates the sensitivity of a given confusion matrix.

    requires:
        - numpy.
"""

import numpy as np


def sensitivity(confusion):
    """ Calculates the sensitivity of a confusion matrix.
        - confusion: confusion np.ndarray of shape (classes, classes),
            where row indices represent the correct labels and
            column indices represent the predicted labels.
    """

    # since sens = tp / fn ∧ fn = sum(conf., axis=1) - tp,
    # ∴ sens ≡ tp / sum(conf., axis=1)
    return np.diag(confusion) / np.sum(confusion, axis=1)
