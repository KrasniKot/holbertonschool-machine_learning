#!/usr/bin/env python3
""" This module contains precision(),
    that calculates the presicion of a given confusion matrix

    requires:
        - numpy.
"""

import numpy as np


def precision(confusion):
    """ Calculates the precision of a confusion matrix
        - confusion: np.ndarray of shape (classes, classes),
            where row indices represent the correct labels
            and column indices represent the predicted labels.
    """
    # since pre = tp / fn ∧ fn = sum(conf., axis=0) - tp,
    # ∴ pre ≡ tp / sum(conf., axis=0)
    return np.diag(confusion) / np.sum(confusion, axis=0)
