#!/usr/bin/env python3
""" This module contains specificity(),
    that calculates the specificity of a given confusion matrix.

    requires: numpy.
"""

import numpy as np


def specificity(confusion):
    """ Calculates the specificity of a confusion matrix.
        - confusion: confusion np.ndarray of shape (classes, classes),
            where row indices represent the correct labels
            and column indices represent the predicted labels.
    """
    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp
    tn = np.sum(confusion) - (tp + fp + fn)

    return tn / (tn + fp)
