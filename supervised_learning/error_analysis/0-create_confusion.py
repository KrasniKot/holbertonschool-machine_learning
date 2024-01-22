#!/usr/bin/env python3
""" This module contains create_confusion_matrix(),
    that creates a confusion matrix.

    requires:
        - numpy.
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """ Creates a confusion matrix
        - labels: one-hot np.ndarray of shape (m, classes),
            containing the correct labels.
        - logits: one-hot np.ndarray of shape (m, classes),
            containing the predicted labels.
    """
    return labels.T @ logits
