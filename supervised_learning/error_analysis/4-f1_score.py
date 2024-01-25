#!/usr/bin/env python3
""" Thi module contains f1_score(),
    that calculates the f1 score of a given confusoin matrix.

    requires:
        - numpy;
        - files:
            - 1-sensitivity.py,
            - 2-precision.py.
"""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ Calculates the f1 score of a confusion matrix.
        - confusion: confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels
            and column indices represent the predicted labels.
    """
    p = precision(confusion)
    r = sensitivity(confusion)
    return 2 * p * r / (p + r)
