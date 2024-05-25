#!/usr/bin/env python3
""" This module contains ...
"""

def early_stopping(cost, opt_cost, threshold, patience, count):
    """ Determines if you should stop gradient descent early

     - cost: Current validation cost of the NN
     - opt_cost: Lowest recorded validation cost of the NN
     - threshold: Threshold used for early stopping
     - patience: Patience count used for early stopping
     - count: Count of how long the threshold has not been met
    """
    count = 0 if (opt_cost - cost) > threshold else count + 1
    should = True if count == patience else False

    return should, count