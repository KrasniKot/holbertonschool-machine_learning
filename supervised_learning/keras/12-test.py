#!/usr/bin/env python3
""" This module contains test_model(),
    that tests a given model:

    requires:
        - tensorflow;
        - numpy.
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ Tests a neural network:
        - network: model to be tested,
        - data: input data to test the model with,
        - labels: correct one-hot labels of data,
        - verbose: boolean that determines if output should be printed
                   during the testing process.
    """
    return network.evaluate(data, labels, verbose=verbose)
