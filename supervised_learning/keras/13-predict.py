#!/usr/bin/env python3
""" This module contains predict(),
    that makes a prediction:

    requires;
        - tensorflow;
        - numpy.
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ Makes a prediction by a given model:
        - network: model to make the prediction with,
        - data: input data to make the prediction with,
        - verbose: boolean, determines if output should be printed
                   during the prediction process.
    """
    return network.predict(data, verbose=0 if not verbose else 1)
