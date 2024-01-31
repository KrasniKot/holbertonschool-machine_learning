#!/usr/bin/env python3
""" This module contains train_model(),
    that also analyzes validation data.

    requires:
        - numpy,
        - tensorflow.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """ Also analyzes validation data.
        - network: model to train,
        - data: numpy.ndarray of shape (m, nx), input data,
        - labels: one-hot numpy.ndarray of shape (m, classes), data labels,
        - batch_size: size of the batches used for Mini-Batch GD,
        - epochs: number of passes the model will go through the data,
        - validation_data: data to validate the model wit,
        - verbose: boolean, determines whether to print training information,
        - shuffle: boolean, determines whether to shuffle the data.
    """
    return network.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data)
