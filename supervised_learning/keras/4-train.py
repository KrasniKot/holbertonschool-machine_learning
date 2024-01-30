#!/usr/bin/env python3
""" This module contains train_model(),
    that trains a model using Mini-Batch GD.

    requires:
        - numpy,
        - tensorflow.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """ Trains a model using mini-batch gradient descent.
        - network: model to train,
        - data: numpy.ndarray of shape (m, nx) containing the input data,
        - labels: one-hot numpy.ndarray of shape (m, classes)
            containing the labels of data
        - epochs: number of passes through data
            for mini-batch gradient descent,
        - verbose: boolean that determines whether output should be printed,
        - shuffle: boolean that determines
            whether to shuffle the batches every epoch.
    """
    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose, shuffle=shuffle)
