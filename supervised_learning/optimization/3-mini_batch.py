#!/usr/bin/env python3
""" This module contains create_mini_batches(),
    which trains a nn using mini_batch gradient descent.

    requires:
        - numpy,
        - tensoflow.
"""
import numpy as np

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """ Creates mini-batches from the input data for mini-batch gd.
        - X (numpy.ndarray): Input data of shape (m, nx)
        - Y (numpy.ndarray): Labels of shape (m, ny)
        - batch_size (int): Number of data points in each mini-batch
    """
    mini_batches = []
    m = X.shape[0]

    Xs, Ys = shuffle_data(X, Y)  # Shuffle to improve generalization
    minibtchesNum = m // batch_size  # Number of final mini-batches

    for i in range(minibtchesNum):
        X_batch = Xs[i * batch_size: (i + 1) * batch_size]
        Y_batch = Ys[i * batch_size: (i + 1) * batch_size]
        mini_batches.append((X_batch, Y_batch))

    # Handle the case where the final batch may be smaller than batch_size
    if m % batch_size != 0:
        X_batch = Xs[minibtchesNum * batch_size:]
        Y_batch = Ys[minibtchesNum * batch_size:]

        mini_batches.append((X_batch, Y_batch))

    return mini_batches
