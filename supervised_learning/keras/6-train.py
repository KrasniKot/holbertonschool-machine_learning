#!/usr/bin/env python3
""" This module contains train_model(),
    that also analyzes validation data, performing early stopping.

    requires:
        - numpy,
        - tensorflow.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """ Also analyzes validation data.
        - network: model to train,
        - data: numpy.ndarray of shape (m, nx), input data,
        - labels: one-hot numpy.ndarray of shape (m, classes), data labels,
        - batch_size: size of the batches used for Mini-Batch GD,
        - epochs: number of passes the model will go through the data,
        - validation_data: data to validate the model wit,
        - verbose: boolean, determines whether to print training information,
        - shuffle: boolean, determines whether to shuffle the data,
        - early_stopping: boolean, indicates wheter to early stop the training,
        - patience: early stopping patience.
    """
    callbacks = []

    if early_stopping:
        estg = K.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        callbacks.append(estg)

    return network.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=callbacks)
