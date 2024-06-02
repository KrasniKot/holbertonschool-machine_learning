#!/usr/bin/env python3
""" This module contains train_model(),
    that also trains the model with learning rate deacay.

    requires:
        - numpy,
        - tensorflow.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
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
        - patience: early stopping patience,
        - learning_rate_decay:
            boolean, indicates whether learning rate decay shuld be used.
        - alpha: initial learning rate.
        - decay_rate: decay rate.
    """
    def calculate_alpha(epoch):
        """ Returns the learning rate for an epoch
            - epoch: umber of passes through data for mini-batch GD
        """
        return alpha / (1 + decay_rate * epoch)

    callbacks = []
    ES = K.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                   patience=patience)
    LRD = K.callbacks.LearningRateScheduler(calculate_alpha, verbose=1)

    if validation_data and early_stopping:
        callbacks.append(ES)
    if validation_data and learning_rate_decay:
        callbacks.append(LRD)

    return network.fit(x=data, y=labels, batch_size=batch_size,
                       epochs=epochs, validation_data=validation_data,
                       callbacks=callbacks,
                       verbose=verbose, shuffle=shuffle)
