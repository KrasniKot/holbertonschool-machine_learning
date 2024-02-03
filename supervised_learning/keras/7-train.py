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
    callbacks = []

    if early_stopping:
        estg = K.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        callbacks.append(estg)

    if validation_data and early_stopping:
        itd = k.optimizers.schedules.InverseTimeDecay(initial_learning_rate, decay_steps, decay_rate, staircase=True)
        model.compile(loss='binary_crossentropy', optimizer=itd, metrics=['accuracy'])  # the loss function is wrong, maybe the whole code

    return network.fit(x=data, y=labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data, callbacks=callbacks)
