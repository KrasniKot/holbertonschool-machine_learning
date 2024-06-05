#!/usr/bin/env python3
""" This module contains letnet5()
    that defines a modified version of LeNet-5 architecture using tensorflow

    requires:
        - tensorflow
"""

import tensorflow.keras as K


def lenet5(X):
    """ Builds a modified version of LeNet-5 architecture using Keras
        - X: contains the input images for the network
            - m: number of images

    """
    wini = K.initializers.he_normal()
    C1 = K.layers.Conv2D(filters=6,
                         kernel_size=(5, 5),
                         padding='same',
                         activation=K.activations.relu,
                         kernel_initializer=wini)(X)

    P2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2))(C1)

    C3 = K.layers.Conv2D(filters=16,
                         kernel_size=(5, 5),
                         padding='valid',
                         activation=K.activations.relu,
                         kernel_initializer=wini)(P2)

    P4 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C3)
    flatten = K.layers.Flatten()(P4)

    F5 = K.layers.Dense(
        120,
        activation=K.activations.relu,
        kernel_initializer=wini)(flatten)

    F6 = K.layers.Dense(
        84,
        activation=K.activations.relu,
        kernel_initializer=wini)(F5)

    F7 = K.layers.Dense(10, kernel_initializer=wini)(F6)

    smax = K.layers.Softmax()(F7)
    model = K.Model(inputs=X, outputs=smax)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
