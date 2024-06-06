#!/usr/bin/env python3
""" This module contains letnet5()
    that defines a modified version of LeNet-5 architecture using tensorflow

    requires:
        - tensorflow
"""

from tensorflow import keras as K


def lenet5(X):
    """ Builds a modified version of LeNet-5 architecture using Keras
        - X: contains the input images for the network
            - m: number of images

    """
    conv_1 = K.layers.Conv2D(filters=6, kernel_size=5,
                             padding='same', activation='relu',
                             kernel_initializer='he_normal')(X)
    pool_1 = K.layers.MaxPool2D(2, 2)(conv_1)

    conv_2 = K.layers.Conv2D(filters=16, kernel_size=5,
                             activation='relu',
                             kernel_initializer='he_normal')(pool_1)
    pool_2 = K.layers.MaxPool2D(2, 2)(conv_2)

    flatten = K.layers.Flatten()(pool_2)
    dense_1 = K.layers.Dense(120, input_shape=X.shape,
                             activation='relu',
                             kernel_initializer='he_normal')(flatten)
    dense_2 = K.layers.Dense(84, activation='relu',
                             kernel_initializer='he_normal')(dense_1)
    dense_3 = K.layers.Dense(10, activation='softmax',
                             kernel_initializer='he_normal')(dense_2)

    model = K.Model(inputs=X, outputs=dense_3)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model