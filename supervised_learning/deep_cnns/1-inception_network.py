#!/usr/bin/env python3
""" This module contains inception_network(),
    that builds an incpetion network as described in:
    "Going Deeper with Convolutions (2024)"

    requires:
        - tensorflow.
        - 0-inception_block
"""

from tensorflow import keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ Builds an inception netowrk and returns the model """
    # Setting some common parameteres, activation function alias, input shape
    relu = K.activations.relu
    pms = {"padding": "same", "activation": relu,
           "kernel_initializer": K.initializers.he_normal()}
    mxpms = {"pool_size": (3, 3), "strides": (2, 2), "padding": "same"}
    ipt = K.Input(shape=(224, 224, 3))

    # First (7, 7) conv layer's output
    L0 = K.layers.Conv2D(filters=64, strides=(2, 2),
                         kernel_size=(7, 7), **pms)(ipt)

    # Second (3, 3) max pooling layer's output
    L1 = K.layers.MaxPooling2D(**mxpms)(L0)

    # Bottleneck layer
    L2 = K.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1),
                         **pms)(L1)

    # Third (3, 3) conv layer's output
    L2 = K.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1),
                         **pms)(L2)

    # Fourth (3, 3) max pooling layer's output
    L3 = K.layers.MaxPooling2D(**mxpms)(L2)

    # Inception blocks
    IB0 = inception_block(L3, [64, 96, 128, 16, 32, 32])
    IB1 = inception_block(IB0, [128, 128, 192, 32, 96, 64])

    # Fifth (3, 3) max pooling layer's output
    L4 = K.layers.MaxPooling2D(**mxpms)(IB1)

    # More inception blocks...
    IB2 = inception_block(L4, [192, 96, 208, 16, 48, 64])
    IB3 = inception_block(IB2, [160, 112, 224, 24, 64, 64])
    IB4 = inception_block(IB3, [128, 128, 256, 24, 64, 64])
    IB5 = inception_block(IB4, [112, 144, 288, 32, 64, 64])
    IB6 = inception_block(IB5, [256, 160, 320, 32, 128, 128])

    # Sixth (3, 3) max pooling layer's output
    L6 = K.layers.MaxPooling2D(**mxpms)(IB6)

    # Some more inception...
    IB7 = inception_block(L6, [256, 160, 320, 32, 128, 128])
    IB8 = inception_block(IB7, [384, 192, 384, 48, 128, 128])

    # Average pooling layer
    AP0 = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                    padding='valid')(IB8)

    # Dropout layer
    DP0 = K.layers.Dropout(rate=0.4)(AP0)

    # Dense layer
    D0 = K.layers.Dense(1000, activation='softmax',
                        kernel_initializer=K.initializers.he_normal())(DP0)

    return K.Model(inputs=ipt, outputs=D0)
