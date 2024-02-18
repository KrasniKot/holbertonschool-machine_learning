#!/usr/bin/env python3
""" This module contains resnet50()
    that builds a RestNet-50 network as described in
    "Deep Residual Learning for Image Recognition (2015)"

    requires:
        - tensorflow,

        files:
            - 2-identity_block,
            - 3-projection_block.
"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ Builds a ResNet-50 network and returns the model """
    # Setting relu alias he initialization alias, and input shape
    relu = K.activations.relu
    hn = K.initializers.he_normal()
    ipt = K.Input(shape=(224, 224, 3))

    # First (7, 7) conv layer's activated normalized output
    L0 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), padding='same',
                         strides=(2, 2), kernel_initializer=hn)(ipt)
    anL0 = K.layers.Activation(relu)(K.layers.BatchNormalization(axis=3)(L0))

    # Second maax pooling layer's output
    L1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding='same')(L0)

    # Some projection and identity blocks
    PB0 = projection_block(L1, [64, 64, 256], s=1)
    IB0 = identity_block(PB0, [64, 64, 256])
    IB1 = identity_block(IB0, [64, 64, 256])

    PB1 = projection_block(IB1, [128, 128, 512], s=2)
    IB2 = identity_block(PB1, [128, 128, 512])
    IB3 = identity_block(IB2, [128, 128, 512])
    IB4 = identity_block(IB3, [128, 128, 512])

    PB2 = projection_block(IB4, [256, 256, 1024], s=2)
    IB5 = identity_block(PB2, [256, 256, 1024])
    IB6 = identity_block(IB5, [256, 256, 1024])
    IB7 = identity_block(IB6, [256, 256, 1024])
    IB8 = identity_block(IB7, [256, 256, 1024])
    IB9 = identity_block(IB8, [256, 256, 1024])

    PB3 = projection_block(IB9, [512, 512, 2048], s=2)
    IB10 = identity_block(PB3, [512, 512, 2048])
    IB11 = identity_block(IB11, [512, 512, 2048])

    # Average (7, 7) Pooling layer
    AP0 = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                    padding='valid')(IB11)

    # Final dense layer
    DL = K.layers.Dense(1000, activation='softmax', kernel_initializer=hn)(AP0)

    return K.Model(inputs=ipt, outputs=DL)
