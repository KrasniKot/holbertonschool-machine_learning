#!/usr/bin/env python3
""" This module contains letnet5()
    that defines a modified version of LeNet-5 architecture using tensorflow

    requires:
        - tensorflow
"""

import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """ Builds a modified version of LeNet-5 architecture using TensorFlow
        - x: contains the input images for the network
            - m: number of images
        - y: contains the one-hot labels for the network
    """
    wini = tf.variance_scaling_initializer()
    C1 = tf.layers.Conv2D(filters=6,
                          kernel_size=(5, 5),
                          padding='same',
                          activation=tf.nn.relu,
                          kernel_initializer=wini)(x)

    P1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C1)

    C2 = tf.layers.Conv2D(filters=16,
                          kernel_size=(5, 5),
                          padding='valid',
                          activation=tf.nn.relu,
                          kernel_initializer=wini)(P1)

    P2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C2)
    flatten = tf.layers.Flatten()(P2)

    F = tf.layers.Dense(
       120,
       activation=tf.nn.relu,
       kernel_initializer=wini)(flatten)

    F1 = tf.layers.Dense(84, activation=tf.nn.relu, kernel_initializer=wini)(F)

    F2 = tf.layers.Dense(10, kernel_initializer=wini)(F1)

    smax = tf.nn.softmax(F1)
    cost = tf.losses.softmax_cross_entropy(y, logits=F2)
    op = tf.train.AdamOptimizer().minimize(cost)
    ypred = tf.math.argmax(F2, axis=1)
    yout = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(ypred, yout)
    accuracy = tf.reduce_mean(tf.cast(equality, 'float'))

    return smax, op, cost, accuracy
