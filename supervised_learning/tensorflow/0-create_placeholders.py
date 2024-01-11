#!/usr/bin/env python3
""" This module contains create_placeholders() """

import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """ Returns two placeholders, x and y
        - nx: the number of feature columns in our data.
        - the number of classes in our classifier.
    """
    x = tf.placeholder(dtype=tf.float32, shape=(None, nx), name="X")
    y = tf.placeholder(dtype=tf.float32, shape=(None, classes), name="Y")

    return x, y
