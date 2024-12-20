#!/usr/bin/env python3
""" Have a function alter the brightness of an image """

import tensorflow as tf


def change_brightness(image, max_delta):
    """ Change the brightness of an image """
    return tf.image.random_brightness(image=image, max_delta=max_delta)
