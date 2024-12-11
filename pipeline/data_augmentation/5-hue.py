#!/usr/bin/env python3
""" Have a function alter the hue of an image """

import tensorflow as tf


def change_hue(image, delta):
    """ Change the hue of an image """
    return tf.image.adjust_hue(image, delta)
