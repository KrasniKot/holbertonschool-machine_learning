#!/usr/bin/env python3
""" Have a function flip an image horizontally """

import tensorflow as tf


def flip_image(image):
    """ Flips an image horizontally """
    return tf.image.flip_left_right(image)
