#!/usr/bin/env python3
""" Have a function rotate an image """

import tensorflow as tf


def rotate_image(image):
    """ Rotates a given image 90 degrees """
    return tf.image.rot90(image, k=1)