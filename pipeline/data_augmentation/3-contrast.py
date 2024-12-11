#!/usr/bin/env python3
""" Have a function alter the contrast of an image """

import tensorflow as tf


def change_contrast(image, lower, upper):
    """ Randomly change the contrast of an image
        - image .... image whose contrast should be changed
        - lower .... float representing the lower bound of the random contrast factor range
        - upper .... float representing the upper bound of the random contrast factor range
    """
    # Generate a random contrast factor between lower and upper bounds
    contrast_factor = tf.random.uniform([], lower, upper)

    # Adjust the contrast of the image
    return tf.image.adjust_contrast(image, contrast_factor)
