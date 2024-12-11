#!/usr/bin/env python3
""" Have a function crop an image """

import tensorflow as tf


def crop_image(image, size):
    """ Crops an image to a given size"""
    return tf.image.random_crop(image, size)
