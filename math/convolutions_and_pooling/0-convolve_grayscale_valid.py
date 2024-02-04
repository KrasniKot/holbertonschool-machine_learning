#!/usr/bin/env python3
""" This module contains convolve_grayscale_valid()
    that performs valid convolution on grayscale images:

    requires:
        - numpy.

"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """ Performs valid convolution on grayscale images:
        - images: np.ndarray with shape (m, h, w) containing grayscale images,
            - m: number of images;
            - h: height in pixels of the images;
            - w: width in pixels of the images;
        - kernel: np.ndarray with shape (kh, kw) containing the kernel to use,
            - kh height of the kernel;
            - kw: width of the kernel.
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    oh = h - kh + 1
    ow = w - kw + 1

    co = np.zeros([m, oh, ow])

    for x in range(oh):
        for y in range(ow):
            image = images[:, x:kh+x, y:kw+y]
            oc[:, x, y] = np.multiply(image, kernel).sum(axis=(1, 2))

    return co
