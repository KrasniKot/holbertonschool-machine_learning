#!/usr/bin/env python3
""" This module contains convolve_grayscale_same()
    that performs same convolution on grayscale images:

    requires:
        - numpy.
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """ Performs same convolution on grayscale images:
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

    padh = int(kh / 2)
    padw = int(kw / 2)

    co = np.zeros([m, oh, ow])
    imgpd = np.pad(images, pad_width=((0, 0), (padh, padh), (padw, padw)),
                   mode='constant', constant_values=0)

    for x in range(h):
        for y in range(w):
            img = imgpd[:, x: kh+x, y: kw+y]
            co[:, x, y] = np.multiply(img, kernel).sum(axis=(1, 2))

    return co
