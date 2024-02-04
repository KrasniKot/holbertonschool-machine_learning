#!/usr/bin/env python3
""" This module contains convolve_grayscale_valid()
    that performs convolution on grayscale images with custom padding:

    requires:
        - numpy.
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ Performs convolution on grayscale images with custom padding:
        - images: np.ndarray with shape (m, h, w) containing grayscale images,
            - m: number of images;
            - h: height in pixels of the images;
            - w: width in pixels of the images;
        - kernel: np.ndarray with shape (kh, kw) containing the kernel to use,
            - kh height of the kernel;
            - kw: width of the kernel.
        - padding: tuple of (ph, pw),
            - ph: padding for the height of the image;
            - pw: padding for the width of the image.
    """
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]

    padh = padding[0]
    padw = padding[1]

    oh = h + 2 * padh - kh + 1
    ow = w + 2 * padw - kw + 1

    co = np.zeros([m, oh, ow])
    imgpd = np.pad(images, pad_width=((0, 0), (padh, padh), (padw, padw)),
                   mode='constant', constant_values=0)

    for x in range(oh):
        for y in range(ow):
            img = imgpd[:, x: kh+x, y: kw+y]
            co[:, x, y] = np.multiply(img, kernel).sum(axis=(1, 2))

    return co
