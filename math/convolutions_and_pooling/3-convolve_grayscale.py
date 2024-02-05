#!/usr/bin/env python3
""" This module contains convolve_grayscale()
    that performs a convolution on grayscale images:

    requires:
        - numpy.
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """ Performs a convolution on grayscale image with with a custom stride:
        - images: np.ndarray with shape (m, h, w) containing grayscale images,
            - m: number of images;
            - h: height in pixels of the images;
            - w: width in pixels of the images;
        - kernel: np.ndarray with shape (kh, kw) containing the kernel to use,
            - kh height of the kernel;
            - kw: width of the kernel.
        - padding: either a tuple of (ph, pw), 'same', or 'valid',
        - stride: tuple of (sh, sw);
            - sh: stride for the height of the image;
            - sw: stride for the width of the image.
    """
    m = images.shape[0]
    ih = images.shape[1]
    iw = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    sh = stride[0]
    sw = stride[1]

    if padding == "same":
        padh = int(((ih - 1) * sh - ih + kh) / 2)
        padw = int(((iw - 1) * sw - iw + kw) / 2)
    elif padding == "valid":
        padh = padw = 0
    else:
        padh, padw = stride

    oh = int(((ih + (2 * padh) - kh) / stride[0]) + 1)
    ow = int(((iw + (2 * padw) - kw) / stride[1]) + 1)

    co = np.zeros([m, oh, ow])
    imgpd = np.pad(images, pad_width=((0, 0), (padw, padw), (padh, padh)),
                   mode='constant', constant_values=0)

    for i in range(oh):
        for j in range(ow):
            img = imgpd[:, i * sh: i * sh + kh, j * sw: j * sw + kw]
            co[:, i, j] = np.multiply(img, kernel).sum(axis=(1, 2))

    return co

