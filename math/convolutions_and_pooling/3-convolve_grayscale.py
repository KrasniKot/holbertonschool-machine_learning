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
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    padw = 0
    padh = 0

    sh = stride[0]
    sw = stride[1]

    if padding == 'same':
        padh = int((((h - 1) * sh + kh - h) / 2) + 1)
        padw = int((((w - 1) * sw + kw - w) / 2) + 1)

    if type(padding) is tuple:
        padh = padding[0]
        padw = padding[1]

    padimg = np.pad(images, pad_width=((0, 0), (padh, padh),
                                       (padw, padw)), mode='constant')

    oh = int(((h + 2 * padh - kh) / sh) + 1)
    ow = int(((w + 2 * padw - kh) / sw) + 1)

    co = np.zeros((m, oh, ow))

    image = np.arange(m)

    for x in range(oh):
        for y in range(ow):
            co[image, x, y] = (np.sum(padimg[image, x * sh:((x * sh) + kh),
                               y * sw:((y * sw) + kw)] * kernel, axis=(1, 2)))

    return co
