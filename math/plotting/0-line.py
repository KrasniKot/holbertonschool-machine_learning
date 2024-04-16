#!/usr/bin/env python3
""" This module contains line()

    requires:
        - numpy,
        - matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """ Plots a red line graph """

    y = np.arange(0, 11) ** 3  
    plt.figure(figsize=(6.4, 4.8))

    plt.xlim(0, 10)
    plt.plot(y, color='red')
#    plt.savefig("plot.png")
    plt.show()
