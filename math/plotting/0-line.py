#!/usr/bin/env python3
""" This module contains line()

    required:
        - matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """ Plots a red line graph """

    y = np.arange(0, 11) ** 3  # Creates an array of the cubes from 0 to 10
    plt.figure(figsize=(6.4, 4.8))  # Sets the size of the array
    plt.plot(y, color='red')  # Plots Y in red

    plt.show()  # Show the graph
