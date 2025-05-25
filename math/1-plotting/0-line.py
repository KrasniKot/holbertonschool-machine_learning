""" This module contains line()

    requires:
        - numpy,
        - matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """ Plots a red line graph """

    y = np.arange(0, 11) ** 3       # Create a list of numbers from 0 **3to 10**3
    plt.figure(figsize=(6.4, 4.8))  # Create a a blank canvas for the plot

    plt.xlim(0, 10)           # Display the x-axis starting at 0 and ending at 10.
    plt.plot(y, color='red')  # Plot the values of the array or list y on the y-axis as a red line
    plt.savefig('plots/0-line.png')
    plt.show()                # Show the plot


if __name__ == '__main__':
    line()
