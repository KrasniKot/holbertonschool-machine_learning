""" This module contains change_scale()

    requires:
        - numpy,
        - matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """ Plots x ↦ y as a line graph """
    x = np.arange(0, 28651, 5730)  # Generate time values from 0 to 28650 in steps of 5730 (the half-life of C-14)
    r = np.log(0.5)                # Natural logarithm of 0.5 (used in the exponential decay formula)
    t = 5730                       # Half-life of Carbon-14 in years

    #Compute the fraction of Carbon-14 remaining using the exponential decay formula:
    y = np.exp((r / t) * x)  # y = e^((ln(0.5)/half-life) * time)

    # Create a new figure and set the figure size (width=6.4, height=4.8)
    plt.figure(figsize=(6.4, 4.8))

    # Add a title and labels to the x and y axes
    plt.title("Exponential Decay of C-14")
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")

    plt.yscale("log")   # Set y-axis to a logarithmic scale to emphasize the decay trend
    plt.xlim(0, 28650)  # Set x-axis limits from 0 to 28650
    plt.plot(x, y)      # Plot the decay curve

    ######## Save the plot to a PNG file in the "plots" directory and display it on the screen
    plt.savefig("plots/2-change_scale.png")
    plt.show()
    ########


if __name__ == '__main__':
    change_scale()
